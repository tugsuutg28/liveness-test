"""Collect full camera frames into explicitly chosen train/val/test sessions.

Example (opens the camera only when invoked):
    python ml/collect_liveness_v2.py --split train --label real --session day1 \
        --lighting daylight --background office --device built-in

Press S for a three-second countdown, Q to stop. Use separate subjects/sessions
across splits to avoid leakage. Existing data is never moved or overwritten.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'datasets' / 'liveness_v2'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True, choices=('train', 'val', 'test'))
    parser.add_argument('--label', required=True, choices=('real', 'fake'))
    parser.add_argument('--session', required=True)
    for field in ('lighting', 'background', 'device'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--attack-type', help='Required for fake; real uses none')
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', args.session):
        parser.error('--session must be 1–100 letters, digits, underscores or hyphens')
    if args.count <= 0 or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--count and --interval must be finite and positive')
    for field in ('lighting', 'background', 'device'):
        if not getattr(args, field).strip():
            parser.error('--' + field + ' must not be blank')
    if args.label == 'fake' and (not args.attack_type or args.attack_type.strip().lower() == 'none'):
        parser.error('--attack-type is required for fake images')
    if args.attack_type is not None and not args.attack_type.strip():
        parser.error('--attack-type must not be blank')
    if args.label == 'real' and args.attack_type not in (None, 'none'):
        parser.error('real images must use attack type none')
    args.attack_type = args.attack_type or 'none'
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def collect(args, *, cv=None, clock=time.monotonic):
    # Importing this module or asking for --help never initializes a camera.
    if cv is None:
        import cv2 as cv
    output = args.output_root / args.split / args.label / args.session
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # Atomic reservation; reject even empty sessions.
    metadata = {key: getattr(args, key) for key in (
        'split', 'label', 'session', 'lighting', 'background', 'device',
        'attack_type', 'count', 'interval', 'camera')}
    metadata.update(started_at=utc_now(), saved_count=0, status='initializing', countdown_seconds=3)
    metadata_path = output / 'metadata.json'
    with metadata_path.open('x') as handle:
        json.dump(metadata, handle, indent=2)
    camera = None
    saved = 0
    try:
        camera = cv.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera')
        print(f'Saving to {output}. Press S to start countdown; Q to quit.')
        deadline = None
        next_save = None
        metadata['status'] = 'cancelled'
        while saved < args.count:
            success, frame = camera.read()
            if not success:
                raise RuntimeError('Camera read failed')
            now = clock()
            display = frame.copy()
            text = f'{args.split}/{args.label}: {saved}/{args.count}'
            if deadline is not None and now < deadline:
                text += f' | Starting in {math.ceil(deadline - now)}'
            elif deadline is None:
                text += ' | S: start, Q: quit'
            cv.putText(display, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            cv.imshow('Liveness Collection v2', display)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and deadline is None:
                deadline = clock() + 3
                next_save = deadline
            if next_save is not None and now >= next_save:
                success, encoded = cv.imencode('.jpg', frame)
                if not success:
                    raise RuntimeError('Could not encode image')
                # Exclusive creation also guards against unexpected filename collisions.
                with (output / f'frame_{saved:04d}.jpg').open('xb') as handle:
                    handle.write(encoded.tobytes())
                saved += 1
                next_save = now + args.interval
        if saved == args.count:
            metadata['status'] = 'complete'
    except BaseException:
        metadata['status'] = 'error'
        raise
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            try:
                cv.destroyAllWindows()
            finally:
                metadata.update(saved_count=saved, finished_at=utc_now())
                metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return output


def main(argv=None):
    collect(parse_args(argv))


if __name__ == '__main__':
    main()
