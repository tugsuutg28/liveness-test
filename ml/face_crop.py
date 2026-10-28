"""Square face cropping with configurable boundary padding."""

import cv2


def crop_face(
    frame,
    bbox,
    padding=0.60,
    border_type=cv2.BORDER_REPLICATE,
):
    height, width = frame.shape[:2]
    x, y, w, h = map(int, bbox)

    if w <= 0 or h <= 0:
        return None

    cx = x + w / 2
    cy = y + h / 2

    side = max(1, round(max(w, h) * (1 + 2 * padding)))

    left = round(cx - side / 2)
    top = round(cy - side / 2)
    right = left + side
    bottom = top + side

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)

    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        border_type,
        value=(127, 127, 127),
    )

    left += pad_left
    right += pad_left
    top += pad_top
    bottom += pad_top

    return padded[top:bottom, left:right]