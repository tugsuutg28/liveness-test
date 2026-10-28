"""Opt-in event logs contain statuses only, no images or identities."""
import json
from datetime import datetime,timezone
from pathlib import Path

def log_event(kind,status,reason,path='logs/events.jsonl'):
    file=Path(path);file.parent.mkdir(parents=True,exist_ok=True)
    entry={'timestamp':datetime.now(timezone.utc).isoformat(),'event':str(kind),
           'status':str(status),'reason':str(reason)}
    with file.open('a') as f:f.write(json.dumps(entry)+'\n')
