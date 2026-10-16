import os, json, time
from pathlib import Path
from datetime import datetime, timezone
p=Path(__file__).parent
with (p/"smoke_events.jsonl").open("a") as f:
 f.write(json.dumps({"event":"started","pid":os.getpid(),"ppid":os.getppid(),"at":datetime.now(timezone.utc).isoformat()})+"\n")
 f.flush(); os.fsync(f.fileno())
time.sleep(2)
with (p/"smoke_exit.json").open("x") as f:
 json.dump({"pid":os.getpid(),"ppid":os.getppid(),"intentional_exit_code":7,"at":datetime.now(timezone.utc).isoformat()},f)
 f.flush(); os.fsync(f.fileno())
raise SystemExit(7)
