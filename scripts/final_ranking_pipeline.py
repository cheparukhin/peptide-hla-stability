"""Single durable owner for at most two local ranking component workers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pepstab.final_ranking import REPRESENTATIONS,atomic_json
from scripts.final_ranking import OUT,require_authorization


def run(out=OUT):
    out=Path(out)
    protocol=require_authorization(out,"scientific")
    gate=json.loads((out/"gate.json").read_text())
    deadline=datetime.fromisoformat(gate["scientific_deadline_utc"]).timestamp()
    if deadline<=time.time()+30:
        raise ValueError("fixed local deadline expired or lacks shutdown cushion")
    workers=int(gate.get("max_workers",2))
    if workers not in (1,2):
        raise ValueError("frozen allocation is at most two workers")
    receipt=out/"pipeline.launched.json"
    with receipt.open("x") as handle:
        json.dump(dict(status="launched",pid=os.getpid(),protocol_hash=protocol["protocol_hash"],
            deadline_utc=gate["scientific_deadline_utc"],max_workers=workers),handle)
    queue=[(rep,fold) for fold in range(5) for rep in REPRESENTATIONS]
    active=[]
    finished=[]
    try:
        while queue or active:
            if time.time()>=deadline-15:
                raise TimeoutError("fixed local deadline reached; shutdown cushion preserved")
            while queue and len(active)<workers:
                rep,fold=queue.pop(0)
                log=out/f"worker_{rep}_fold{fold}.log"
                stream=log.open("x")
                env=dict(os.environ,OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                         VECLIB_MAXIMUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
                process=subprocess.Popen([sys.executable,str(ROOT/"scripts/final_ranking.py"),"fit",
                    "--out",str(out),"--representation",rep,"--fold",str(fold)],
                    stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active.append((rep,fold,process,stream))
                atomic_json(out/"pipeline.state.json",dict(active=[dict(representation=r,fold=f,pid=p.pid) for r,f,p,s in active],
                    pending=queue,completed=finished,deadline_utc=gate["scientific_deadline_utc"]))
            for rep,fold,process,stream in list(active):
                code=process.poll()
                if code is None:
                    continue
                stream.close()
                active.remove((rep,fold,process,stream))
                atomic_json(out/f"worker_{rep}_fold{fold}.exit.json",dict(returncode=code,pid=process.pid,
                    representation=rep,outer_fold=fold,protocol_hash=protocol["protocol_hash"]))
                if code:
                    raise RuntimeError(f"{rep} fold{fold} worker failed; no automatic retry")
                finished.append((rep,fold))
            time.sleep(.5)
        atomic_json(out/"pipeline.completed.json",dict(status="complete",protocol_hash=protocol["protocol_hash"],
            components=finished,new_scientific_fits=840,completed_at_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        for rep,fold,process,stream in active:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        for rep,fold,process,stream in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            stream.close()
            atomic_json(out/f"worker_{rep}_fold{fold}.exit.json",dict(returncode=process.returncode,
                pid=process.pid,representation=rep,outer_fold=fold,terminated=True))
        atomic_json(out/"pipeline.exit.json",dict(status="failed",error=repr(error),completed=finished))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    run(args.out)
