"""One-owner, guarded Modal batch dispatch for the frozen final transfer scope.

Never creates an app, retries a call, or recovers an uncertain reservation.
The caller owns an already-open app and a fresh ReservationBudget journal.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


RATE = .1266 / 3600
OVERHEAD_SECONDS = 62
TIMEOUT_SECONDS = 1000
CALL_UPPER = RATE * (TIMEOUT_SECONDS + OVERHEAD_SECONDS)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


class GuardedBatchBackend:
    def __init__(self, function, budget, authorization_path, receipts, *, max_batch=6):
        self.function, self.budget = function, budget
        self.authorization_path = Path(authorization_path)
        raw = self.authorization_path.read_bytes()
        self.authorization_hash = hashlib.sha256(raw).hexdigest()
        self.authorization = json.loads(raw)
        self.receipts, self.max_batch = Path(receipts), max_batch
        if self.budget.status()['calls']:
            raise ValueError('new backend requires reconciled fresh journal; no automatic recovery')

    def check_guard(self):
        auth = self.authorization
        if hashlib.sha256(self.authorization_path.read_bytes()).hexdigest() != self.authorization_hash:
            raise ValueError('guard authorization changed')
        if Path(auth['stop_flag_path']).exists():
            raise RuntimeError('independent guard froze new submissions')
        status = json.loads(Path(auth['status_path']).read_text())
        if status.get('authorization_sha256') != self.authorization_hash or status.get('phase') != 'monitoring':
            raise RuntimeError('matching healthy independent guard required')
        stamp = datetime.fromisoformat(status['updated_at_utc']).timestamp()
        if not 0 <= time.time() - stamp <= 25:
            raise RuntimeError('independent guard heartbeat is stale or future-dated')
        pid = status['guard_pid']
        if type(pid) is not int or pid <= 1:
            raise RuntimeError('invalid guard PID')
        os.kill(pid, 0)
        if time.time() >= datetime.fromisoformat(auth['deadline_utc']).timestamp() - 30:
            raise RuntimeError('fixed spending deadline shutdown margin reached')
        return status

    def __call__(self, tasks):
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= self.max_batch:
            raise ValueError('batch exceeds frozen dispatch size')
        ids = [task['task_id'] for task in tasks]
        if len(set(ids)) != len(ids) or any(not isinstance(i, str) or len(i) != 64 or any(c not in '0123456789abcdef' for c in i) for i in ids):
            raise ValueError('unique SHA256 task identities required')
        self.check_guard()
        # Exclusive durable intent prevents repeated dispatch even before a
        # reservation is created. A partial batch intent needs manual review.
        for task in tasks:
            write_once(self.receipts / (task['task_id'] + '.launched.json'),
                       {'task_id': task['task_id'], 'candidate': task['candidate'],
                        'seed': task['seed'], 'regime': task['regime'],
                        'row_counts': {k: len(task[k]) for k in ('gradient', 'stopping', 'held')},
                        'authorization_sha256': self.authorization_hash,
                        'at_utc': datetime.now(timezone.utc).isoformat()})
        token = self.budget.reserve(len(tasks), context=','.join(ids))
        try:
            self.check_guard()
            results = list(self.function.map(tasks, return_exceptions=True))
            if len(results) != len(tasks):
                raise ValueError('remote batch cardinality changed')
            charged = 0.
            for task, result in zip(tasks, results):
                if isinstance(result, BaseException):
                    raise RuntimeError('remote member failed; entire batch charged at reservation') from result
                if not isinstance(result, dict) or result.get('task_id') != task['task_id']:
                    raise ValueError('remote task identity mismatch')
                if result.get('pair_ids') != [row['pair_id'] for row in task['held']]:
                    raise ValueError('remote prediction row identity mismatch')
                values = np.asarray(result['predictions'])
                if values.shape != (len(task['held']),) or not np.isfinite(values).all():
                    raise ValueError('invalid remote predictions')
                seconds = result['worker_seconds']
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= TIMEOUT_SECONDS:
                    raise ValueError('remote metering exceeds frozen bound')
                charged += (seconds + OVERHEAD_SECONDS) * RATE
            for task, result in zip(tasks, results):
                write_once(self.receipts / (task['task_id'] + '.completed.json'), result)
            self.budget.settle(token, charged, details={'task_ids': ids, 'worker_seconds': [r['worker_seconds'] for r in results]})
            return results
        except BaseException as error:
            # A dispatch/result/receipt failure can hide successful remote work.
            # Preserve uncertainty conservatively; never call the batch again.
            if token in self.budget.active:
                self.budget.burn(token, outcome='uncertain_or_failed_batch_full_reservation', details={'error': repr(error), 'task_ids': ids})
            raise
