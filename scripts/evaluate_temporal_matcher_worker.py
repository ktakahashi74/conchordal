"""M0 thread transport for detached queries; not the instrument runtime.

The controller thread owns scheduler/bank mutations and dispatch/receipt. One
executor thread owns the matcher's scratch. Its clock must share the controller's
monotonic time origin; validation always uses the original dispatch evidence cut.
"""

from concurrent.futures import ThreadPoolExecutor
import math
import time


class NativeQueryWorker:
    """One outstanding future per bus; the scheduler owns pending replacement."""

    def __init__(self, scheduler, bank, matcher, clock=time.perf_counter):
        if scheduler.active_slot is not None:
            raise ValueError('an idle scheduler is required for worker ownership')
        self.scheduler, self.bank, self.matcher, self.clock = scheduler, bank, matcher, clock
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='m0-query')
        self.future = None
        self.ticket = None
        self.dispatched_at = None
        self.closed = False
        self.counts = dict(dispatched=0, received=0, failed=0, stale=0)

    def dispatch(self):
        if self.closed:
            raise RuntimeError('worker is closed')
        if self.future is not None:
            return False
        cut = self.clock()
        job = self.scheduler.take(cut, packed=True)
        if job is None:
            return False
        try:
            episodes = self.bank.episodes(cut, packed=True)
            future = self.executor.submit(self._run, job['query'], episodes, cut)
        except Exception:
            self.scheduler.finish(job['ticket'], None, {}, self.clock())
            self.counts['failed'] += 1
            raise
        self.future, self.ticket, self.dispatched_at = future, job['ticket'], cut
        self.counts['dispatched'] += 1
        return True

    def _run(self, query, episodes, evidence_cut):
        started = self.clock()
        try:
            if not math.isfinite(started) or started < evidence_cut:
                raise ValueError('worker start cannot precede its dispatch cut')
            result = self.matcher.match_query(query, episodes, evidence_cut)
            completed = self.clock()
            if not math.isfinite(completed) or completed < started:
                raise ValueError('worker completion clock must be monotonic')
            result['completed_at'] = completed
            return result, None, started, completed
        except Exception as error:
            return None, error, started, self.clock()

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        result, error, started, completed = self.future.result()
        received = self.clock()
        if not math.isfinite(received) or not math.isfinite(completed) or received < completed:
            raise ValueError('receipt cannot precede actual worker completion')
        # Resolve bindings at receipt: retired/replaced bank identities stay absent.
        accepted = self.scheduler.finish(self.ticket, result, self.bank.bindings(), received)
        receipt = dict(ticket=self.ticket, accepted=accepted, result=result,
                       dispatched_at=self.dispatched_at, started_at=started,
                       completed_at=completed, received_at=received)
        self.future, self.ticket, self.dispatched_at = None, None, None
        self.counts['received'] += 1
        if error is not None:
            self.counts['failed'] += 1
            raise error
        if not accepted:
            self.counts['stale'] += 1
        return receipt

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True)
        return self.poll()
