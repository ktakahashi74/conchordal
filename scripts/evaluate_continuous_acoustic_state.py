"""Research PCM state updates separated from requests for joint future forecasts.

Input chunks declare contiguous [start, end) intervals, with None for missing
audio. Each actual sample updates the existing driven acoustic posterior once.
Query scoring never updates it. This is not stream identity or cognitive memory.
"""
import numpy as np


def stream_predictions(model, chunks, queries):
    """Yield observation, immutable issuance, outcome, censoring and EOF records.

Queries are ordered (issue_sample, horizon_samples) pairs; repeated requests
are allowed. Forecast objects in issued records retain their own covariance.
"""
    queries = tuple(queries)
    previous = model.next_sample
    for issue, horizon in queries:
        if (not isinstance(issue, int) or issue < previous
                or not isinstance(horizon, int) or horizon < 1):
            raise ValueError('expected ordered integer queries with positive horizons')
        previous = issue
    pending = {}
    requested = completed = censored = 0
    for start, end, values in chunks:
        if (not isinstance(start, int) or not isinstance(end, int)
                or start != model.next_sample or end <= start):
            raise ValueError('declare each contiguous observed or missing interval once')
        audio = None if values is None else np.asarray(values, dtype=float)
        if audio is not None and (audio.shape != (end-start,) or not np.isfinite(audio).all()):
            raise ValueError('expected a complete finite mono interval or explicit missing input')
        while True:
            now = model.next_sample
            for query_id, (forecast, target, observed) in tuple(pending.items()):
                if target == now:
                    yield dict(kind='completed', query_id=query_id, target_sample=now,
                               issued_sample=forecast.issued_sample,
                               issued_sha256=forecast.digest(),
                               joint_log_density=forecast.log_density(observed),
                               observed_samples=model.observed_samples)
                    del pending[query_id]
                    completed += 1
            while requested < len(queries) and queries[requested][0] == now:
                _, horizon = queries[requested]
                forecast = model.forecast(horizon)
                pending[requested] = (forecast, now+horizon, np.empty(horizon))
                yield dict(kind='issued', query_id=requested, issued_sample=now,
                           target_sample=now+horizon, observed_samples=model.observed_samples,
                           last_observed_sample=model.last_observed_sample,
                           state_sample=model.next_sample, forecast=forecast,
                           issued_sha256=forecast.digest())
                requested += 1
            if now == end:
                break
            stop = min(end, queries[requested][0] if requested < len(queries) else end,
                       min((target for _, target, _ in pending.values()), default=end))
            if audio is None:
                model.advance_to(stop)
                for query_id, (forecast, target, _) in pending.items():
                    yield dict(kind='censored', query_id=query_id, target_sample=target,
                               issued_sample=forecast.issued_sample, issued_sha256=forecast.digest(),
                               missing_start_sample=now, missing_end_sample=stop, reason='input_gap')
                censored += len(pending)
                pending.clear()
                yield dict(kind='missing', start_sample=now, end_sample=stop,
                           observed_samples=model.observed_samples,
                           last_observed_sample=model.last_observed_sample)
            else:
                block = audio[now-start:stop-start]
                density = model.observe(now, block)
                for forecast, _, observed in pending.values():
                    offset = now-forecast.issued_sample
                    observed[offset:offset+len(block)] = block
                yield dict(kind='observed', start_sample=now, end_sample=stop,
                           observed_samples=model.observed_samples,
                           conditional_log_density=density)
    assert requested == completed+censored+len(pending)
    yield dict(kind='eof', state_sample=model.next_sample, observed_samples=model.observed_samples,
               last_observed_sample=model.last_observed_sample, completed=completed,
               censored=censored, pending_targets=[target for _, target, _ in pending.values()],
               unissued_queries=len(queries)-requested)
