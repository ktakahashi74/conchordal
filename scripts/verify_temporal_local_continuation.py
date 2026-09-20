"""Audit issue-known continuation/release/gap branches, separately from policy targets."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from verify_temporal_policy_defaults import require, sha, verify as verify_defaults


def verify(root, inputs, previous):
    default_result = verify_defaults(root, inputs)
    registration = json.loads((inputs / 'registration.json').read_text())
    contract = registration['local_continuation']
    require(contract['schema'] == 'i10-local-continuation-v1', 'local contract')
    issue, horizon, hop = (registration[k] for k in
                           ['issue_sample', 'horizon_samples', 'hop_samples'])
    results = []
    unchanged = 0
    for case in registration['cases']:
        base = root / case['id']
        directory = base / 'local-continuation'
        snapshot = json.loads((directory / 'issue.json').read_text())
        data = json.loads((directory / 'branches.json').read_text())
        trace = [json.loads(line) for line in (base / 'policy-inputs.jsonl').read_text().splitlines()]
        require(data['registration'] == contract and snapshot['issue'] == issue, 'issue/contract')
        prefix_ids = {tone['tone_id'] for r in trace if r['now'] < issue
                      for b in r['batches'] for tone in b['tones']}
        future_offs = {(c['Off']['tone_id'], c['Off']['off_tick'])
                       for r in trace if r['now'] >= issue
                       for b in r['batches'] for c in b['cmds'] if 'Off' in c}
        releases = snapshot['planned_releases']
        require(releases == sorted(releases, key=lambda t: (t[1], t[0])), 'release order')
        require(all(tone in prefix_ids and at >= issue and (tone, at) in future_offs
                    for tone, at in releases), 'release does not match actual scheduled receipt')
        envelopes = copy.deepcopy(snapshot['envelopes'])
        require(len({tone for tone, _ in envelopes}) == len(envelopes), 'duplicate envelope')
        for tone, envelope in envelopes:
            require(tone in prefix_ids, 'future tone leaked into snapshot')
            require(envelope['onset'] < issue, 'queued onset requires an explicit cancellation model')
            for released, at in releases:
                if released == tone and at < envelope['hold_end']:
                    envelope['hold_end'] = at
                    envelope['release_end'] = at + envelope['release_ticks']
        require(envelopes == data['planned_envelopes'], 'planned envelope mismatch')
        expected_keys = [('continue', 0)] + [(c, t) for c in ['release', 'gap']
                                            for t in contract['offsets_samples']]
        require([(r['class'], r['at'] - issue) for r in data['candidates']] == expected_keys,
                'candidate census')
        expected = {'planned-control': releases, 'unplanned-control': []}
        inputs_by_name = {name: None for name in expected}
        exclusions = 0
        for candidate in data['candidates']:
            cls, at, name = (candidate[k] for k in ['class', 'at', 'name'])
            active = [tone for tone, e in envelopes if e['onset'] <= at < e['release_end']]
            require(candidate['active_tones'] == active, 'candidate activity')
            legal = bool(active) if cls != 'gap' else snapshot['period'] is not None
            require((candidate['input'] is not None) == legal, 'candidate eligibility')
            if not legal:
                exclusions += 1
                continue
            action = dict(class_=cls, issued_at=issue, at=at, excitation_at=None,
                          release_at=at if cls in ['release', 'gap'] and active else None,
                          reconsider_at=None, consumes_due_opportunity=False,
                          withhold_until=at + snapshot['period'] if cls == 'gap' else None)
            action['class'] = action.pop('class_')
            require(candidate['input'] == action, 'class contract')
            extra = [[tone, at] for tone in active] if action['release_at'] is not None else []
            expected[name] = [list(c) for c in sorted(set(map(tuple, releases + extra)),
                                                    key=lambda t: (t[1], t[0]))]
            inputs_by_name[name] = action
        require([b['name'] for b in data['branches']] == list(expected), 'branch census')
        waves = {}
        for branch in data['branches']:
            name = branch['name']
            require(branch['commands'] == expected[name], 'branch command sequence')
            require(branch['input'] == inputs_by_name[name], 'branch input')
            ends = []
            for tone, envelope in snapshot['envelopes']:
                hold, end = envelope['hold_end'], envelope['release_end']
                for target, at in expected[name]:
                    if tone == target and at < hold:
                        hold, end = at, at + envelope['release_ticks']
                ends.append(end)
            last_end = max(ends, default=issue)
            waves[name] = []
            for bus in range(2):
                wave = np.fromfile(directory / f'{name}-bus{bus}.f32le', dtype='<f4')
                require(len(wave) == horizon and np.isfinite(wave).all(), 'local PCM length/finite')
                routed = case['routing'] != ('presentation' if bus == 0 else 'habitat')
                require(routed or not np.any(wave), 'local routing')
                require(not np.any(wave[max(0, last_end - issue):]), 'natural tail endpoint')
                energy = (wave.astype(float).reshape(-1, hop)**2).mean(axis=1)
                saved = branch['buses'][bus]
                require(saved['bus'] == bus and np.shape(saved['energy_per_hop']) == energy.shape
                        and np.allclose(saved['energy_per_hop'], energy, rtol=2e-14, atol=1e-20),
                        'local physical energy')
                waves[name].append(wave)
                if name == 'continue-0':
                    require(np.array_equal(wave.view('<u4'), waves['planned-control'][bus].view('<u4')),
                            'continue changed planned trajectory')
                if name.startswith('gap-'):
                    release_name = name.replace('gap-', 'release-')
                    if release_name in waves:
                        require(np.array_equal(wave.view('<u4'), waves[release_name][bus].view('<u4')),
                                'gap discarded release tail')
                if branch['input']:
                    before = (branch['input']['at'] - issue) // hop * hop
                    require(np.array_equal(wave[:before].view('<u4'),
                                           waves['planned-control'][bus][:before].view('<u4')),
                            'intervention changed earlier hops')
        for old in sorted((previous / case['id']).glob('*.f32le')):
            require(sha(old) == sha(base / old.name), 'ordinary policy audio changed')
            unchanged += 1
        differences = [int(np.count_nonzero(waves['planned-control'][b].view('<u4') !=
                                           waves['unplanned-control'][b].view('<u4'))) for b in range(2)]
        results.append(dict(case=case['id'], pending_releases=len(releases),
                            branches=len(expected), excluded=exclusions,
                            different_samples_from_omitting_releases=differences,
                            continuation_end=max((e['release_end'] for _, e in envelopes), default=issue)))
    require(sum(sum(r['different_samples_from_omitting_releases']) for r in results) > 0,
            'no positive missing-release contrast')
    return dict(schema='i10-local-continuation-verification-v1', cases=results,
                unchanged_policy_pcm_files=unchanged,
                default_hops=default_result['total_hops'],
                hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.glob('*/local-continuation/*'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('previous', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.inputs, args.previous)
    (args.root / 'local-verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, indent=2))
