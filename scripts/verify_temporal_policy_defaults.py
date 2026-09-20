"""Verify real policy-default acquisition against reports and saved PCM.

Replay correctness is asserted by the Rust acquisition; this checks its saved
inputs, receipts, routing, byte identity and physical energy independently.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, inputs):
    manifest = json.loads((root / 'manifest.json').read_text())
    registration = json.loads((inputs / 'registration.json').read_text())
    require(manifest['schema'] == 'i10-real-policy-default-corpus-v1'
            and manifest['registration'] == registration, 'registration identity')
    require(sha(inputs / 'config.toml') == registration['config_sha256'], 'config changed')
    issue, horizon = registration['issue_sample'], registration['horizon_samples']
    fs, hop = registration['sample_rate'], registration['hop_samples']
    require((issue, horizon, fs, hop) == (24576, 192000, 48000, 512), 'physical grid')
    require([c['id'] for c in manifest['cases']] == [c['id'] for c in registration['cases']], 'case order')
    results = []
    for case in registration['cases']:
        require(case['split'] == 'development', 'unexpected held-out acquisition')
        require(sha(inputs / case['script']) == case['script_sha256'], 'scenario changed')
        directory = root / case['id']
        metadata = json.loads((directory / 'default.json').read_text())
        trace = [json.loads(line) for line in (directory / 'policy-inputs.jsonl').read_text().splitlines()]
        report = [json.loads(line) for line in (directory / 'ordinary.jsonl').read_text().splitlines()]
        require(metadata['case'] == case and metadata['schema'] == 'i10-real-policy-default-v1', 'case metadata')
        require([r['now'] for r in trace] == list(range(0, issue + horizon, hop)), 'missing or reordered hop')
        require(metadata['issue_input'] == trace[issue // hop], 'issue input mismatch')
        expected_routing = dict(to_habitat=case['routing'] != 'presentation',
                                to_presentation=case['routing'] != 'habitat')
        receipts = {(r['issued_at_sample'], r['tone_id'], r['action']): r
                    for r in report if r['type'] == 'self_sound_outcome'}
        counts = [0, 0, 0]
        first = None
        closed = 0
        submitted = 0
        for r in trace:
            require(len(r['batches']) <= 1, 'multiple sources in private target')
            body = r['body']
            if body:
                require((body['source_id'], body['source_generation']) == (1, 0), 'body identity')
                require(body['routing'] == expected_routing, 'body routing')
            for batch in r['batches']:
                require((batch['source_id'], batch['source_generation']) == (1, 0), 'batch identity')
                require(batch['routing'] == expected_routing, 'batch routing')
                require(batch['body_policy']['at'] == r['now'], 'policy issue clock')
                closed += not batch['body_policy']['gate_allows_onset']
                opportunity = batch['body_opportunity']
                if opportunity:
                    require(opportunity['issued_at'] == r['now'] and opportunity['at'] >= r['now'], 'opportunity clock')
                specs = {t['tone_id']: t for t in batch['tones']}
                require(len(specs) == len(batch['tones']), 'duplicate tone specification')
                for cmd in batch['cmds']:
                    require(len(cmd) == 1, 'command encoding')
                    kind, parameters = next(iter(cmd.items()))
                    require(kind in ['On', 'Off', 'Update'], 'command kind')
                    if r['now'] >= issue:
                        counts[['On', 'Off', 'Update'].index(kind)] += 1
                    if kind == 'Update':
                        continue
                    tone_id = parameters['tone_id']
                    action = 'onset' if kind == 'On' else 'release'
                    receipt = receipts.get((r['now'], tone_id, action))
                    require(receipt is not None, 'missing ordinary command receipt')
                    at = specs[tone_id]['onset'] if kind == 'On' else parameters['off_tick']
                    require(receipt['scheduled_action_sample'] == at
                            and receipt['source_id'] == 1 and receipt['source_generation'] == 0, 'receipt identity or time')
                    submitted += 1
                    if kind == 'On' and r['now'] >= issue and first is None:
                        first = dict(issued_at=r['now'], tone=specs[tone_id],
                                     source_generation=batch['source_generation'], routing=batch['routing'],
                                     policy=batch['body_policy'], opportunity=opportunity,
                                     intrinsic_period_sec=batch['intrinsic_period_sec'])
        require(counts == metadata['future_commands'], 'future command census')
        require(first == metadata['first_future_onset'], 'first real future onset')
        buses = []
        for bus in range(2):
            prefix = np.fromfile(directory / f'prefix-bus{bus}.f32le', dtype='<f4')
            require(len(prefix) == issue and np.isfinite(prefix).all(), 'prefix PCM')
            actual_path = directory / f'actual-default-bus{bus}.f32le'
            actual = np.fromfile(actual_path, dtype='<f4')
            control = np.fromfile(directory / f'no-commands-bus{bus}.f32le', dtype='<f4')
            require(len(actual) == len(control) == horizon
                    and np.isfinite(actual).all() and np.isfinite(control).all(), 'future PCM')
            require(sha(actual_path) == sha(directory / f'serialized-replay-bus{bus}.f32le')
                    == sha(directory / f'frozen-default-bus{bus}.f32le'), 'default replay PCM mismatch')
            routed = expected_routing['to_habitat' if bus == 0 else 'to_presentation']
            require(routed or not (np.any(prefix) or np.any(actual) or np.any(control)), 'unrouted bus is not silent')
            energies = np.array([(wave.astype(float).reshape(-1, hop) ** 2).mean(axis=1)
                                 for wave in [actual, control]])
            saved = np.array(metadata['buses'][bus]['energy_per_hop'])
            require(saved.shape == energies.shape and np.allclose(saved, energies, rtol=2e-14, atol=1e-20), 'physical energy mismatch')
            different = int(np.count_nonzero(actual.view('<u4') != control.view('<u4')))
            require(different == metadata['buses'][bus]['different_samples_from_no_commands'], 'control difference census')
            buses.append(dict(bus=bus, different_samples=different, default_sha256=sha(actual_path),
                              default_energy_integral=float(energies[0].sum() * hop / fs),
                              no_commands_energy_integral=float(energies[1].sum() * hop / fs)))
        results.append(dict(case=case['id'], hops=len(trace), future_commands=counts,
                            ordinary_receipts=submitted, gate_closed_hops=closed, buses=buses))
    return dict(schema='i10-policy-default-verification-v1', cases=results,
                total_hops=sum(r['hops'] for r in results),
                future_commands=np.sum([r['future_commands'] for r in results], axis=0).tolist(),
                hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                        if p.is_file() and p.suffix in ['.jsonl', '.f32le']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.inputs)
    (args.root / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'hashes'}, indent=2))
