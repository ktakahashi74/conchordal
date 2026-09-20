"""Check actual onset recipes and pre-resolution due receipts without treating them as live permission."""

import argparse
import copy
import json
from pathlib import Path

from verify_temporal_local_continuation import verify as verify_local
from verify_temporal_policy_defaults import require, sha


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root, inputs, previous):
    local = verify_local(root, inputs, previous)
    registration = json.loads((inputs / 'registration.json').read_text())
    require(registration['onset_opportunity']['schema'] == 'i10-onset-receipts-v1', 'receipt contract')
    results = []
    for case in registration['cases']:
        directory, old = root / case['id'], previous / case['id']
        trace = read_jsonl(directory / 'policy-inputs.jsonl')
        old_trace = read_jsonl(old / 'policy-inputs.jsonl')
        report = read_jsonl(directory / 'ordinary.jsonl')
        old_report = read_jsonl(old / 'ordinary.jsonl')
        # Removing only the new field must recover every original producer input.
        plain = copy.deepcopy(trace)
        for row in plain:
            for batch in row['batches']:
                for tone in batch['tones']:
                    tone.pop('opportunity')
        require(plain == old_trace, 'existing policy inputs or recipe changed')
        require(len(report) == len(old_report), 'report census changed')
        timing_fields = {
            'hop_timing': {'elapsed_us', 'analysis_wait_us', 'listener_wait_us'},
            'body_observation': {'processing_us', 'max_processing_us'},
            'temporal_observation': {'observation'},
        }
        for current, prior in zip(report, old_report):
            require(current['type'] == prior['type'], 'report order changed')
            for key, value in prior.items():
                if key in timing_fields.get(prior['type'], set()):
                    continue
                if key == 'version' and prior['type'] == 'body_default':
                    require(current[key] == 3, 'body-default version')
                else:
                    require(current[key] == value, f'old report field changed: {prior["type"]}/{key}')
        # Reports sample an asynchronous observer's latest frame. Align its source identity.
        observations = []
        for rows in [report, old_report]:
            observations.append({(r['observation']['bus'], r['observation']['source_epoch'],
                                  r['observation']['frame_id']): r['observation']
                                 for r in rows if r['type'] == 'temporal_observation'
                                 and r['observation']['frame_id'] is not None})
        common = observations[0].keys() & observations[1].keys()
        require(bool(common), 'no common observed frames')
        for identity in common:
            for key, value in observations[1][identity].items():
                if key != 'delivery_delay_us':
                    require(observations[0][identity][key] == value, f'observed frame changed: {identity}/{key}')
        releases = {(r['tone_id'], r['scheduled_action_sample']) for r in report
                    if r['type'] == 'self_sound_outcome' and r['action'] == 'release'
                    and r['command_status'] == 'accepted'}
        receipts, due_matches, shifted, sampled = 0, 0, 0, 0
        by_hop = {}
        last_intrinsic_due = None
        for row in trace:
            current_recipes = []
            for batch in row['batches']:
                clock = batch['body_opportunity']
                if clock and clock['basis'] == 'participation_due':
                    last_intrinsic_due = clock['at']
                onsets = {c['On']['tone_id'] for c in batch['cmds'] if 'On' in c}
                events = {e['onset_tick']: e for e in batch['onsets']}
                for tone in batch['tones']:
                    receipt = tone['opportunity']
                    require((receipt is None) == (case['id'] == 'dev-sine-hold'), 'missing/false clock receipt')
                    if receipt is None:
                        continue
                    receipts += 1
                    require(tone['tone_id'] in onsets, 'recipe without On command')
                    require(batch['body_policy']['is_alive'] and batch['body_policy']['gate_allows_onset'],
                            'grant under closed policy')
                    require(receipt['issued_at'] == row['now'] and receipt['at'] == tone['onset']
                            and row['now'] <= receipt['at'] < row['now'] + 512, 'receipt clock')
                    require(events[receipt['at']]['gate'] == receipt['gate'], 'receipt gate')
                    require((tone['tone_id'], receipt['planned_release_at']) in releases, 'planned release receipt')
                    if case['id'] == 'dev-harmonic-pulse':
                        require(receipt['intrinsic_period_ticks'] is None
                                and receipt['intrinsic_due_at'] == receipt['at'], 'theta is not body period')
                    else:
                        require(receipt['intrinsic_period_ticks'] == 24000, 'intrinsic body period')
                        require(receipt['intrinsic_due_at'] is not None, 'missing selected intrinsic due')
                        if last_intrinsic_due is not None:
                            require(receipt['intrinsic_due_at'] == last_intrinsic_due, 'intrinsic due changed before selection')
                            due_matches += 1
                        last_intrinsic_due = None
                        shifted += receipt['intrinsic_due_at'] != receipt['at']
                    current_recipes.append(tone)
            by_hop[row['now']] = current_recipes
        for record in report:
            if record['type'] != 'body_default' or record['issued_at'] not in by_hop:
                continue
            recipes = by_hop[record['issued_at']]
            require(record['onset_recipe_count'] == len(recipes), 'sampled recipe count')
            expected = recipes[0] if recipes else None
            require(record['onset_recipe_tone_id'] == (expected['tone_id'] if expected else None)
                    and record['onset_opportunity'] == (expected['opportunity'] if expected else None),
                    'sampled recipe join')
            sampled += bool(recipes)
        unchanged = 0
        for path in old.glob('local-continuation/*.f32le'):
            require(sha(path) == sha(directory / 'local-continuation' / path.name), 'local branch PCM changed')
            unchanged += 1
        results.append(dict(case=case['id'], granted_recipes=receipts,
                            original_due_comparisons=due_matches, shifted_selected_plans=shifted,
                            sampled_recipe_records=sampled, unchanged_local_pcm_files=unchanged,
                            report_records=len(report), unchanged_observed_frames=len(common),
                            unmatched_observed_frames=[len(o) - len(common) for o in observations]))
    require(sum(r['granted_recipes'] for r in results) > 0
            and sum(r['sampled_recipe_records'] for r in results) > 0, 'no ordinary receipt consumer')
    return dict(schema='i10-onset-receipts-verification-v1', cases=results,
                unchanged_policy_pcm_files=local['unchanged_policy_pcm_files'],
                hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.glob('*/*.jsonl'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('previous', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.inputs, args.previous)
    (args.root / 'receipt-verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, indent=2))
