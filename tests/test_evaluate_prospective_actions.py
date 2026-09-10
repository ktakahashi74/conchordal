from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_trajectory import TrajectoryFit
from evaluate_perceptual_action import FIELDS
from evaluate_prospective_actions import main
from evaluate_structural_predictions import build_driven_bank


class ProspectiveActionTests(unittest.TestCase):
    def test_shared_body_builder_deduplicates_support_and_rejects_future_parameters(self):
        fit = TrajectoryFit(24000, 0., np.array([411.]), np.zeros(1), np.array([-12.]),
                            np.zeros(1, complex), 0, 32)
        candidate = ('refitted_passive', SimpleNamespace(trajectory=fit), dict(adaptation_start=0))
        audio = np.random.default_rng(185241).normal(0., .1, 32)
        settings = dict(drive_per_sec=[0., 1000.], initial_precision=.01, variance_prior=[2., .01])
        one, rows = build_driven_bank([candidate], audio, 24000, 0, settings)
        repeated, repeated_rows = build_driven_bank([candidate, candidate], audio, 24000, 0, settings)
        self.assertEqual(rows, repeated_rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(one.forecast(20).digest(), repeated.forecast(20).digest())
        future = ('refitted_passive', SimpleNamespace(trajectory=replace(fit, fit_end_sample=33)), candidate[2])
        with self.assertRaises(ValueError):
            build_driven_bank([future], audio, 24000, 0, settings)

    def test_prospective_branches_and_chosen_learning_have_distinct_information_boundaries(self):
        # This bounded observation stand-in checks orchestration, not native R/H kernels.
        original_read = np.fromfile
        def native_stub(command, *, env, stdout, stderr, check):
            plan = json.loads(Path(env['CONCHORDAL_PERCEPTUAL_ACTION_PLAN']).read_text())
            for case in plan['cases']:
                past = original_read(case['past'], dtype='<f4').astype(float)
                self.assertEqual(len(past), case['issued_sample'])
                with Path(case['output']).open('x') as out:
                    out.write(json.dumps(dict(type='contract', id=case['id'], issued_sample=len(past), hop_samples=128))+'\n')
                    for branch in case['branches']:
                        future = original_read(branch['audio'], dtype='<f4').astype(float)
                        end = (len(past)+len(future))//128*128
                        audio = np.r_[past, future][end-128:end]
                        values = [.5+.2*np.tanh(20*audio.mean()), .5+.2*np.tanh(100*np.mean(audio**2))]
                        observation = dict(end_sample=end, startup_zero_samples=0, **{f: values for f in FIELDS})
                        out.write(json.dumps(dict(type='branch', id=branch['id'], observation=observation))+'\n')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root/'native/case';source.mkdir(parents=True)
            fs, issued, horizon, update = 24000, 1200, 480, 120
            rng = np.random.default_rng(185243)
            time = np.arange(issued+horizon)/fs
            external = .05*np.cos(2*np.pi*777*time)+rng.normal(0., 1e-4, len(time))
            own = .03*np.cos(2*np.pi*411*time)
            wait, sound = own[issued:].copy(), own[issued:]+.02*np.cos(2*np.pi*999*time[issued:])
            for key, values in dict(past_mix=external[:issued]+own[:issued], past_own=own[:issued],
                                     own_wait=wait, own_sound=sound).items():
                np.asarray(values, dtype='<f4').tofile(source/(key+'.f32le'))
            (source.parent/'inputs.json').write_text(json.dumps([dict(name='case', fs=fs, issue_sample=issued,
                seed=7, body='sine', route='habitat', external_condition='held')]))
            plan = dict(chosen_actions=['sound'], update_sec=.005,
                hypotheses=dict(base_fit_sec=.01, base_validation_sec=.005, adaptation_sec=.004,
                    selection_sec=.002, horizon_sec=.02, counts=[0, 1], orders=[0, 4], variance_floor=1e-12,
                    adaptation_offsets_sec=[0.], amplitude_degrees=[0], added_counts=[0, 1], refitted_passive_counts=[0, 1]),
                family_prior_mass=dict(continued_state=1., changed_amplitude=1., added_components=1., refitted_passive=1.),
                driven_body=dict(drive_per_sec=[0., 1000.], initial_precision=.0001, variance_prior=[2., .000001]),
                perceptual_sampling=dict(seed=185247, target_draws=8, min_per_component=2, max_omitted_mass=1e-5, alpha=.05))
            (root/'plan.json').write_text(json.dumps(plan))
            outputs = []
            for attempt in range(4):
                actual_wait = external[issued:]+wait
                actual_sound = external[issued:]+sound
                if attempt == 1:
                    actual_sound[update:] += .03
                if attempt == 2:
                    actual_wait += .07
                if attempt == 3:
                    actual_sound[:update] += .04
                for key, values in dict(wait=actual_wait, sound=actual_sound).items():
                    np.asarray(values, dtype='<f4').tofile(source/('truth_'+key+'.f32le'))
                output = root/f'run-{attempt}'
                def ordered_read(path, *args, **kwargs):
                    if Path(path).name.startswith('truth_'):
                        directory = output/'case'
                        self.assertTrue((directory/'prediction-seal.json').exists())
                        if kwargs.get('count') == update:
                            self.assertEqual(Path(path).name, 'truth_sound.f32le')
                            self.assertFalse((directory/'updates-issued.json').exists())
                        else:
                            self.assertEqual(kwargs.get('count'), horizon)
                            self.assertTrue((directory/'updates-issued.json').exists())
                    return original_read(path, *args, **kwargs)
                argv = ['evaluate', '--input', str(source.parent), '--output', str(output),
                        '--plan', str(root/'plan.json'), '--config', str(root/'config.toml')]
                with patch.object(sys, 'argv', argv), patch('evaluate_prospective_actions.subprocess.run', native_stub), \
                        patch('evaluate_prospective_actions.np.fromfile', ordered_read), redirect_stdout(io.StringIO()):
                    main()
                outputs.append(output/'case')
            for other in outputs[1:]:
                for name in ('issued.json', 'forecast-moments.npz', 'prospective-estimates.npz', 'numerical.json'):
                    self.assertEqual((outputs[0]/name).read_bytes(), (other/name).read_bytes())
            for other in outputs[1:3]:
                for name in ('updated-sound-issued.json', 'updated-sound-estimates.npz'):
                    self.assertEqual((outputs[0]/name).read_bytes(), (other/name).read_bytes())
            self.assertNotEqual((outputs[0]/'updated-sound-issued.json').read_bytes(),
                                (outputs[3]/'updated-sound-issued.json').read_bytes())
            # Pairing changes only the known own waveform, up to f32 summation roundoff.
            for path in outputs[0].glob('*.wait.draw.*.f32le'):
                a = original_read(path, dtype='<f4').astype(float)
                b = original_read(path.with_name(path.name.replace('.wait.', '.sound.')), dtype='<f4').astype(float)
                np.testing.assert_allclose(b-a, sound-wait, atol=3e-8, rtol=0.)


if __name__ == '__main__':
    unittest.main()
