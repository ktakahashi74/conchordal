from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_hypotheses import propose_hypotheses
from acoustic_mixture import AcousticMixtureForecast
from evaluate_perceptual_action import FIELDS
from evaluate_structural_predictions import build_bank, main


HYPOTHESES = dict(base_fit_sec=.04, base_validation_sec=.01, adaptation_sec=.008,
    selection_sec=.002, horizon_sec=.01, counts=[0, 1], orders=[0, 4], variance_floor=1e-12,
    amplitude_degrees=[0, 1], adaptation_offsets_sec=[0., .004], added_counts=[0, 1])
FAMILY_MASS = dict(continued_state=1., changed_amplitude=1., added_components=1.)


class StructuralPredictionTests(unittest.TestCase):
    def test_passive_extension_preserves_the_conditioned_old_family_subset(self):
        fs, issued = 24000, 1200
        rng = np.random.default_rng(185113)
        audio = .05*np.cos(2*np.pi*777*np.arange(issued+192)/fs)+rng.normal(0., 1e-4, issued+192)
        old_pending, old_unavailable, _ = propose_hypotheses(audio[:issued], audio[issued:], fs, issued,
                                                           HYPOTHESES, learn_variance=True)
        pending, unavailable, _ = propose_hypotheses(audio[:issued], audio[issued:], fs, issued,
            dict(HYPOTHESES, refitted_passive_counts=[0, 1, 2]), learn_variance=True)
        old, _ = build_bank(old_pending, old_unavailable, FAMILY_MASS, 288)
        extended, record = build_bank(pending, unavailable, dict(FAMILY_MASS, refitted_passive=1.), 288)
        self.assertGreater(record['family_counts'].get('refitted_passive', 0), 0)
        for row in record['candidates']:
            if row['family'] == 'refitted_passive':
                self.assertTrue(all(gain <= 0. for gain in row['log_gain_per_sec']))
                self.assertIn(row['adaptation_start'], [issued, issued+96])
        comparison = rng.normal(0., .01, 48)
        old.observe(issued+192, comparison)
        extended.observe(issued+192, comparison)
        indices = [i for i, row in enumerate(record['candidates']) if row['family'] in FAMILY_MASS]
        weights = extended.log_weights[indices].copy()
        weights -= np.logaddexp.reduce(weights)
        full = extended.forecast(240)
        retained = AcousticMixtureForecast(tuple(full.components[i] for i in indices), weights)
        target = rng.normal(0., .01, 240)
        np.testing.assert_allclose(retained.mean, old.forecast(240).mean, atol=1e-12)
        np.testing.assert_allclose(retained.diagonal_variance(), old.forecast(240).diagonal_variance(), rtol=1e-10)
        self.assertAlmostEqual(retained.log_density(target), old.forecast(240).log_density(target), places=7)

    def test_family_prior_and_forecast_hold_before_common_comparison(self):
        fs, issued = 24000, 1200
        rng = np.random.default_rng(185063)
        audio = .05*np.cos(2*np.pi*777*np.arange(issued+192)/fs)+rng.normal(0., 1e-4, issued+192)
        pending, unavailable, _ = propose_hypotheses(audio[:issued], audio[issued:], fs, issued,
                                                    HYPOTHESES, learn_variance=True)
        bank, record = build_bank(pending, unavailable, FAMILY_MASS, 288)
        self.assertEqual(set(record['family_counts']), set(FAMILY_MASS))
        for family in FAMILY_MASS:
            mass = sum(np.exp(weight) for row, weight in zip(record['candidates'], bank.log_weights)
                       if row['family'] == family)
            self.assertAlmostEqual(mass, 1/3)
        duplicate = copy.deepcopy(pending)
        duplicate += copy.deepcopy([entry for entry in pending if entry[0] == 'added_components'])
        repeated, _ = build_bank(duplicate, unavailable, FAMILY_MASS, 288)
        prior = bank.forecast(288)
        np.testing.assert_allclose(prior.mean, repeated.forecast(288).mean, atol=1e-15)
        np.testing.assert_allclose(prior.diagonal_variance(), repeated.forecast(288).diagonal_variance(), rtol=1e-14)
        chunked = copy.deepcopy(bank)
        comparison = rng.normal(0., .01, 48)
        initial = bank.forecast(48)
        frozen = initial.digest()
        density = bank.observe(issued+192, comparison)[0]
        split = sum(chunked.observe(issued+192+offset, comparison[offset:offset+16])[0]
                    for offset in range(0, 48, 16))
        self.assertAlmostEqual(density, initial.log_density(comparison), places=9)
        self.assertAlmostEqual(density, split, places=8)
        np.testing.assert_allclose(bank.log_weights, chunked.log_weights, atol=1e-8)
        np.testing.assert_allclose(bank.forecast(240).mean, chunked.forecast(240).mean, atol=1e-12)
        np.testing.assert_allclose(bank.forecast(240).diagonal_variance(),
                                   chunked.forecast(240).diagonal_variance(), rtol=1e-10)
        self.assertEqual(initial.digest(), frozen)
        with self.assertRaises(ValueError):
            build_bank(pending, unavailable, dict(FAMILY_MASS, added_components=0.), 288)

    def test_comparison_future_and_unchosen_audio_have_separate_information_boundaries(self):
        # This observer stand-in tests pipeline timing, not native R/H kernels.
        def native_stub(command, *, env, stdout, stderr, check):
            native = json.loads(Path(env['CONCHORDAL_PERCEPTUAL_ACTION_PLAN']).read_text())
            for case in native['cases']:
                directory = Path(case['output']).parent
                self.assertTrue((directory/'issued.json').exists())
                contract = dict(type='contract', id=case['id'], issued_sample=case['issued_sample'], hop_samples=128)
                with Path(case['output']).open('x') as out:
                    out.write(json.dumps(contract)+'\n')
                    for branch in case['branches']:
                        count = Path(branch['audio']).stat().st_size//4
                        observation = dict(end_sample=(case['issued_sample']+count)//128*128,
                                           **{field: [0., 0.] for field in FIELDS})
                        out.write(json.dumps(dict(type='branch', id=branch['id'], observation=observation))+'\n')

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'native'/'case'
            source.mkdir(parents=True)
            fs, issued = 24000, 1200
            case = dict(name='case', fs=fs, issue_sample=issued, seed=53,
                        body='sine', route='habitat', external_condition='held')
            (source.parent/'inputs.json').write_text(json.dumps([case]))
            rng = np.random.default_rng(185071)
            external = .05*np.cos(2*np.pi*777*np.arange(issued+480)/fs)+rng.normal(0., 1e-4, issued+480)
            own = .03*np.cos(2*np.pi*411*np.arange(issued+480)/fs)
            actual = external[issued:]+own[issued:]
            for name, values in dict(past_mix=external[:issued]+own[:issued], past_own=own[:issued],
                                     own_sound=own[issued:]).items():
                np.asarray(values, dtype='<f4').tofile(source/(name+'.f32le'))
            plan = root/'plan.json'
            plan.write_text(json.dumps(dict(draws=4, sampling_seed=59,
                hypotheses=dict(HYPOTHESES, refitted_passive_counts=[0, 1, 2]),
                family_prior_mass=dict(FAMILY_MASS, refitted_passive=1.),
                comparison_family_subset=list(FAMILY_MASS),
                driven_body=dict(drive_per_sec=[0., 1000.], initial_precision=.0001,
                                 variance_prior=[2., .000001]),
                perceptual_sampling=dict(seed=185219, target_draws=8, min_per_component=2,
                                         max_omitted_mass=1e-5, alpha=.05),
                parameter_uncertainty=dict(frequency_range=[20., 11999.], max_decay=32001.,
                    amplitude_precision=.0001, variance_prior=[2., .000001], seed=61,
                    warmup=32, draws=8, thin=1))))
            outputs = []
            original_read = np.fromfile
            for attempt in range(3):
                changed = actual.copy()
                if attempt == 1:
                    changed[240:] += .3
                elif attempt == 2:
                    changed[192:240] += .1
                np.asarray(changed, dtype='<f4').tofile(source/'truth_sound.f32le')
                np.full(480, attempt, dtype='<f4').tofile(source/'truth_wait.f32le')
                output = root/f'output-{attempt}'
                def ordered_read(path, *args, **kwargs):
                    self.assertNotEqual(Path(path).name, 'truth_wait.f32le')
                    if Path(path).name == 'truth_sound.f32le':
                        directory = output/'case.sound'
                        offset = kwargs.get('offset', 0)
                        if offset == 0:
                            self.assertEqual(kwargs['count'], 192)
                        elif offset == 192*4:
                            self.assertTrue((directory/'comparison.npz').exists())
                            self.assertTrue((directory/'proposed.json').exists())
                            self.assertTrue((directory/'body-proposed.json').exists())
                        elif offset == 240*4:
                            self.assertTrue((directory/'prediction-seal.json').exists())
                        else:
                            self.fail('unexpected observation window')
                    return original_read(path, *args, **kwargs)
                argv = ['evaluate', '--input', str(source.parent), '--output', str(output), '--plan', str(plan),
                        '--config', str(root/'config.toml'), '--action', 'sound']
                with patch.object(sys, 'argv', argv), patch('evaluate_structural_predictions.subprocess.run', native_stub), \
                        patch('evaluate_structural_predictions.np.fromfile', ordered_read), redirect_stdout(io.StringIO()):
                    main()
                outputs.append(output)
            for attempt in (1, 2):
                for name in ('proposed.json', 'comparison.npz', 'body-proposed.json'):
                    self.assertEqual((outputs[0]/'case.sound'/name).read_bytes(),
                                     (outputs[attempt]/'case.sound'/name).read_bytes())
            for name in ('updated.json', 'issued.json', 'forecast-moments.npz',
                         'parameters.json', 'parameter-chains.npz', 'body-updated.json'):
                self.assertEqual((outputs[0]/'case.sound'/name).read_bytes(), (outputs[1]/'case.sound'/name).read_bytes())
                self.assertNotEqual((outputs[0]/'case.sound'/name).read_bytes(), (outputs[2]/'case.sound'/name).read_bytes())
            for path in (outputs[0]/'case.sound').glob('*.draw.*.f32le'):
                self.assertEqual(path.read_bytes(), (outputs[1]/'case.sound'/path.name).read_bytes())
            for path in (outputs[0]/'case.sound').glob('*.stratum.*.f32le'):
                self.assertEqual(path.read_bytes(), (outputs[1]/'case.sound'/path.name).read_bytes())
            self.assertEqual((outputs[0]/'case.sound/integration-moments.npz').read_bytes(),
                             (outputs[1]/'case.sound/integration-moments.npz').read_bytes())
            results = [json.loads((path/'summary.json').read_text())[0]['metrics'] for path in outputs]
            self.assertNotEqual(results[0]['mixture']['joint_log_density'], results[1]['mixture']['joint_log_density'])


if __name__ == '__main__':
    unittest.main()
