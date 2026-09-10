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
from evaluate_chosen_outcomes import build_models, main, observe_chosen
from evaluate_perceptual_action import FIELDS


PLAN = dict(fit_sec=.04, validation_sec=.01, trajectory_counts=[0],
            residual_orders=[0], variance_floor=1e-12)


class ChosenOutcomeTests(unittest.TestCase):
    def test_owned_cancellation_chunking_and_observation_order(self):
        rng = np.random.default_rng(185021)
        past = rng.normal(0., .01, 3600)
        models, _ = build_models(past, np.zeros(3600), 24000, 3600, 480, PLAN)
        prior = models['learned'].forecast(240)
        np.testing.assert_array_equal(prior.mean, models['fixed'].forecast(240).mean)
        np.testing.assert_array_equal(prior.diagonal_variance(), models['fixed'].forecast(240).diagonal_variance())
        external = rng.normal(0., .1, 240)
        own = .15*np.sin(np.arange(240)*.15)
        block, other_action = copy.deepcopy(models), copy.deepcopy(models)
        ledger = observe_chosen(models, 3600, external+own, own, 24)
        direct = observe_chosen(block, 3600, external, np.zeros(240), 240)
        observe_chosen(other_action, 3600, external-own, -own, 24)
        self.assertEqual(len(ledger), 10)
        for name in models:
            a, b, c = [collection[name].forecast(240) for collection in (models, block, other_action)]
            np.testing.assert_allclose(a.mean, b.mean, atol=1e-14)
            np.testing.assert_allclose(a.diagonal_variance(), b.diagonal_variance(), rtol=1e-12)
            np.testing.assert_allclose(a.diagonal_variance(), c.diagonal_variance(), rtol=1e-12)
            self.assertAlmostEqual(sum(row[name]['pre_update_log_density'] for row in ledger),
                                   direct[0][name]['pre_update_log_density'], places=8)
        self.assertGreater(models['learned'].forecast(1).innovation_variance, prior.innovation_variance)
        self.assertEqual(models['fixed'].forecast(1).innovation_variance, prior.innovation_variance)
        saved = {name: model.forecast(20).digest() for name, model in models.items()}
        with self.assertRaises(ValueError):observe_chosen(models, 3841, external, own, 24)
        self.assertEqual(saved, {name: model.forecast(20).digest() for name, model in models.items()})

    def test_pipeline_keeps_other_action_and_unobserved_suffix_out_of_learning(self):
        # This stand-in checks I/O ordering, not the native R/H mathematics.
        def native_stub(command, *, env, stdout, stderr, check):
            plan = json.loads(Path(env['CONCHORDAL_PERCEPTUAL_ACTION_PLAN']).read_text())
            for case in plan['cases']:
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
            case = dict(name='case', fs=24000, issue_sample=3600, seed=41,
                        body='sine', route='habitat', external_condition='held')
            (source.parent/'inputs.json').write_text(json.dumps([case]))
            rng = np.random.default_rng(185029)
            known = np.linspace(0., .1, 480)
            truth_wait = rng.normal(0., .01, 480)
            truth_sound = known+truth_wait
            for name, values in dict(past_mix=rng.normal(0., .01, 3600), past_own=np.zeros(3600),
                                     own_wait=np.zeros(480), own_sound=known).items():
                np.asarray(values, dtype='<f4').tofile(source/(name+'.f32le'))
            plan = root/'plan.json'
            plan.write_text(json.dumps(dict(draws=4, observe_sec=.01, sampling_seed=43, forecast=PLAN)))
            results = []
            for index in range(2):
                np.asarray(truth_wait+.5*index, dtype='<f4').tofile(source/'truth_wait.f32le')
                changed = truth_sound.copy()
                changed[240:] += 2.*index
                np.asarray(changed, dtype='<f4').tofile(source/'truth_sound.f32le')
                output = root/f'output-{index}'
                argv = ['evaluate', '--input', str(source.parent), '--output', str(output), '--plan', str(plan)]
                with patch.object(sys, 'argv', argv), patch('evaluate_chosen_outcomes.subprocess.run', native_stub), redirect_stdout(io.StringIO()):
                    main()
                results.append(output)
            for name in ['initial.json', 'updates.json', 'updated.json']:
                self.assertEqual((results[0]/'case.sound'/name).read_text(), (results[1]/'case.sound'/name).read_text())
            for path in (results[0]/'case.sound').glob('*.draw.*.f32le'):
                self.assertEqual(path.read_bytes(), (results[1]/'case.sound'/path.name).read_bytes())
            self.assertNotEqual((results[0]/'case.wait'/'updated.json').read_text(),
                                (results[1]/'case.wait'/'updated.json').read_text())
            summaries = [json.loads((path/'summary.json').read_text()) for path in results]
            self.assertNotEqual(summaries[0][-1]['metrics'], summaries[1][-1]['metrics'])


if __name__ == '__main__':
    unittest.main()
