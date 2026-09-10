from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_perceptual_action import FIELDS, compare, prepare, prepare_actual


class PerceptualActionTests(unittest.TestCase):
    def test_unseen_actual_audio_cannot_change_issued_perceptual_trajectories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'native'
            directory = source / 'case'
            directory.mkdir(parents=True)
            case = dict(name='case', fs=24000, issue_sample=3600, seed=41)
            (source / 'inputs.json').write_text(json.dumps([case]))
            past = .07 * np.sin(2*np.pi*413*np.arange(3600)/24000)
            for name, values in dict(past_mix=past, past_own=np.zeros(3600),
                    own_wait=np.zeros(240), own_sound=np.linspace(0., .03, 240)).items():
                np.asarray(values, dtype='<f4').tofile(directory / (name+'.f32le'))
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(draws=4, sampling_seed=71, forecast=dict(
                fit_sec=.04, validation_sec=.01, trajectory_counts=[0, 1],
                residual_orders=[0, 4], variance_floor=1e-12))))
            outputs = []
            for index in range(2):
                for action in ('wait', 'sound'):
                    np.full(240, 17.*index, dtype='<f4').tofile(directory / ('truth_'+action+'.f32le'))
                args = SimpleNamespace(input=source, output=root / f'issued-{index}', plan=plan,
                                       config=Path('config.toml'))
                with redirect_stdout(io.StringIO()):
                    prepare(args)
                outputs.append(args.output / 'case')
                with self.assertRaises(FileNotFoundError):
                    prepare_actual(args)
                self.assertFalse((args.output / 'actual-plan.json').exists())
            for file in outputs[0].glob('*.f32le'):
                self.assertEqual(file.read_bytes(), (outputs[1]/file.name).read_bytes())
            first, second = [json.loads((path/'issued.json').read_text()) for path in outputs]
            self.assertEqual(first, second)

    def test_native_observation_alignment_and_nonlinear_mean_are_scored_separately(self):
        contract = dict(issued_sample=1000, hop_samples=128,
                        **{'observed_'+field: [0., 0.] for field in FIELDS})
        def observation(value, end=1280):
            return dict(end_sample=end, startup_zero_samples=0, **{f: [value, value] for f in FIELDS})
        predicted = {f'{a}.mean': dict(observation=observation(0.)) for a in ('wait', 'sound')}
        for action in ('wait', 'sound'):
            for index in range(4):
                predicted[f'{action}.draw.{index}'] = dict(observation=observation(.4 if action=='sound' else 0.))
        actual = {a: dict(observation=observation(.4 if a=='sound' else 0.)) for a in ('wait', 'sound')}
        result = compare(contract, predicted, actual, 4)
        self.assertEqual(result['status'], 'observed')
        for field in FIELDS:
            score = result['branches']['sound'][field]
            self.assertEqual(score['expected_field_mse'], 0.)
            self.assertAlmostEqual(score['mean_waveform_field_mse'], .16)
            self.assertEqual(result['action_difference'][field]['mse'], 0.)
        predicted['wait.draw.0']['observation']['end_sample'] = 1279
        with self.assertRaises(ValueError):
            compare(contract, predicted, actual, 4)
        actual['wait']['observation']['end_sample'] = 1024
        self.assertEqual(compare(contract, predicted, actual, 4), dict(status='no_complete_future_hop'))


if __name__ == '__main__':
    unittest.main()
