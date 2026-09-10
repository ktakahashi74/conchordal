import copy
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_body import DrivenBodyPosterior
from acoustic_mixture import AcousticModelBank
from acoustic_revision import revise_body_bank


class AcousticRevisionTests(unittest.TestCase):
    def setUp(self):
        self.fs = 24000
        self.settings = dict(drive_per_sec=[0.,1000.], initial_precision=.001, variance_prior=[2.,1e-6])
        self.rows = [dict(frequency_hz=[f],log_gain_per_sec=[-4.],drive_per_sec=d)
                     for f,d in [(240.,0.),(510.,1000.)]]
        history = .08*np.cos(2*np.pi*240*np.arange(256)/self.fs)
        models = []
        for row in self.rows:
            model = DrivenBodyPosterior(self.fs, **row, initial_precision=.001, variance_prior=[2.,1e-6])
            model.observe(0,history)
            models.append(model)
        self.original = AcousticModelBank(models,[.8,.2])
        self.audio = .12*np.cos(2*np.pi*720*np.arange(128)/self.fs+.31)

    def revise(self, original=None, rows=None, audio=None, **overrides):
        options = dict(counts=[0,1],offsets_sec=[0.,.002],prior_mass=dict(retained=3.,fresh=1.))
        options.update(overrides)
        return revise_body_bank(self.original if original is None else original,
            self.rows if rows is None else rows, self.audio if audio is None else audio,
            self.fs,self.settings,**options)

    def test_controls_separate_state_scale_and_weight_and_leave_original_unchanged(self):
        original = self.original.forecast(64)
        digest = original.digest()
        banks, record = self.revise()
        self.assertEqual(original.digest(),digest)
        self.assertEqual(self.original.forecast(64).digest(),digest)
        self.assertEqual(self.original.next_sample,256)
        self.assertEqual(record['comparison_start_sample'],384)
        self.assertTrue(all(b.next_sample==384 for b in banks.values()))
        for i in range(2):
            old = self.original.models[i]
            state = banks['state_reset'].models[i]
            scale = banks['state_scale_reset'].models[i]
            self.assertEqual(state.variance_shape,old.variance_shape+len(self.audio)/2)
            self.assertEqual(scale.variance_shape,self.settings['variance_prior'][0]+len(self.audio)/2)
            # Changing only the conjugate scale prior leaves the normalized state filter identical.
            np.testing.assert_array_equal(state.state_mean,scale.state_mean)
            np.testing.assert_array_equal(state.state_covariance,scale.state_covariance)
            self.assertNotEqual(state.variance_scale,scale.variance_scale)
            self.assertEqual(scale.forecast(64).digest(),banks['uniform_reset'].models[i].forecast(64).digest())
            self.assertFalse(np.array_equal(banks['retained'].models[i].state_covariance,state.state_covariance))
        self.assertFalse(np.array_equal(banks['uniform_reset'].log_weights,banks['state_scale_reset'].log_weights))
        comparison = .12*np.cos(2*np.pi*720*np.arange(128,160)/self.fs+.31)
        frozen = {n:b.forecast(len(comparison)) for n,b in banks.items()}
        hashes = {n:f.digest() for n,f in frozen.items()}
        for bank in banks.values():
            bank.observe(384,comparison)
        self.assertEqual({n:f.digest() for n,f in frozen.items()},hashes)
        self.assertEqual(self.original.forecast(64).digest(),digest)

    def test_new_support_explains_a_frequency_absent_from_retained_candidates(self):
        banks, record = self.revise()
        frequencies = [f for row in record['fresh_candidates'] for f in row['frequency_hz']]
        self.assertLess(min(abs(f-720.) for f in frequencies),.1)
        self.assertTrue(all(abs(f-720.)>100 for row in self.rows for f in row['frequency_hz']))
        target = .12*np.cos(2*np.pi*720*np.arange(128,160)/self.fs+.31)
        self.assertGreater(banks['fresh'].forecast(len(target)).log_density(target),
                           banks['uniform_reset'].forecast(len(target)).log_density(target))
        self.assertTrue(all(r['condition_start_sample']==256 and r['fit_end_sample']==384
                            for r in record['fresh_candidates']))

    def test_revision_preserves_tiny_log_mass_and_updates_groups_by_common_evidence(self):
        original = copy.deepcopy(self.original)
        original.log_weights = np.array([0.,-10000.])
        banks, _ = self.revise(original=original)
        retained,fresh,revision = [banks[n] for n in ('retained','fresh','revision')]
        self.assertLess(float(np.min(retained.log_weights)),-745.)
        np.testing.assert_allclose(revision.log_weights[:2],retained.log_weights+np.log(.75),atol=1e-12,rtol=0.)
        self.assertTrue(np.isfinite(revision.log_weights).all())
        target = .12*np.cos(2*np.pi*720*np.arange(128,160)/self.fs+.31)
        scores = np.array([retained.forecast(32).log_density(target),fresh.forecast(32).log_density(target)])
        joint = float(np.logaddexp.reduce(np.log([.75,.25])+scores))
        self.assertAlmostEqual(revision.forecast(32).log_density(target),joint,places=9)
        revision.observe(384,target)
        groups = np.array([np.logaddexp.reduce(revision.log_weights[:2]),np.logaddexp.reduce(revision.log_weights[2:])])
        np.testing.assert_allclose(groups,np.log([.75,.25])+scores-joint,atol=1e-9,rtol=0.)

    def test_zero_mode_candidate_handles_silence_and_invalid_control_metadata_is_rejected(self):
        banks, record = self.revise(audio=np.zeros(128),counts=[0])
        self.assertEqual(len(record['fresh_candidates']),1)
        self.assertEqual(record['fresh_candidates'][0]['frequency_hz'],[])
        forecast = banks['fresh'].forecast(64)
        np.testing.assert_array_equal(forecast.mean,np.zeros(64))
        self.assertTrue(np.isfinite(forecast.log_density(np.zeros(64))))
        self.assertTrue(np.all(forecast.diagonal_variance()>0))
        wrong = copy.deepcopy(self.rows)
        wrong[0]['frequency_hz']=[241.]
        with self.assertRaises(ValueError):
            self.revise(rows=wrong)
        for options in [dict(offsets_sec=[0.,.00001]),dict(counts=[1]),dict(prior_mass=dict(retained=0.,fresh=1.))]:
            with self.assertRaises(ValueError):
                self.revise(**options)


if __name__ == '__main__':
    unittest.main()
