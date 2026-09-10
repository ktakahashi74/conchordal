import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_hypotheses import AmplitudeBasis
from acoustic_posterior import AcousticPosterior
from acoustic_trajectory import TrajectoryFit
from evaluate_acoustic_hypotheses import main
from evaluate_driven_acoustic_state import DrivenFit


class AcousticHypothesisTests(unittest.TestCase):
    def test_affine_amplitude_prediction_matches_independent_regression_covariance(self):
        fs=24000;start=1200;count=480;future=240;origin=start+(count-1)/2
        frequency=np.array([440.,981.])
        carrier=TrajectoryFit(fs,0.,frequency,np.zeros(2),np.array([852.,-669.]),np.ones(2,dtype=complex),0,start)
        basis=AmplitudeBasis(carrier,None,origin,count,1)
        t=np.arange(start,start+count+future)/fs
        u=(np.arange(start,start+count+future)-origin)/count
        cosine=np.cos(2*np.pi*t[:,None]*frequency);sine=np.sin(2*np.pi*t[:,None]*frequency)
        design=np.column_stack([cosine,u[:,None]*cosine,sine,u[:,None]*sine])
        coefficient=np.array([.06,.02,-.03,.01,-.02,.015,.005,-.004]);q=1e-7
        observed=design[:count]@coefficient+np.sqrt(q)*np.random.default_rng(851).normal(size=count)
        noise=DrivenFit(np.empty(0),q,start,start+count)
        forecast=AcousticPosterior(basis,noise,observed,start).forecast(future)
        fitted=np.linalg.lstsq(design[:count],observed,rcond=None)[0]
        covariance=q*(np.eye(future)+design[count:]@np.linalg.solve(design[:count].T@design[:count],design[count:].T))
        np.testing.assert_allclose(forecast.mean,design[count:]@fitted,rtol=1e-11,atol=1e-14)
        np.testing.assert_allclose(forecast.diagonal_variance(),np.diag(covariance),rtol=1e-11)
        target=design[count:]@coefficient
        error=target-forecast.mean
        density=-.5*(future*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]+error@np.linalg.solve(covariance,error))
        self.assertAlmostEqual(forecast.log_density(target),density,places=9)

    def test_identical_added_support_is_unidentifiable(self):
        fs=24000;start=1200;n=240
        carrier=TrajectoryFit(fs,0.,np.array([440.]),np.zeros(1),np.zeros(1),np.ones(1,dtype=complex),0,start)
        basis=AmplitudeBasis(carrier,carrier,start,n,0)
        observed=.04*np.cos(2*np.pi*440*np.arange(start,start+n)/fs)
        with self.assertRaisesRegex(ArithmeticError,'rank deficient'):
            AcousticPosterior(basis,DrivenFit(np.empty(0),1e-8,start,start+n),observed,start)

    def test_future_and_unchosen_outcome_change_scores_but_not_issued_hypotheses(self):
        fs=24000;issue=1200;n=480;used=240
        t=np.arange(issue+n)/fs
        outside=.05*np.cos(2*np.pi*777*t)
        own=.03*np.cos(2*np.pi*411*t)
        actual=outside[issue:]+own[issue:]
        plan=dict(base_fit_sec=.04,base_validation_sec=.01,adaptation_sec=.008,selection_sec=.002,horizon_sec=.01,
            counts=[0,1],orders=[0,4],variance_floor=1e-10,amplitude_degrees=[0,1],
            adaptation_offsets_sec=[0,.004,.006],added_counts=[0,1])
        case=dict(name='prefix',fs=fs,issue_sample=issue,external_condition='unknown')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'native';native.mkdir();source=native/'prefix';source.mkdir()
            (native/'inputs.json').write_text(json.dumps([case]));(root/'plan.json').write_text(json.dumps(plan))
            for key,audio in dict(past_mix=outside[:issue]+own[:issue],past_own=own[:issue],
                                  own_sound=own[issue:],truth_sound=actual,truth_wait=actual).items():
                np.asarray(audio,dtype='<f4').tofile(source/(key+'.f32le'))
            for attempt in range(2):
                if attempt:
                    changed=actual.copy();changed[used:]+=.3
                    changed.astype('<f4').tofile(source/'truth_sound.f32le')
                    np.full(n,.7,dtype='<f4').tofile(source/'truth_wait.f32le')
                with patch.object(sys,'argv',['evaluate_acoustic_hypotheses','--input',str(native),
                     '--output',str(root/str(attempt)),'--plan',str(root/'plan.json')]),contextlib.redirect_stdout(io.StringIO()):main()
            a=json.loads((root/'0/prefix.json').read_text());b=json.loads((root/'1/prefix.json').read_text())
            self.assertEqual(a,b)
            self.assertEqual(a['inference']['base_fit_end'],960)
            self.assertEqual(a['inference']['selection_end'],issue+used)
            np.testing.assert_allclose(a['inference']['retained_frequency_hz'],[777.],atol=.001)
            with np.load(root/'0/prefix.npz') as a,np.load(root/'1/prefix.npz') as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            a=json.loads((root/'0/summary.json').read_text());b=json.loads((root/'1/summary.json').read_text())
            self.assertNotEqual(a[0]['metrics']['refit']['pcm_mse'],b[0]['metrics']['refit']['pcm_mse'])


if __name__=='__main__':unittest.main()
