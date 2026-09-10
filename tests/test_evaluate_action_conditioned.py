from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_posterior import AcousticForecast,posterior_auditory_power
from evaluate_action_conditioned import forecast_branches, main
from evaluate_auditory_envelope import AuditoryEnvelopeObserver
from evaluate_driven_acoustic_state import auditory_power_moments


class ActionConditionedTests(unittest.TestCase):
    def test_known_signal_translates_joint_density_without_changing_uncertainty(self):
        n=64
        impulse=np.exp(-np.arange(n)/10.)
        factor=np.column_stack([np.linspace(0.,.01,n),np.cos(np.arange(n))*.005])
        original=AcousticForecast(0,np.sin(np.arange(n))*.03,impulse,factor,np.array([-np.exp(-.1)]),.001)
        known=np.cos(np.arange(n)*.5)*.1
        before=original.digest()
        shifted=original.with_known_waveform(known)
        target=shifted.mean+np.random.default_rng(183).normal(0.,.02,n)
        self.assertAlmostEqual(shifted.log_density(target),original.log_density(target-known),places=10)
        np.testing.assert_array_equal(shifted.diagonal_variance(),original.diagonal_variance())
        known[:]=100.
        self.assertEqual(original.digest(),before)
        self.assertFalse(shifted.mean.flags.writeable)
        with self.assertRaises(ValueError):original.with_known_waveform(np.zeros(n-1))

    def test_coherent_cancellation_keeps_cross_term_through_the_auditory_filter(self):
        fs=24000;n=240
        signal=.1*np.cos(2*np.pi*440*np.arange(n)/fs)
        impulse=np.zeros(n);impulse[0]=1.
        forecast=AcousticForecast(24,signal,impulse,np.empty((n,0)),np.empty(0),1e-10)
        observer=AuditoryEnvelopeObserver(fs,np.array([np.log2(440.)]),stride_samples=24)
        observer.process(0,np.zeros(24))
        cancelled=forecast.with_known_waveform(-signal)
        power=posterior_auditory_power(observer,cancelled)[2]
        noise_only=auditory_power_moments(observer,np.zeros(n),impulse,1e-10)[1]
        np.testing.assert_allclose(power,noise_only,rtol=1e-13,atol=0.)
        independent_sum=posterior_auditory_power(observer,forecast)[2]+auditory_power_moments(observer,-signal,np.zeros(n),0.)[0]
        self.assertGreater(float(np.max(independent_sum)),1000*float(np.max(power)))

    def test_environment_fit_uses_the_observed_waveform_after_own_subtraction(self):
        fs=24000;issue=1200;n=240
        t=np.arange(issue+n)/fs
        outside=.06*np.cos(2*np.pi*777*t)
        own=.09*np.cos(2*np.pi*411*t)
        plan=dict(fit_sec=.04,validation_sec=.01,trajectory_counts=[0,1],residual_orders=[0,4],variance_floor=1e-10)
        a,branches,meta=forecast_branches(outside[:issue]+own[:issue],own[:issue],own[issue:],np.zeros(n),fs,issue,plan)
        b,_,other=forecast_branches(outside[:issue],np.zeros(issue),np.zeros(n),np.zeros(n),fs,issue,plan)
        np.testing.assert_allclose(a.mean,b.mean,atol=2e-9)
        np.testing.assert_allclose(branches['wait'].mean-branches['sound'].mean,own[issue:],atol=1e-15)
        self.assertEqual(meta['trajectory_fit_end'],960)
        self.assertEqual(other['external_evidence_start'],0)

    def test_outcome_update_cannot_read_unchosen_branch_or_unobserved_suffix(self):
        fs=24000;issue=1200;horizon=480;observed=240
        time=np.arange(issue+horizon)/fs
        outside=.06*np.cos(2*np.pi*777*time)
        own=.03*np.cos(2*np.pi*411*time)
        acted=own[issue:]+.04*np.sin(2*np.pi*530*time[issue:])
        chosen=outside[issue:]+acted+.01*np.sin(2*np.pi*900*time[issue:])
        plan=dict(fit_sec=.04,validation_sec=.01,trajectory_counts=[0,1],residual_orders=[0,4],
                  variance_floor=1e-10,reobserve_sec=.01,chosen_branch='sound')
        case=dict(name='prefix',fs=fs,issue_sample=issue,body='test',route='habitat',external_condition='test')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'native';native.mkdir();source=native/'prefix';source.mkdir()
            (native/'inputs.json').write_text(json.dumps([case]))
            (root/'plan.json').write_text(json.dumps(plan))
            for key,audio in dict(past_mix=outside[:issue]+own[:issue],past_own=own[:issue],
                                  own_wait=own[issue:],own_sound=acted,truth_sound=chosen,
                                  truth_wait=outside[issue:]+own[issue:]).items():
                np.asarray(audio,dtype='<f4').tofile(source/(key+'.f32le'))
            for attempt in range(2):
                if attempt:
                    altered=chosen.copy();altered[observed:]+=.2
                    altered.astype('<f4').tofile(source/'truth_sound.f32le')
                    np.full(horizon,.3,dtype='<f4').tofile(source/'truth_wait.f32le')
                argv=['evaluate_action_conditioned','--input',str(native),'--output',str(root/str(attempt)),
                      '--plan',str(root/'plan.json')]
                with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):main()
            for suffix in ['forecast','updated']:
                left=json.loads((root/'0'/f'prefix-{suffix}.json').read_text())
                right=json.loads((root/'1'/f'prefix-{suffix}.json').read_text())
                self.assertEqual(left,right)
                with np.load(root/'0'/f'prefix-{suffix}.npz') as a,np.load(root/'1'/f'prefix-{suffix}.npz') as b:
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            a=json.loads((root/'0'/'summary.json').read_text())[0]['update']
            b=json.loads((root/'1'/'summary.json').read_text())[0]['update']
            self.assertEqual(a['observed_end_sample'],issue+observed)
            self.assertNotEqual(a['metrics']['pcm_mse'],b['metrics']['pcm_mse'])


if __name__=='__main__':unittest.main()
