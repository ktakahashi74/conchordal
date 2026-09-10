from pathlib import Path
import copy
import math
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from acoustic_posterior import (AcousticPosterior, fixed_amplitude_forecast,
                                posterior_auditory_power, select_acoustic_posterior)
from acoustic_trajectory import TrajectoryFit, fit_trajectory_candidates
from evaluate_auditory_envelope import AuditoryEnvelopeObserver
from evaluate_driven_acoustic_state import DrivenFit


def example():
    fs=24000
    fit=TrajectoryFit(fs,600.,np.array([417.,863.]),np.array([1700.,-900.]),
                      np.array([-3.,-.7]),np.array([.1+.04j,-.05+.08j]),0,1200)
    residual=DrivenFit(np.array([-.9,.81]),.00004,0,1200)
    observed=fit.predict(0,1200)+np.random.default_rng(913).normal(0.,.01,1200)
    return fit,residual,observed


def independent_joint(fit,residual,observed,start,count):
    # Direct AR polynomial, normal equations and a full joint covariance.
    polynomial=np.array([1.])
    for k in residual.reflection:
        polynomial=np.pad(polynomial,(0,1))+k*np.pad(polynomial[::-1],(1,0))
    p=len(polynomial)-1
    basis=fit.basis(0,start+count)
    basis=np.column_stack((basis.real,basis.imag))
    whitened=np.column_stack([np.convolve(basis[:start,j],polynomial)[p:start]
                              for j in range(basis.shape[1])])
    target=np.convolve(observed,polynomial)[p:start]
    gram=whitened.T@whitened
    mean=np.linalg.solve(gram,whitened.T@target)
    covariance=residual.drive_variance*np.linalg.inv(gram)
    companion=np.zeros((p,p));companion[0]=-polynomial[1:]
    companion[1:,:-1]=np.eye(p-1)
    transition=companion.copy(); mapping=[];impulse=[]
    state=np.zeros(p);state[0]=1.
    for n in range(count):
        mapping.append(transition[0].copy());transition=transition@companion
        impulse.append(state[0]);state=companion@state
    mapping=np.array(mapping)
    previous_basis=basis[start-p:start][::-1]
    cross=-covariance@previous_basis.T
    joint_covariance=np.block([[covariance,cross],
                               [cross.T,previous_basis@covariance@previous_basis.T]])
    joint_mean=np.r_[mean,observed[-p:][::-1]-previous_basis@mean]
    projection=np.column_stack((basis[start:],mapping))
    noise=np.zeros((count,count))
    for n in range(count):noise[n: ,n]=impulse[:count-n]
    return projection@joint_mean,projection@joint_covariance@projection.T+residual.drive_variance*noise@noise.T


def independent_scale(fit,residual,audio):
    polynomial=np.array([1.])
    for coefficient in residual.reflection:
        polynomial=np.pad(polynomial,(0,1))+coefficient*np.pad(polynomial[::-1],(1,0))
    order=len(polynomial)-1
    raw=fit.basis(0,len(audio))
    basis=np.column_stack((raw.real,raw.imag))
    design=np.column_stack([np.convolve(basis[:,j],polynomial)[order:len(audio)]
                            for j in range(basis.shape[1])])
    target=np.convolve(audio,polynomial)[order:len(audio)]
    error=target-design@np.linalg.lstsq(design,target,rcond=None)[0]
    return (len(target)-design.shape[1])/2,float(error@error)/2


class AcousticPosteriorTests(unittest.TestCase):
    def test_proper_prior_matches_dense_marginal_and_student_prediction(self):
        fs, n, horizon = 24000, 45, 12
        fit = TrajectoryFit(fs, 0., np.array([417., 863.]), np.zeros(2),
                            np.array([-13., -7.]), np.zeros(2, complex), 0, n)
        residual = DrivenFit(np.array([]), .001, 0, n)
        audio = np.random.default_rng(1047).normal(0., .1, n)
        precision, a0, b0 = .07, 2.3, .008
        model = AcousticPosterior(fit, residual, audio, 0, learn_variance=True,
                                   amplitude_precision=precision, variance_prior=(a0, b0))
        raw = fit.basis(0, n+horizon)
        basis = np.column_stack((raw.real, raw.imag))
        train, future = basis[:n], basis[n:]
        marginal = np.eye(n)+train@train.T/precision
        a = a0+n/2
        b = b0+audio@np.linalg.solve(marginal, audio)/2
        evidence = (math.lgamma(a)-math.lgamma(a0)+a0*np.log(b0)-a*np.log(b)
                    -.5*(n*np.log(2*np.pi)+np.linalg.slogdet(marginal)[1]))
        self.assertAlmostEqual(model.log_evidence, evidence, places=10)
        cross = future@train.T/precision
        mean = cross@np.linalg.solve(marginal, audio)
        shape = np.eye(horizon)+future@future.T/precision-cross@np.linalg.solve(marginal, cross.T)
        prediction = model.forecast(horizon)
        np.testing.assert_allclose(prediction.mean, mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(prediction.diagonal_variance(), np.diag(shape)*b/(a-1), rtol=1e-10)
        target = mean+np.linspace(-.1, .13, horizon)
        error = target-mean
        density = (math.lgamma(a+horizon/2)-math.lgamma(a)
                   -.5*(horizon*np.log(2*np.pi*b)+np.linalg.slogdet(shape)[1])
                   -(a+horizon/2)*np.log1p(error@np.linalg.solve(shape, error)/(2*b)))
        self.assertAlmostEqual(prediction.log_density(target), density, places=9)

    def test_proper_prior_evidence_is_additive_with_ar_and_rank_deficiency(self):
        for frequencies in ([417., 863.], [417., 417.], []):
            k, n = len(frequencies), 84
            fit = TrajectoryFit(24000, 0., np.array(frequencies), np.zeros(k),
                                np.zeros(k), np.zeros(k, complex), 0, 32)
            noise = DrivenFit(np.array([-.7, .4]), .002, 0, 32)
            audio = np.random.default_rng(1051).normal(0., .1, n)
            kwargs = dict(learn_variance=True, amplitude_precision=.003, variance_prior=(2., .001))
            batch = AcousticPosterior(fit, noise, audio, 0, **kwargs)
            online = AcousticPosterior(fit, noise, audio[:32], 0, **kwargs)
            evidence = online.log_evidence
            for start, end in [(32, 33), (33, 49), (49, n)]:
                expected = online.forecast(end-start).log_density(audio[start:end])
                observed = online.observe(start, audio[start:end])
                self.assertAlmostEqual(observed, expected, places=9)
                evidence += observed
            self.assertAlmostEqual(evidence, batch.log_evidence, places=9)
            self.assertAlmostEqual(online.log_evidence, batch.log_evidence, places=9)
            for attr in ('mean', 'diagonal_variance'):
                x, y = getattr(batch.forecast(30), attr), getattr(online.forecast(30), attr)
                np.testing.assert_allclose(x() if callable(x) else x, y() if callable(y) else y,
                                           rtol=1e-9, atol=1e-12)

    def test_proper_prior_keeps_unobserved_amplitude_uncertainty(self):
        fit = TrajectoryFit(24000, 0., np.array([6000.]), np.zeros(1), np.zeros(1),
                            np.zeros(1, complex), 0, 1)
        noise = DrivenFit(np.array([]), .001, 0, 1)
        model = AcousticPosterior(fit, noise, np.zeros(1), 0, learn_variance=True,
                                   amplitude_precision=.1, variance_prior=(2., .001))
        # At t=0 the sine coefficient is unobserved; the next sample observes it.
        self.assertAlmostEqual(model.forecast(1).diagonal_variance()[0], .001/1.5*(1+10), places=14)
        self.assertTrue(np.isfinite(model.log_evidence))
        for kwargs in [dict(amplitude_precision=0., variance_prior=(2., .1)),
                       dict(amplitude_precision=.1, variance_prior=(1., .1)),
                       dict(amplitude_precision=.1, variance_prior=(2., 0.)),
                       dict(amplitude_precision=.1)]:
            with self.assertRaises(ValueError):
                AcousticPosterior(fit, noise, np.zeros(1), 0, learn_variance=True, **kwargs)

    def test_innovation_scale_learning_matches_joint_student_conditioning(self):
        fit,residual,audio=example()
        posterior=AcousticPosterior(fit,residual,audio,0,learn_variance=True)
        forecast=posterior.forecast(80)
        alpha,beta=independent_scale(fit,residual,audio)
        mean,covariance=independent_joint(fit,residual,audio,len(audio),80)
        shape=covariance/residual.drive_variance
        self.assertEqual(posterior.variance_shape,alpha)
        self.assertAlmostEqual(posterior.variance_scale,beta,places=14)
        np.testing.assert_allclose(forecast.diagonal_variance(),np.diag(shape)*beta/(alpha-1),rtol=1e-10,atol=3e-17)
        observed=mean[:20]+np.linspace(-.07,.11,20)
        error=observed-mean[:20]
        quadratic=error@np.linalg.solve(shape[:20,:20],error)
        density=(math.lgamma(alpha+10)-math.lgamma(alpha)
            -.5*(20*np.log(2*np.pi*beta)+np.linalg.slogdet(shape[:20,:20])[1])
            -(alpha+10)*np.log1p(quadratic/(2*beta)))
        self.assertAlmostEqual(posterior.forecast(20).log_density(observed),density,places=8)
        before=forecast.digest()
        chunked=copy.deepcopy(posterior)
        self.assertAlmostEqual(posterior.observe(1200,observed),density,places=8)
        chunked_density=chunked.observe(1200,observed[:7])+chunked.observe(1207,observed[7:])
        self.assertAlmostEqual(chunked_density,density,places=8)
        self.assertEqual(posterior.variance_shape,alpha+10)
        self.assertAlmostEqual(posterior.variance_scale,beta+quadratic/2,places=12)
        batch_alpha,batch_beta=independent_scale(fit,residual,np.r_[audio,observed])
        self.assertEqual(posterior.variance_shape,batch_alpha)
        self.assertAlmostEqual(posterior.variance_scale,batch_beta,places=12)
        coupling=shape[20:,:20]@np.linalg.inv(shape[:20,:20])
        expected_mean=mean[20:]+coupling@error
        expected_shape=shape[20:,20:]-coupling@shape[:20,20:]
        next_forecast=posterior.forecast(60)
        np.testing.assert_allclose(next_forecast.mean,expected_mean,rtol=1e-10,atol=4e-14)
        np.testing.assert_allclose(next_forecast.diagonal_variance(),
            np.diag(expected_shape)*(beta+quadratic/2)/(alpha+9),rtol=1e-10,atol=3e-17)
        np.testing.assert_allclose(chunked.forecast(60).diagonal_variance(),next_forecast.diagonal_variance(),rtol=1e-12)
        self.assertEqual(forecast.digest(),before)
        shape_before=(posterior.variance_shape,posterior.variance_scale)
        with self.assertRaises(ValueError):posterior.observe(1221,observed)
        self.assertEqual((posterior.variance_shape,posterior.variance_scale),shape_before)

    def test_scale_mixture_draws_have_the_independent_student_covariance(self):
        fit,residual,audio=example()
        posterior=AcousticPosterior(fit,residual,audio,0,learn_variance=True)
        forecast=posterior.forecast(12)
        alpha,beta=independent_scale(fit,residual,audio)
        mean,covariance=independent_joint(fit,residual,audio,len(audio),12)
        covariance*=beta/(alpha-1)/residual.drive_variance
        draws=forecast.sample(np.random.default_rng(185017),40000)
        error=draws.mean(axis=0)-mean
        self.assertLess(np.max(np.abs(error)/np.sqrt(np.diag(covariance)/len(draws))),5.)
        empirical=np.cov(draws,rowvar=False,bias=True)
        self.assertLess(np.max(np.abs(empirical-covariance)/np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))),.04)
        owned=np.linspace(0.,.1,12)
        np.testing.assert_allclose(forecast.with_known_waveform(owned).sample(np.random.default_rng(185017),40000),
                                   draws+owned,atol=3e-17)

    def test_scale_learning_rejects_an_improper_zero_energy_posterior(self):
        empty=np.array([])
        fit=TrajectoryFit(24000,0.,empty,empty,empty,np.array([],dtype=complex),0,16)
        residual=DrivenFit(empty,1e-12,0,16)
        with self.assertRaises(ArithmeticError):
            AcousticPosterior(fit,residual,np.zeros(16),0,learn_variance=True)
        # The fixed-variance control remains well-defined on this input.
        AcousticPosterior(fit,residual,np.zeros(16),0).forecast(4)

    def test_trajectory_samples_reproduce_independent_joint_moments(self):
        fit,residual,audio=example()
        forecast=AcousticPosterior(fit,residual,audio,0).forecast(12)
        expected_mean,expected_cov=independent_joint(fit,residual,audio,len(audio),12)
        dimension=forecast.amplitude_factor.shape[1]+len(forecast.mean)
        class IsotropicPoints:
            def standard_normal(self,shape):
                assert shape==(2*dimension,dimension)
                return np.r_[np.eye(dimension),-np.eye(dimension)]*np.sqrt(dimension)
        draws=forecast.sample(IsotropicPoints(),2*dimension)
        centered=draws-draws.mean(axis=0)
        np.testing.assert_allclose(draws.mean(axis=0),expected_mean,atol=3e-14,rtol=1e-11)
        np.testing.assert_allclose(centered.T@centered/len(draws),expected_cov,atol=3e-17,rtol=1e-10)
        owned=np.linspace(0.,.1,12)
        np.testing.assert_allclose(forecast.with_known_waveform(owned).sample(IsotropicPoints(),2*dimension),
                                   draws+owned,atol=3e-17)
        with self.assertRaises(ValueError):forecast.sample(np.random.default_rng(1),0)

    def test_correlated_amplitude_residual_covariance_matches_dense_joint_state(self):
        fit,residual,audio=example()
        posterior=AcousticPosterior(fit,residual,audio,0)
        forecast=posterior.forecast(160)
        mean,covariance=independent_joint(fit,residual,audio,len(audio),160)
        np.testing.assert_allclose(forecast.mean,mean,atol=3e-14,rtol=1e-11)
        noise=np.zeros((160,160))
        for j in range(160):noise[j:,j]=forecast.impulse[:160-j]
        actual=forecast.amplitude_factor@forecast.amplitude_factor.T+residual.drive_variance*noise@noise.T
        np.testing.assert_allclose(actual,covariance,atol=3e-17,rtol=1e-10)
        np.testing.assert_allclose(forecast.diagonal_variance(),np.diag(covariance),atol=3e-17)

    def test_observation_update_and_density_match_dense_gaussian_conditioning(self):
        fit,residual,audio=example()
        posterior=AcousticPosterior(fit,residual,audio,0)
        mean,covariance=independent_joint(fit,residual,audio,len(audio),160)
        observed=mean[:80]+np.linalg.cholesky(covariance[:80,:80])@np.random.default_rng(191).normal(size=80)
        error=observed-mean[:80]
        expected_density=-.5*(80*np.log(2*np.pi)+np.linalg.slogdet(covariance[:80,:80])[1]
                               +error@np.linalg.solve(covariance[:80,:80],error))
        issued=posterior.forecast(80)
        before=issued.digest()
        self.assertAlmostEqual(issued.log_density(observed),expected_density,places=9)
        self.assertAlmostEqual(posterior.observe(1200,observed),expected_density,places=9)
        prediction=posterior.forecast(80)
        coupling=covariance[80:,:80]@np.linalg.inv(covariance[:80,:80])
        expected_mean=mean[80:]+coupling@error
        expected_cov=covariance[80:,80:]-coupling@covariance[:80,80:]
        noise=np.zeros((80,80))
        for j in range(80):noise[j:,j]=prediction.impulse[:80-j]
        actual_cov=prediction.amplitude_factor@prediction.amplitude_factor.T+residual.drive_variance*noise@noise.T
        np.testing.assert_allclose(prediction.mean,expected_mean,atol=4e-14,rtol=1e-10)
        np.testing.assert_allclose(actual_cov,expected_cov,atol=4e-17,rtol=1e-10)
        self.assertEqual(issued.digest(),before)

    def test_chunked_conditioning_and_unavailable_input(self):
        fit,residual,audio=example()
        whole=AcousticPosterior(fit,residual,audio,0)
        split=copy.deepcopy(whole)
        new=fit.predict(1200,480)+np.random.default_rng(119).normal(0.,.01,480)
        score=whole.observe(1200,new)
        accumulated=0.
        for n in range(0,480,37):accumulated+=split.observe(1200+n,new[n:n+37])
        self.assertAlmostEqual(score,accumulated,places=8)
        a,b=whole.forecast(240),split.forecast(240)
        np.testing.assert_allclose(a.mean,b.mean,atol=4e-14,rtol=1e-10)
        np.testing.assert_allclose(a.amplitude_factor@a.amplitude_factor.T,
                                   b.amplitude_factor@b.amplitude_factor.T,atol=3e-17,rtol=1e-10)
        before=a.digest()
        with self.assertRaises(ValueError):whole.observe(1700,np.zeros(80))
        with self.assertRaises(ValueError):whole.observe(1680,np.array([np.nan]))
        self.assertEqual(whole.forecast(240).digest(),before)

    def test_no_trajectory_reduces_to_existing_driven_state(self):
        fit,residual,audio=example()
        empty=TrajectoryFit(fit.fs,600.,np.empty(0),np.empty(0),np.empty(0),np.empty(0,dtype=complex),0,1200)
        posterior=AcousticPosterior(empty,residual,audio,0)
        predicted=posterior.forecast(240)
        mean,impulse=residual.forecast(audio,240)
        np.testing.assert_array_equal(predicted.mean,mean)
        np.testing.assert_array_equal(predicted.impulse,impulse)
        observed=np.random.default_rng(73).normal(0.,.01,240)
        self.assertAlmostEqual(predicted.log_density(observed),residual.log_density(audio,observed),places=8)
        white=DrivenFit(np.empty(0),.0001,0,1200)
        state=AcousticPosterior(fit,white,audio,0)
        self.assertEqual(state.conditioned_initial_samples,0)
        self.assertTrue(np.all(np.isfinite(state.forecast(240).mean)))

    def test_rank_deficiency_is_not_zero_uncertainty(self):
        fit,residual,audio=example()
        duplicate=TrajectoryFit(fit.fs,600.,np.array([417.,417.]),np.zeros(2),np.zeros(2),
                                np.ones(2,dtype=complex)*.1,0,1200)
        with self.assertRaises(ArithmeticError):AcousticPosterior(duplicate,residual,audio,0)

    def test_whitening_annihilation_is_not_amplitude_information(self):
        fit=TrajectoryFit(24000,0.,np.array([6000.]),np.zeros(1),np.zeros(1),
                          np.array([.1+0j]),0,960)
        # x[n] + x[n-2] annihilates an exact quarter-sample-rate sinusoid.
        residual=DrivenFit(np.array([0.,1.]),.0001,0,960)
        with self.assertRaises(ArithmeticError):
            AcousticPosterior(fit,residual,fit.predict(0,960),0)

    def test_candidate_selection_uses_past_density_and_preserves_fitted_trajectories(self):
        fs=24000
        t=np.arange(1440)/fs
        audio=.1*np.cos(2*np.pi*(417*t+700*t*t))
        audio+=np.random.default_rng(137).normal(0.,.002,len(t))
        models=fit_trajectory_candidates(audio[:960],0,fs,counts=(0,1,2))
        before=[m.predict(1200,240).copy() for m in models]
        result=select_acoustic_posterior([('glide',m) for m in models],audio[:960],audio[960:1200],0,
                                         noise_orders=(0,4,16),variance_floor=(1/32768)**2/12,forecast_samples=240)
        candidate=result['candidates'][result['posterior_index']]
        self.assertEqual(candidate['posterior_log_density'],
                         max(r['posterior_log_density'] for r in result['candidates'] if 'posterior_log_density' in r))
        self.assertEqual(result['posterior'].next_sample,1200)
        fixed=fixed_amplitude_forecast(*result['fixed'],audio[:1200],0,240)
        predicted=result['posterior'].forecast(240)
        self.assertTrue(np.isfinite(fixed.log_density(audio[1200:])))
        self.assertTrue(np.isfinite(predicted.log_density(audio[1200:])))
        for m,prior in zip(models,before):np.testing.assert_array_equal(m.predict(1200,240),prior)

    def test_auditory_moments_match_independent_filter_matrix(self):
        fit,residual,audio=example()
        posterior=AcousticPosterior(fit,residual,audio,0)
        forecast=posterior.forecast(240)
        listener=AuditoryEnvelopeObserver(fit.fs,np.log2([250.,500.,1000.]),stride_samples=24)
        listener.process(0,audio)
        before=listener.state.copy()
        mean_power,drive_power,total=posterior_auditory_power(listener,forecast)
        np.testing.assert_array_equal(listener.state,before)
        _,covariance=independent_joint(fit,residual,audio,1200,240)
        expected=np.empty_like(total)
        for band,(pole,gain) in enumerate(zip(listener.pole,listener.gain)):
            impulse=np.array([gain*math.comb(n+3,3)*pole**n for n in range(240)])
            transform=np.zeros((240,240),dtype=complex)
            for n in range(240):transform[n:,n]=impulse[:240-n]
            variance=np.einsum('ij,ij->i',transform@covariance,transform.conj()).real
            expected[:,band]=mean_power[:,band]+variance.reshape(-1,24).mean(axis=1)
        np.testing.assert_allclose(total,expected,atol=3e-17,rtol=2e-10)
        self.assertTrue(np.all(total>=drive_power))


if __name__=='__main__':unittest.main()
