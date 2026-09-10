#!/usr/bin/env python3
"""Conditional trajectory amplitudes and correlated AR residual state.

This conditions acoustic state, not perceptual source identity.
The linear posterior follows the weight-space calculation in Rasmussen and
Williams (2006), section 2.1, with zero prior precision after a full-rank fit:
https://gaussianprocess.org/gpml/chapters/RW2.pdf
Optional innovation-variance learning uses an inverse-Gamma posterior and a
joint Student prediction, conditional on the fitted trajectory/AR response.
Variance conjugacy conventions: Murphy (2007), sections 6 and 10.2,
https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
Explicit amplitude precision and variance_prior instead select a proper
Normal-inverse-Gamma prior, including rank-deficient amplitude directions.
Its log_evidence is conditional on the initial p samples and the AR response.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from evaluate_driven_acoustic_state import (analyze_lattice, auditory_power_moments,
                                           fit_burg, synthesize_lattice)


@dataclass(frozen=True)
class AcousticForecast:
    issued_sample: int
    mean: np.ndarray
    impulse: np.ndarray
    amplitude_factor: np.ndarray
    reflection: np.ndarray
    innovation_variance: float
    variance_shape: float | None = None
    variance_scale: float | None = None

    def with_known_waveform(self, audio):
        """Translate a forecast by a known signal, preserving its uncertainty.

        For owned actions, the input forecast must describe the environment
        after removing the known owned contribution from past observations.
        """
        audio=np.asarray(audio,dtype=float)
        if audio.shape!=self.mean.shape or not np.all(np.isfinite(audio)):
            raise ValueError("known signal must cover the complete forecast window")
        mean=self.mean+audio
        mean.setflags(write=False)
        return AcousticForecast(self.issued_sample,mean,self.impulse,self.amplitude_factor,
                                self.reflection,self.innovation_variance,self.variance_shape,self.variance_scale)

    def log_density(self, audio):
        """Joint density with amplitude and optional innovation-scale uncertainty."""
        audio=np.asarray(audio,dtype=float)
        if audio.shape!=self.mean.shape or not np.all(np.isfinite(audio)):
            raise ValueError("expected one complete finite forecast target")
        q=self.innovation_variance
        error=analyze_lattice(audio-self.mean,self.reflection)[0]/np.sqrt(q)
        width=self.amplitude_factor.shape[1]
        whitened=analyze_lattice(self.amplitude_factor,self.reflection)[0]/np.sqrt(q)
        augmented=np.block([[np.eye(width),np.zeros((width,1))],
                            [whitened,error[:,None]]])
        root=np.linalg.qr(augmented,mode='r')
        logdet=2*np.sum(np.log(np.abs(np.diag(root)[:width])))
        if self.variance_shape is not None:
            a,b=self.variance_shape,self.variance_scale
            return float(math.lgamma(a+len(audio)/2)-math.lgamma(a)
                -.5*(len(audio)*np.log(2*np.pi*b)+logdet)
                -(a+len(audio)/2)*np.log1p(q*root[-1,-1]**2/(2*b)))
        return float(-.5*(len(audio)*np.log(2*np.pi*q)+logdet+root[-1,-1]**2))

    def diagonal_variance(self):
        """Predictive covariance diagonal; innovation_variance is E[q] when learned."""
        return (np.sum(self.amplitude_factor**2,axis=1)
                +self.innovation_variance*np.cumsum(self.impulse**2))

    def sample(self, rng, count):
        """Draw whole correlated trajectories before a nonlinear observation.

        These samples retain this model's conditional uncertainty; they do not
        integrate model selection uncertainty or calibrate its predictions.
        """
        if not isinstance(count,int) or count<1:
            raise ValueError("expected a positive trajectory count")
        size=len(self.mean)
        width=self.amplitude_factor.shape[1]
        normal=rng.standard_normal((count,width+size))
        draws=(self.mean if self.variance_shape is None else np.zeros_like(self.mean))+normal[:,:width]@self.amplitude_factor.T
        for index in range(count):
            draws[index]+=np.sqrt(self.innovation_variance)*np.convolve(
                normal[index,width:],self.impulse)[:size]
        if self.variance_shape is not None:
            # One variance draw scales the entire correlated trajectory.
            scale=self.variance_scale/(self.innovation_variance*rng.gamma(self.variance_shape,1.,size=count))
            draws*=np.sqrt(scale[:,None])
            draws+=self.mean
        return draws

    def digest(self):
        digest=hashlib.sha256()
        for array in (self.mean,self.impulse,self.amplitude_factor,self.reflection):
            digest.update(np.asarray(array.shape,dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        digest.update(np.float64(self.innovation_variance).tobytes())
        if self.variance_shape is not None:
            digest.update(np.asarray([self.variance_shape,self.variance_scale],dtype=np.float64).tobytes())
        return digest.hexdigest()


class AcousticPosterior:
    """Keep trajectory/noise-response parameters while conditioning on new PCM."""

    def __init__(self,trajectory,residual,audio,start_sample,*,learn_variance=False,
                 amplitude_precision=None,variance_prior=None):
        audio=np.asarray(audio,dtype=float)
        order=len(residual.reflection)
        if (audio.ndim!=1 or not np.all(np.isfinite(audio)) or len(audio)<=order
                or not isinstance(start_sample,int) or start_sample<0
                or not np.isfinite(residual.drive_variance) or residual.drive_variance<=0
                or trajectory.fit_end_sample>start_sample+len(audio)
                or residual.fit_end_sample>start_sample+len(audio)):
            raise ValueError("expected observed fitting audio and already available parameters")
        self.trajectory,self.residual=trajectory,residual
        self.next_sample=start_sample+len(audio)
        self.conditioned_initial_samples=order
        raw=trajectory.basis(start_sample,len(audio))
        basis=np.column_stack((raw.real,raw.imag))
        self.width=basis.shape[1]
        proper=amplitude_precision is not None or variance_prior is not None
        if proper and (not learn_variance or amplitude_precision is None or variance_prior is None
                       or not np.isfinite(amplitude_precision) or amplitude_precision<=0
                       or len(variance_prior)!=2 or not np.isfinite(variance_prior).all()
                       or variance_prior[0]<=1 or variance_prior[1]<=0):
            raise ValueError('proper priors require positive amplitude precision and inverse-Gamma shape > 1 / scale > 0')
        if not proper and len(audio)-order<=self.width:
            raise ArithmeticError("insufficient evidence for a proper amplitude posterior")
        whitened,self.basis_state=analyze_lattice(basis,residual.reflection)
        observed,self.audio_state=analyze_lattice(audio,residual.reflection)
        # Condition on the first p actual samples; do not assume an unknown past is zero.
        design=whitened[order:]
        self.scales=np.linalg.norm(basis[order:],axis=0)
        self.log_evidence=None
        if proper:
            # Amplitudes are physical basis coefficients: c | q ~ N(0, q/lambda I).
            # The prior keeps duplicate or unobserved amplitude directions uncertain.
            self.scales[self.scales==0]=1.
            augmented=np.block([[np.diag(np.sqrt(amplitude_precision)/self.scales),np.zeros((self.width,1))],
                                [design/self.scales,observed[order:,None]]])
            root=np.linalg.qr(augmented,mode='r')
            self.root=root[:self.width,:self.width].copy()
            self.rhs=root[:self.width,-1].copy()
            self.observations=len(audio)-order
            a0,b0=variance_prior
            self.variance_shape=a0+self.observations/2
            self.variance_scale=b0+float(root[-1,-1]**2)/2
            logdet=2*np.sum(np.log(np.abs(np.diag(self.root))*self.scales))
            self.log_evidence=float(math.lgamma(self.variance_shape)-math.lgamma(a0)
                +a0*np.log(b0)-self.variance_shape*np.log(self.variance_scale)
                +.5*(self.width*np.log(amplitude_precision)-logdet-self.observations*np.log(2*np.pi)))
            return
        if np.any(self.scales==0):
            raise ArithmeticError("unidentified amplitude direction")
        scaled=design/self.scales
        if self.width:
            singular=np.linalg.svd(scaled,compute_uv=False)
            reference=np.linalg.norm(basis[order:]/self.scales,ord=2)
            # Whitening can annihilate a trajectory; normalizing its roundoff
            # alone would manufacture information from a mathematically zero column.
            floor=np.finfo(float).eps*max(design.shape)*reference
            if singular[-1]<=max(singular[0]*1e-10,floor):
                raise ArithmeticError("amplitude posterior is numerically rank deficient")
        orthogonal,self.root=np.linalg.qr(scaled,mode='reduced')
        self.rhs=orthogonal.T@observed[order:]
        self.observations=len(audio)-order
        self.variance_shape=self.variance_scale=None
        if learn_variance:
            # p(amplitude, q) proportional to 1/q, conditional on the chosen
            # trajectory and AR response. Finite predictive power needs E[q].
            error=observed[order:]-orthogonal@self.rhs
            self.variance_shape=(self.observations-self.width)/2
            self.variance_scale=float(error@error)/2
            if (self.variance_shape<=1 or not self.variance_scale>0
                    or not np.isfinite(self.variance_scale)):
                raise ArithmeticError("innovation-scale posterior needs positive residual energy and finite mean")

    def observe(self,start_sample,audio):
        """Return pre-update block density, then retain all new observed state."""
        audio=np.asarray(audio,dtype=float)
        if (start_sample!=self.next_sample or audio.ndim!=1 or not np.all(np.isfinite(audio))):
            raise ValueError("posterior observations must be contiguous finite PCM")
        if not len(audio):
            return 0.
        raw=self.trajectory.basis(start_sample,len(audio))
        basis=np.column_stack((raw.real,raw.imag))
        whitened,basis_state=analyze_lattice(basis,self.residual.reflection,self.basis_state)
        observed,audio_state=analyze_lattice(audio,self.residual.reflection,self.audio_state)
        augmented=np.block([[self.root,self.rhs[:,None]],
                            [whitened/self.scales,observed[:,None]]])
        root=np.linalg.qr(augmented,mode='r')
        logdet=2*(np.sum(np.log(np.abs(np.diag(root)[:self.width])))
                  -np.sum(np.log(np.abs(np.diag(self.root)))))
        q=self.residual.drive_variance
        if self.variance_shape is None:
            log_density=float(-.5*(len(audio)*np.log(2*np.pi*q)+logdet+root[-1,-1]**2/q))
        else:
            a,b=self.variance_shape,self.variance_scale
            log_density=float(math.lgamma(a+len(audio)/2)-math.lgamma(a)
                -.5*(len(audio)*np.log(2*np.pi*b)+logdet)
                -(a+len(audio)/2)*np.log1p(root[-1,-1]**2/(2*b)))
            self.variance_shape+=len(audio)/2
            self.variance_scale+=root[-1,-1]**2/2
        self.root=root[:self.width,:self.width].copy()
        self.rhs=root[:self.width,-1].copy()
        self.audio_state,self.basis_state=audio_state,basis_state
        self.next_sample+=len(audio)
        self.observations+=len(audio)
        if self.log_evidence is not None:
            self.log_evidence+=log_density
        return log_density

    def forecast(self,count):
        if not isinstance(count,int) or count<1:
            raise ValueError("expected a positive forecast sample count")
        raw=self.trajectory.basis(self.next_sample,count)
        basis=np.column_stack((raw.real,raw.imag))
        reflection=self.residual.reflection
        zeros=np.zeros(count)
        audio_future=synthesize_lattice(zeros,reflection,self.audio_state)[0]
        effective=basis-synthesize_lattice(np.zeros_like(basis),reflection,self.basis_state)[0]
        # Residual state depends on amplitude through x_past - B_past * amplitude.
        # Its covariance must be combined with B_future before forming the factor.
        scaled=effective/self.scales
        mean=audio_future+scaled@np.linalg.solve(self.root,self.rhs)
        q=(self.residual.drive_variance if self.variance_shape is None
           else self.variance_scale/(self.variance_shape-1))
        factor=np.sqrt(q)*scaled@np.linalg.solve(self.root,np.eye(self.width))
        drive=np.zeros(count);drive[0]=1.
        impulse=synthesize_lattice(drive,reflection,np.zeros(len(reflection)))[0]
        reflection=reflection.copy()
        for values in (mean,factor,impulse,reflection):
            values.setflags(write=False)
        return AcousticForecast(self.next_sample,mean,impulse,factor,reflection,q,
                                self.variance_shape,self.variance_scale)


def posterior_auditory_power(observer,forecast):
    """Conditional expected power including correlated amplitude-state uncertainty."""
    if (observer.next_sample!=forecast.issued_sample or observer.pending!=0
            or len(forecast.mean)%observer.stride):
        raise ValueError("auditory state must end at the forecast issue boundary")
    mean_power,with_drive=auditory_power_moments(observer,forecast.mean,forecast.impulse,
                                               forecast.innovation_variance)
    state=np.zeros((4,len(observer.pole),forecast.amplitude_factor.shape[1]),dtype=complex)
    variance=np.empty((len(forecast.mean),len(observer.pole)))
    for n,amplitudes in enumerate(forecast.amplitude_factor):
        state*=observer.pole[None,:,None]
        state[0]+=observer.gain[:,None]*amplitudes
        state[1]+=state[0];state[2]+=state[1];state[3]+=state[2]
        variance[n]=np.sum(state[3].real**2+state[3].imag**2,axis=1)
    shape=(-1,observer.stride,len(observer.pole))
    return mean_power,with_drive,with_drive+variance.reshape(shape).mean(axis=1)


def fixed_amplitude_forecast(trajectory,residual,observed,start_sample,count):
    """Ablation: condition residual state while treating fitted amplitudes as known."""
    history=observed-trajectory.predict(start_sample,len(observed))
    mean,impulse=residual.forecast(history,count)
    issued=start_sample+len(observed)
    mean+=trajectory.predict(issued,count)
    factor=np.empty((count,0))
    reflection=residual.reflection.copy()
    for values in (mean,impulse,factor,reflection):values.setflags(write=False)
    return AcousticForecast(issued,mean,impulse,factor,reflection,residual.drive_variance)


def select_acoustic_posterior(trajectories,train,validation,start_sample,*,
                             noise_orders=(0,4,8,16,32,64),variance_floor,forecast_samples):
    """Score fixed-parameter candidates on known audio before issuing new predictions."""
    train,validation=np.asarray(train,dtype=float),np.asarray(validation,dtype=float)
    issued=start_sample+len(train)+len(validation)
    if not len(validation) or forecast_samples<1:
        raise ValueError("expected validation evidence and a positive forecast horizon")
    best_posterior=best_fixed=None
    posterior_score=fixed_score=float('-inf')
    candidates=[]
    posterior_index=fixed_index=None
    for kind,trajectory in trajectories:
        try:
            trajectory.predict(issued,forecast_samples)
            past_residual=train-trajectory.predict(start_sample,len(train))
            new_residual=validation-trajectory.predict(start_sample+len(train),len(validation))
        except ArithmeticError as error:
            candidates.append(dict(kind=kind,components=len(trajectory.frequency_hz),unavailable=str(error)))
            continue
        for noise in fit_burg(past_residual,start_sample,orders=noise_orders,variance_floor=variance_floor):
            fixed_density=noise.log_density(past_residual,new_residual)
            row=dict(kind=kind,components=len(trajectory.frequency_hz),order=len(noise.reflection),
                     innovation_variance=noise.drive_variance,fixed_log_density=fixed_density)
            index=len(candidates)
            if np.isfinite(fixed_density) and fixed_density>fixed_score:
                best_fixed=(trajectory,noise);fixed_score=fixed_density;fixed_index=index
            try:
                posterior=AcousticPosterior(trajectory,noise,train,start_sample)
                density=posterior.observe(start_sample+len(train),validation)
                row['posterior_log_density']=density
                if np.isfinite(density) and density>posterior_score:
                    best_posterior=posterior;posterior_score=density;posterior_index=index
            except (ArithmeticError,np.linalg.LinAlgError) as error:
                row['posterior_unavailable']=str(error)
            candidates.append(row)
    if best_posterior is None or best_fixed is None:
        raise ArithmeticError("no available acoustic candidate")
    return dict(posterior=best_posterior,fixed=best_fixed,candidates=candidates,
                posterior_index=posterior_index,fixed_index=fixed_index)
