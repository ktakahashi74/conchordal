"""Continuous passive acoustic parameters conditional on mode count and AR response.

Uniform log-frequency and decay priors live on an explicit bounded domain.
Metropolis updates use symmetric reflected Gaussian or uniform proposals;
step sizes adapt only during discarded warmup. See Hastings (1970):
https://doi.org/10.1093/biomet/57.1.97
Rank-normalized split/folded R-hat and bulk ESS follow Vehtari et al. (2021):
https://arxiv.org/abs/1903.08008
Finite chains and these diagnostics do not establish posterior convergence,
perceptual source identity, or cognitive retention times.
"""
from statistics import NormalDist

import numpy as np

from acoustic_mixture import AcousticMixtureForecast, AcousticModelBank
from acoustic_posterior import AcousticForecast, AcousticPosterior
from acoustic_trajectory import TrajectoryFit
from evaluate_driven_acoustic_state import synthesize_lattice


def parameter_forecast(bank, count):
    """Batch independent lattice columns without changing their arithmetic order."""
    if not isinstance(count, int) or count < 1:
        raise ValueError('expected a positive forecast sample count')
    first = bank.models[0]
    reflection, width = first.residual.reflection.copy(), first.width
    if any(model.width != width or not np.array_equal(model.residual.reflection, reflection)
           or not np.array_equal(model.audio_state, first.audio_state) for model in bank.models):
        raise ValueError('parameter bank must share amplitude width and observed AR state')
    zeros = np.zeros(count)
    continuation = synthesize_lattice(zeros, reflection, first.audio_state)[0]
    drive = zeros.copy(); drive[0] = 1.
    impulse = synthesize_lattice(drive, reflection, np.zeros(len(reflection)))[0]
    # Bound peak scratch storage while retaining all posterior components and weights.
    components = []
    for begin in range(0, len(bank.models), 128):
        group = bank.models[begin:begin+128]
        states = np.concatenate([model.basis_state for model in group], axis=1)
        response = synthesize_lattice(np.zeros((count, len(group)*width)), reflection, states)[0]
        for index, model in enumerate(group):
            raw = model.trajectory.basis(bank.next_sample, count)
            basis = np.column_stack((raw.real, raw.imag))
            effective = (basis-response[:, index*width:(index+1)*width])/model.scales
            mean = continuation+effective@np.linalg.solve(model.root, model.rhs)
            q = model.variance_scale/(model.variance_shape-1)
            factor = np.sqrt(q)*effective@np.linalg.solve(model.root, np.eye(width))
            for values in (mean, factor, impulse, reflection):
                values.setflags(write=False)
            components.append(AcousticForecast(bank.next_sample, mean, impulse, factor, reflection, q,
                                                model.variance_shape, model.variance_scale))
    return AcousticMixtureForecast(tuple(components), bank.log_weights.copy())


def chain_diagnostics(values):
    """Diagnose scalar functions of samples; axes are chain, draw, variable."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 8
            or not np.isfinite(values).all()):
        raise ValueError('expected at least two finite chains with eight draws')
    half = values.shape[1]//2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    result = []
    for variable in range(values.shape[2]):
        original = split[:, :, variable]
        statistics = []
        for x in (original, np.abs(original-np.median(original))):
            flat = x.ravel()
            _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
            ranks = (np.cumsum(counts)-(counts-1)/2)[inverse]
            z = np.array([NormalDist().inv_cdf(p) for p in
                          (ranks-.375)/(len(flat)+.25)]).reshape(x.shape)
            within = np.mean(np.var(z, axis=1, ddof=1))
            between = half*np.var(z.mean(axis=1), ddof=1)
            variance = (half-1)/half*within+between/half
            rhat = (np.sqrt(variance/within) if within else
                    (1. if not between else float('inf')))
            # Biased FFT autocovariances and Geyer's initial positive monotone pairs.
            centered = z-z.mean(axis=1, keepdims=True)
            spectrum = np.fft.rfft(centered, n=2*half, axis=1)
            covariance = np.fft.irfft(abs(spectrum)**2, n=2*half, axis=1)[:, :half]/half
            if variance == 0:
                ess = float(z.size)
            else:
                rho = 1-(within-covariance.mean(axis=0))/variance
                rho[0] = 1.
                pairs = []
                for lag in range(0, half-1, 2):
                    pair = rho[lag]+rho[lag+1]
                    if pair < 0:
                        break
                    pairs.append(min(pair, pairs[-1]) if pairs else pair)
                tau = max(-1+2*sum(pairs), 1/np.log10(z.size))
                ess = z.size/tau
            statistics.append((float(rhat), float(ess)))
        rhat = max(statistics[0][0], statistics[1][0])
        result.append(dict(rhat=rhat if np.isfinite(rhat) else None,
                           bulk_ess=statistics[0][1]))
    return result


def sample_passive_parameters(audio, start_sample, fs, residual, initial, *,
                              frequency_range, max_decay, amplitude_precision,
                              variance_prior, rng, warmup, draws, thin=1,
                              global_probability=.1):
    """Return an empirical posterior bank, a point control, and replayable draws.

Initial arrays have axes chain, mode, (Hz, log-gain/sec). Each mode has the
same prior, so labels are exchangeable. The amplitude origin is always the
observed block start, independent of decay. No target audio is accepted here.
The point control is the highest marginal density visited, not a global MAP.
"""
    audio, initial = np.asarray(audio, dtype=float), np.asarray(initial, dtype=float)
    band = np.asarray(frequency_range, dtype=float)
    if (not isinstance(fs, int) or fs < 16000 or not isinstance(start_sample, int) or start_sample < 0
            or audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) <= len(residual.reflection)
            or band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] < band[1] < fs/2
            or not np.isfinite(max_decay) or max_decay <= 0
            or initial.ndim != 3 or initial.shape[0] < 2 or initial.shape[2] != 2
            or not np.isfinite(initial).all()
            or any(not isinstance(n, int) for n in (warmup, draws, thin))
            or warmup < 0 or draws < 8 or thin < 1 or not 0 < global_probability <= 1):
        raise ValueError('expected observed audio, explicit priors and finite multi-chain sampling limits')
    chains, modes, _ = initial.shape
    if (np.any(initial[:, :, 0] < band[0]) or np.any(initial[:, :, 0] > band[1])
            or np.any(initial[:, :, 1] > 0) or np.any(initial[:, :, 1] < -max_decay)):
        raise ValueError('initial parameters must lie inside the prior domain')
    lo, span = np.log2(band[0]), np.log2(band[1]/band[0])
    unit = initial.copy()
    unit[:, :, 0] = (np.log2(initial[:, :, 0])-lo)/span
    unit[:, :, 1] = -initial[:, :, 1]/max_decay

    def conditional(coordinates):
        frequency = 2**(lo+span*coordinates[:, 0])
        gain = -max_decay*coordinates[:, 1]
        fit = TrajectoryFit(fs, float(start_sample), frequency, np.zeros(modes), gain,
                            np.zeros(modes, complex), start_sample, start_sample+len(audio))
        return AcousticPosterior(fit, residual, audio, start_sample, learn_variance=True,
                                  amplitude_precision=amplitude_precision, variance_prior=variance_prior)

    current = [conditional(x) for x in unit]
    best = max(current, key=lambda model: model.log_evidence)
    steps = np.full((chains, modes, 2), .03)
    accepted = np.zeros_like(steps)
    attempted = np.zeros_like(steps)
    local_accepted = np.zeros_like(steps)
    local_attempted = np.zeros_like(steps)
    parameters = np.empty((chains, draws, modes, 2))
    densities = np.empty((chains, draws))
    retained = [[] for _ in range(chains)]
    for sweep in range(warmup+draws*thin):
        for chain in range(chains):
            for coordinate in rng.permutation(2*modes):
                mode, axis = divmod(int(coordinate), 2)
                proposal = unit[chain].copy()
                local = rng.random() >= global_probability
                if local:
                    value = proposal[mode, axis]+steps[chain, mode, axis]*rng.normal()
                    proposal[mode, axis] = 1-abs(value % 2-1)
                else:
                    proposal[mode, axis] = rng.random()
                candidate = conditional(proposal)
                if not np.isfinite(candidate.log_evidence):
                    raise ArithmeticError('nonfinite conditional marginal density')
                take = np.log(rng.random()) < candidate.log_evidence-current[chain].log_evidence
                if candidate.log_evidence > best.log_evidence:
                    best = candidate
                if take:
                    unit[chain], current[chain] = proposal, candidate
                if sweep < warmup and local:
                    local_attempted[chain, mode, axis] += 1
                    local_accepted[chain, mode, axis] += take
                elif sweep >= warmup:
                    attempted[chain, mode, axis] += 1
                    accepted[chain, mode, axis] += take
        if sweep < warmup and (sweep+1) % 32 == 0:
            rate = np.divide(local_accepted, local_attempted, out=np.full_like(steps, .44),
                             where=local_attempted > 0)
            steps *= np.exp(rate-.44)
            np.clip(steps, 1e-6, .5, out=steps)
            local_accepted.fill(0)
            local_attempted.fill(0)
        if sweep >= warmup and (sweep-warmup+1) % thin == 0:
            index = (sweep-warmup+1)//thin-1
            parameters[:, index, :, 0] = 2**(lo+span*unit[:, :, 0])
            parameters[:, index, :, 1] = -max_decay*unit[:, :, 1]
            for chain in range(chains):
                # Repeated MCMC states retain multiplicity without sharing mutable updates.
                retained[chain].append(conditional(unit[chain]))
                densities[chain, index] = current[chain].log_evidence
    models = [model for chain in retained for model in chain]
    bank = AcousticModelBank(models, np.ones(len(models)))
    ordered = np.take_along_axis(parameters, np.argsort(parameters[:, :, :, 0], axis=2)[..., None], axis=2)
    diagnostic_values = np.concatenate((ordered.reshape(chains, draws, 2*modes), densities[:, :, None]), axis=2)
    diagnostics = chain_diagnostics(diagnostic_values)
    report = dict(parameters=parameters, log_density=densities, diagnostics=diagnostics,
                  acceptance=np.divide(accepted, attempted, out=np.zeros_like(steps), where=attempted > 0),
                  frozen_step_size=steps, point_log_density=best.log_evidence,
                  diagnostic_names=[f'{name}_{i}' for i in range(modes) for name in ('frequency_hz', 'log_gain_per_sec')]
                                   + ['log_density'])
    return bank, best, report
