//! Frozen temporal evidence and allocation-reusing consumer compatibility.
use super::{HMAX, Kernel, Prediction, Source, Value};
use crate::core::harmonic_support::{
    Snapshot,
    preview::{Observer, Preview},
};
use crate::core::landscape::Landscape;
// Frozen development IQRs; provenance: HV-9 additive calibration.
// HV-8 development nonzero paired differences: type-7 IQR, floored at 1e-6.
const CONSEQUENCE_SCALES: [f64; 2] = [0.006584158711807825, 0.0002224898411490413];

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub(crate) struct Decision {
    pub prediction: Prediction,
    pub variances: [f64; 9],
    pub shape_variance: f64,
    pub expires_at: f64,
}
impl Decision {
    pub(crate) fn new(prediction: Prediction, variances: [f64; 9], shape_variance: f64) -> Self {
        Self {
            prediction,
            variances,
            shape_variance,
            expires_at: f64::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Evidence {
    masks: [u8; HMAX],
    values: [f64; HMAX],
}

#[derive(Clone, Copy, Debug)]
struct Reference {
    pitch: Option<f32>,
    source: Option<Source>,
    evidence: Evidence,
}

#[derive(Clone, Debug)]
struct Work {
    preview: Preview,
    likelihoods: Vec<f64>,
    candidates: Vec<(Option<f32>, Option<Source>, f64, bool)>,
    references: [Option<Reference>; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct Use {
    pub weight: f64,
    pub continuation: f64,
    decision: Option<Arc<Decision>>,
    baseline: Arc<Snapshot>,
    observer: Observer,
    source: Option<Source>,
    template: Option<Arc<[(isize, f32)]>>,
    single_density: f32,
    fs: f64,
    sample: Cell<u64>,
    work: RefCell<Work>,
}
impl Use {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refresh(
        slot: &mut Option<Self>,
        landscape: &Landscape,
        source: Option<Source>,
        observer: Observer,
        baseline: Option<Arc<Snapshot>>,
        template: Option<Arc<[(isize, f32)]>>,
        single_density: f32,
        fs: f32,
        sample: u64,
    ) {
        if landscape.harmonic_relations_weight == 0. && landscape.harmonic_continuation_weight == 0.
        {
            if let Some(active) = slot {
                active.weight = 0.;
                active.continuation = 0.;
                active.decision = None;
            }
            return;
        }
        let Some(baseline) = baseline else {
            if let Some(active) = slot {
                active.decision = None;
            }
            return;
        };
        if slot
            .as_ref()
            .is_none_or(|active| active.work.borrow().preview.p.len() != baseline.p.len())
        {
            *slot = Some(Self {
                weight: landscape.harmonic_relations_weight,
                continuation: landscape.harmonic_continuation_weight,
                decision: None,
                observer,
                source,
                template: None,
                single_density,
                fs: f64::from(fs),
                sample: Cell::new(sample),
                work: RefCell::new(Work {
                    preview: Preview::new(&baseline),
                    references: [None; 2],
                    likelihoods: Vec::with_capacity(baseline.geometry.space.n_bins() * 2),
                    candidates: Vec::with_capacity(baseline.geometry.space.n_bins() * 2),
                }),
                baseline: baseline.clone(),
            });
        }
        let active = slot.as_mut().unwrap();
        active.weight = landscape.harmonic_relations_weight;
        active.continuation = landscape.harmonic_continuation_weight;
        active.decision = landscape.harmonic_relations.clone();
        active.baseline = baseline;
        active.observer = observer;
        active.source = source;
        active.template = template;
        active.single_density = single_density;
        active.fs = f64::from(fs);
        active.sample.set(sample);
    }

    pub(crate) fn set_sample(&self, sample: u64) {
        self.sample.set(sample);
    }
    pub(crate) fn sample_after(&self, seconds: f64) -> u64 {
        self.sample
            .get()
            .saturating_add((seconds.max(0.) * self.fs).ceil() as u64)
    }
    pub(crate) fn set_source(&mut self, source: Option<Source>) {
        self.source = source;
    }
    pub(crate) fn source(&self) -> Option<Source> {
        self.source
    }
    pub(crate) fn availability(&self) -> super::runtime::Availability {
        let mut result = super::runtime::Availability::from_decision(self.decision.as_deref());
        result.baseline_end = Some(self.baseline.end_sample);
        result.baseline_coordinate = Some(self.baseline.baseline(self.observer).coordinate_version);
        result
    }
    pub(crate) fn active(&self) -> bool {
        (self.weight != 0. || self.continuation != 0.)
            && self.decision.as_ref().is_some_and(|d| {
                d.prediction.coordinate_version
                    == self.baseline.baseline(self.observer).coordinate_version
                    && self.baseline.end_sample <= self.sample.get()
                    && d.prediction.issued_at <= self.sample.get() as f64 / self.fs
                    && self.sample.get() as f64 / self.fs <= d.expires_at
            })
    }

    #[cfg(test)]
    fn likelihood_into(&self, pitch: Option<f32>, preview: &mut Preview) -> f64 {
        (self.decision.as_ref().unwrap().prediction.unknown
            + self
                .evidence_for(pitch, self.source, preview)
                .values
                .iter()
                .sum::<f64>())
        .clamp(0., 1.)
    }

    fn evidence_for(
        &self,
        pitch: Option<f32>,
        source: Option<Source>,
        preview: &mut Preview,
    ) -> Evidence {
        if !self.active() {
            return Evidence {
                masks: [0; HMAX],
                values: [0.; HMAX],
            };
        }
        let d = self.decision.as_ref().unwrap();
        if let Some(pitch) = pitch {
            let single = [(0, self.single_density)];
            self.baseline.preview(
                self.observer,
                pitch,
                self.template.as_deref().unwrap_or(&single),
                preview,
            );
        }
        let (p, mass, focus, conflict, shape_valid) = if pitch.is_some() {
            (
                &preview.p,
                preview.mass,
                preview.focus,
                preview.conflict,
                preview.shape_valid,
            )
        } else {
            (
                &self.baseline.p,
                self.baseline.mass,
                self.baseline.focus,
                self.baseline.conflict,
                self.baseline.mass > 0.,
            )
        };
        let errors = if pitch.is_some() {
            preview.errors
        } else {
            self.baseline.numerical_errors()
        };
        let acoustic_valid = pitch.is_none() || !preview.partials_outside;
        let mut masks = [0; HMAX];
        let mut values = [0.; HMAX];
        for (index, atom) in d.prediction.atoms[..d.prediction.len].iter().enumerate() {
            let mut mask = 0;
            let mut kernel = Kernel::default();
            for (i, (actual, expected)) in [
                (mass, atom.acoustic.mass),
                (focus, atom.acoustic.focus),
                (conflict, atom.acoustic.competition),
            ]
            .into_iter()
            .enumerate()
            {
                if acoustic_valid && expected.valid {
                    mask |= 1 << i;
                }
                kernel.scalar(
                    Value {
                        value: actual,
                        error: [errors.mass, errors.focus, errors.conflict][i],
                        valid: acoustic_valid,
                        bound: true,
                    },
                    expected,
                    d.variances[i],
                );
            }
            kernel.requested += 1;
            if shape_valid && atom.shape_valid && atom.residual < 1. {
                let total = atom.bins.iter().map(|b| f64::from(b.mass)).sum::<f64>();
                if total > 0. {
                    let mut distance = 0.;
                    let mut cursor = 0;
                    for (index, &current) in p.iter().enumerate() {
                        let expected =
                            if cursor < atom.bins.len() && atom.bins[cursor].index == index {
                                let value = f64::from(atom.bins[cursor].mass) / total;
                                cursor += 1;
                                value
                            } else {
                                0.
                            };
                        distance += (current.sqrt() - expected.sqrt()).powi(2);
                    }
                    mask |= 8;
                    kernel.distance(
                        0.5 * distance,
                        errors.shape.powi(2) + atom.shape_error.powi(2) + d.shape_variance,
                    );
                }
            }
            if let (Some(source), Some(pitch)) = (source, pitch) {
                let expected = atom
                    .trajectories
                    .iter()
                    .find(|t| t.source == source)
                    .map_or(Value::default(), |t| t.pitch);
                if expected.valid {
                    mask |= 16;
                }
                kernel.scalar(Value::exact(f64::from(pitch)), expected, d.variances[5]);
            }
            let available = if kernel.requested > 0 {
                kernel.valid as f64 / kernel.requested as f64
            } else {
                0.
            };
            values[index] = atom.weight * (1. - available + available * kernel.compatibility());
            masks[index] = mask;
        }
        Evidence { masks, values }
    }

    #[cfg(test)]
    pub(crate) fn likelihood_pitch(&self, pitch: f32) -> f64 {
        self.likelihood_into(Some(pitch), &mut self.work.borrow_mut().preview)
    }
    /// Paired supported consequences; the neutral unknown mass never becomes reward.
    fn consequence(
        &self,
        column: usize,
        pitch: Option<f32>,
        source: Option<Source>,
        reference: Option<f32>,
        reference_source: Option<Source>,
        work: &mut Work,
    ) -> f64 {
        let source = if column == 0 { source } else { None };
        let reference_source = if column == 0 { reference_source } else { None };
        if pitch == reference && source == reference_source {
            return 0.;
        }
        let value = self.evidence_for(pitch, source, &mut work.preview);
        let base = if let Some(reference_evidence) = work.references[column]
            && reference_evidence.pitch == reference
            && reference_evidence.source == reference_source
        {
            reference_evidence.evidence
        } else {
            let evidence = self.evidence_for(reference, reference_source, &mut work.preview);
            work.references[column] = Some(Reference {
                pitch: reference,
                source: reference_source,
                evidence,
            });
            evidence
        };
        // Pair each retained empirical reference, without renormalizing survivors.
        let difference = value
            .masks
            .iter()
            .zip(base.masks)
            .enumerate()
            .filter(|(_, (mask, base_mask))| **mask != 0 && **mask == *base_mask)
            .map(|(i, _)| value.values[i] - base.values[i])
            .sum::<f64>();
        (difference / CONSEQUENCE_SCALES[column]).tanh()
    }

    /// Scores are in the consumer's existing utility units, before its selector.
    /// `retain` distinguishes continuation from a conditional destination choice.
    pub(crate) fn choose_scored(
        &self,
        count: usize,
        baseline: usize,
        candidate: impl FnMut(usize) -> (Option<f32>, Option<Source>, f64, bool),
    ) -> usize {
        self.choose_conditioned(count, baseline, 0., 1., candidate)
    }

    /// Add the existing deterministic Flora terms to a common action comparison.
    pub(crate) fn action_difference(&self, pitch: Option<f32>, baseline: f32, retain: bool) -> f64 {
        if !self.active() {
            return 0.;
        }
        let mut work = self.work.borrow_mut();
        work.references = [None; 2];
        if retain {
            self.continuation * self.consequence(1, pitch, None, Some(baseline), None, &mut work)
        } else {
            self.weight
                * self.consequence(
                    0,
                    pitch,
                    self.source,
                    Some(baseline),
                    self.source,
                    &mut work,
                )
        }
    }

    pub(crate) fn no_add_difference(&self, pitch: f32) -> f64 {
        if !self.active() || self.continuation == 0. {
            return 0.;
        }
        let mut work = self.work.borrow_mut();
        work.references = [None; 2];
        self.continuation * self.consequence(1, None, None, Some(pitch), None, &mut work)
    }

    pub(crate) fn choose_conditioned(
        &self,
        count: usize,
        baseline: usize,
        temperature: f64,
        gain: f64,
        mut candidate: impl FnMut(usize) -> (Option<f32>, Option<Source>, f64, bool),
    ) -> usize {
        if !self.active() {
            return baseline;
        }
        let mut work = self.work.borrow_mut();
        work.references = [None; 2];
        work.candidates.clear();
        for i in 0..count {
            work.candidates.push(candidate(i));
        }
        let a0 = work.candidates[baseline];
        work.likelihoods.resize(count, 0.);
        let mut nonzero = false;
        // One representative per action class keeps destination out of continuation.
        let mut representatives = [None, None];
        for (i, a) in work.candidates.iter().enumerate() {
            let slot = &mut representatives[usize::from(a.3)];
            if slot.is_none_or(|j: usize| a.2 > work.candidates[j].2) {
                *slot = Some(i);
            }
        }
        representatives[usize::from(a0.3)] = Some(baseline);
        let mut class_bonus = [0.; 2];
        if self.continuation != 0. {
            for (class, representative) in representatives.iter().enumerate() {
                if class == usize::from(a0.3) {
                    continue;
                }
                if let Some(i) = representative {
                    let a = work.candidates[*i];
                    class_bonus[class] = gain
                        * self.continuation
                        * self.consequence(1, a.0, None, a0.0, None, &mut work);
                }
            }
        }
        let mut selected_class = a0.3;
        for (class, representative) in representatives.iter().enumerate() {
            if let Some(i) = representative {
                let a = work.candidates[*i];
                if class_bonus[class] != 0. {
                    let delta = a.2 + class_bonus[class] - a0.2;
                    let accepted = if temperature > 0. {
                        self.dedicated_draw(0x636c6173) < (delta / temperature).exp().min(1.)
                    } else {
                        delta > 0.
                    };
                    if accepted {
                        selected_class = a.3;
                    }
                }
            }
        }
        for i in 0..count {
            let a = work.candidates[i];
            let relation = if !selected_class && !a.3 && !a0.3 && self.weight != 0. {
                gain * self.weight * self.consequence(0, a.0, a.1, a0.0, a0.1, &mut work)
            } else {
                0.
            };
            nonzero |= relation != 0. || class_bonus[usize::from(a.3)] != 0.;
            work.likelihoods[i] = a.2 + relation + class_bonus[usize::from(a.3)];
        }
        if !nonzero {
            return baseline;
        }
        let start = representatives[usize::from(selected_class)].unwrap_or(baseline);
        if temperature > 0. {
            // A conditional Metropolis step changes destination, never the action class.
            let alternatives = work
                .candidates
                .iter()
                .filter(|a| a.3 == selected_class)
                .count();
            let pick = (self.dedicated_draw(0x64657374) * alternatives as f64) as usize;
            let proposed = work
                .candidates
                .iter()
                .enumerate()
                .filter(|(_, a)| a.3 == selected_class)
                .nth(pick.min(alternatives - 1))
                .unwrap()
                .0;
            let ratio =
                ((work.likelihoods[proposed] - work.likelihoods[start]) / temperature).exp();
            return if self.dedicated_draw(0x61636365) < ratio.min(1.) {
                proposed
            } else {
                start
            };
        }
        let mut best = start;
        for (i, a) in work.candidates.iter().enumerate() {
            if a.3 == selected_class && work.likelihoods[i] > work.likelihoods[best] {
                best = i;
            }
        }
        best
    }

    pub(crate) fn softmax_scores(
        &self,
        weights: &mut [f32],
        baseline: usize,
        temperature: f64,
        mut candidate: impl FnMut(usize) -> (Option<f32>, Option<Source>, f64, bool),
    ) -> bool {
        if !self.active() {
            return false;
        }
        let mut work = self.work.borrow_mut();
        work.references = [None; 2];
        work.candidates.clear();
        for i in 0..weights.len() {
            work.candidates.push(candidate(i));
        }
        let a0 = work.candidates[baseline];
        work.likelihoods.resize(weights.len(), 0.);
        let mut representatives = [None, None];
        for (i, a) in work.candidates.iter().enumerate() {
            let slot = &mut representatives[usize::from(a.3)];
            if slot.is_none_or(|j: usize| a.2 > work.candidates[j].2) {
                *slot = Some(i);
            }
        }
        let mut continuation = [0.; 2];
        if self.continuation != 0. {
            for (class, representative) in representatives.iter().enumerate() {
                if class == usize::from(a0.3) {
                    continue;
                }
                if let Some(i) = representative {
                    let a = work.candidates[*i];
                    continuation[class] =
                        self.continuation * self.consequence(1, a.0, None, a0.0, None, &mut work);
                }
            }
        }
        let mut changed = false;
        let mut max = f64::NEG_INFINITY;
        for i in 0..weights.len() {
            let a = work.candidates[i];
            let relation = if !a.3 && !a0.3 && self.weight != 0. {
                self.weight * self.consequence(0, a.0, a.1, a0.0, a0.1, &mut work)
            } else {
                0.
            };
            let continuation = continuation[usize::from(a.3)];
            changed |= relation != 0. || continuation != 0.;
            let score = if self.continuation == 0. && a.3 != a0.3 {
                f64::NEG_INFINITY
            } else {
                (a.2 + relation + continuation) / temperature
            };
            work.likelihoods[i] = score;
            max = max.max(score);
        }
        if !changed {
            return false;
        }
        let total = weights.iter().map(|&w| f64::from(w)).sum::<f64>();
        let mut sum = 0.;
        for score in &mut work.likelihoods {
            *score = (*score - max).exp();
            sum += *score;
        }
        if sum == 0. || !sum.is_finite() {
            return false;
        }
        for (weight, &q) in weights.iter_mut().zip(&work.likelihoods) {
            *weight = (total * q / sum) as f32;
        }
        true
    }

    /// Spawn densities already encode their native scores as log masses (temperature 1).
    /// Preserve the old total and draw range while adding a supported consequence.
    pub(crate) fn mix(
        &self,
        weights: &mut [f64],
        baseline: usize,
        mut candidate: impl FnMut(usize) -> Option<f32>,
    ) {
        if !self.active() || self.weight == 0. {
            return;
        }
        let reference = candidate(baseline);
        let mut work = self.work.borrow_mut();
        work.references = [None; 2];
        work.likelihoods.resize(weights.len(), 0.);
        let total = weights.iter().sum::<f64>();
        let mut changed = false;
        let mut sum = 0.;
        for (i, &weight) in weights.iter().enumerate() {
            let bonus = self.weight
                * self.consequence(
                    0,
                    candidate(i),
                    self.source,
                    reference,
                    self.source,
                    &mut work,
                );
            changed |= bonus != 0.;
            let value = weight * bonus.exp();
            work.likelihoods[i] = value;
            sum += value;
        }
        if !changed || sum == 0. || !sum.is_finite() {
            return;
        }
        for (weight, &q) in weights.iter_mut().zip(&work.likelihoods) {
            *weight = total * q / sum;
        }
    }

    /// Waiting support belongs only to continuation, after existing eligibility.
    pub(crate) fn allow_now(&self, next_sample: u64) -> bool {
        if !self.active() || self.continuation == 0. || next_sample <= self.sample.get() {
            return true;
        }
        let d = self.decision.as_ref().unwrap();
        let now = self.sample.get() as f64 / self.fs;
        let next = next_sample as f64 / self.fs;
        let mut act = 0.;
        let mut hold = 0.;
        let mut supported = false;
        for atom in &d.prediction.atoms[..d.prediction.len] {
            if !atom.time.valid || atom.time.value + atom.time.error < now {
                continue;
            }
            supported = true;
            let score = |time| {
                let mut kernel = Kernel::default();
                kernel.scalar(Value::exact(time), atom.time, d.variances[3]);
                kernel.compatibility()
            };
            act += atom.weight * score(now);
            hold += atom.weight * score(next);
        }
        if !supported {
            return true;
        }
        let difference = hold - act;
        self.continuation * (difference / CONSEQUENCE_SCALES[1]).tanh() <= 0.
    }

    /// A separate keyed stream for a draw absent from the original selector.
    pub(crate) fn dedicated_draw(&self, tag: u64) -> f64 {
        let source = self.source.unwrap_or_default();
        let mut state = tag
            ^ self.sample.get()
            ^ source.id.rotate_left(17)
            ^ source.generation.rotate_left(33)
            ^ u64::from(source.kind).rotate_left(49)
            ^ self
                .decision
                .as_ref()
                .map_or(0, |d| d.prediction.query_version);
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((state ^ (state >> 31)) >> 11) as f64 * (1. / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
pub(crate) fn test_landscape(weight: f64, known: bool) -> (Landscape, [f32; 2]) {
    let (use_, pitches) = test_use(1., known);
    let mut landscape = Landscape::new(use_.baseline.geometry.space.clone());
    landscape.support = Some(use_.baseline.clone());
    landscape.support_planned = true;
    landscape.harmonic_relations_weight = weight;
    landscape.harmonic_relations = use_.decision;
    landscape.harmonic_now_sample = 512;
    (landscape, pitches)
}

#[cfg(test)]
pub(crate) fn test_use(weight: f64, known: bool) -> (Use, [f32; 2]) {
    use super::{Acoustic, Bin};
    use crate::core::{
        harmonic_support::Geometry, harmonicity_kernel::HarmonicityKernel, log2space::Log2Space,
    };
    let space = Log2Space::new(80., 2000., 48);
    let geometry = Geometry::new(&space, &HarmonicityKernel::new(&space, Default::default()));
    let mut baseline = Snapshot::new(geometry);
    let mut density = vec![0.; space.n_bins()];
    density[50] = 1.;
    baseline.observe_projection(&density, None, 1e-6, 512);
    let baseline = Arc::new(baseline);
    let pitches = [space.centers_log2[20], space.centers_log2[80]];
    let mut prediction = Prediction::new(baseline.p.len(), 1);
    prediction.coordinate_version = baseline.baseline(Observer::Mixed).coordinate_version;
    prediction.issued_at = 512. / 48000.;
    prediction.query_version = 3;
    if known {
        let mut preview = Preview::new(&baseline);
        baseline.preview(Observer::Mixed, pitches[0], &[(0, 1.)], &mut preview);
        prediction.unknown = 0.;
        prediction.len = 1;
        let atom = &mut prediction.atoms[0];
        atom.weight = 1.;
        atom.acoustic = Acoustic {
            coordinate_version: prediction.coordinate_version,
            mass: Value::exact(preview.mass),
            focus: Value::exact(preview.focus),
            competition: Value::exact(preview.conflict),
            ..Default::default()
        };
        atom.shape_valid = true;
        atom.shape_error = preview.errors.shape;
        atom.shape_bound = true;
        atom.bins = preview
            .p
            .iter()
            .enumerate()
            .filter_map(|(index, &p)| {
                (p > 0.).then_some(Bin {
                    index,
                    mass: p as f32,
                })
            })
            .collect();
        atom.time = Value::exact(1.);
    }
    let mut landscape = Landscape::new(space);
    landscape.harmonic_relations_weight = weight;
    landscape.harmonic_relations = Some(Arc::new(Decision::new(prediction, [0.; 9], 0.)));
    let mut slot = None;
    // Create storage even for the disabled-path test, then disable without querying it.
    landscape.harmonic_relations_weight = 1.;
    Use::refresh(
        &mut slot,
        &landscape,
        None,
        Observer::Mixed,
        Some(baseline.clone()),
        None,
        1.,
        48000.,
        512,
    );
    landscape.harmonic_relations_weight = weight;
    Use::refresh(
        &mut slot,
        &landscape,
        None,
        Observer::Mixed,
        Some(baseline),
        None,
        1.,
        48000.,
        512,
    );
    (slot.unwrap(), pitches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_and_zero_weight_keep_existing_candidates_probabilities_and_ties() {
        for weight in [0., 0.5, 1.] {
            let (use_, pitches) = test_use(weight, false);
            let mut weights = [0.125, 0.875];
            use_.mix(&mut weights, 1, |i| Some(pitches[i]));
            assert_eq!(weights, [0.125, 0.875]);
            assert_eq!(
                use_.choose_scored(2, 1, |i| (Some(pitches[i]), use_.source(), 0., false)),
                1
            );
        }
        let (disabled, _) = test_use(0., true);
        assert_eq!(
            disabled.choose_scored(2, 1, |_| panic!("disabled preview")),
            1
        );
        disabled.mix(&mut [0.5, 0.5], 1, |_| panic!("disabled preview"));
    }

    #[test]
    fn causal_history_and_ablations_reach_consumer_choices() {
        use super::super::{
            Accumulator, Acoustic, BOUNDARY, Bin, Event, Limits, Load, Model, OPPORTUNITY,
            Observation, Support,
        };
        let (seed, pitches) = test_use(1., false);
        let baseline = seed.baseline.clone();
        let coordinate_version = baseline.baseline(Observer::Mixed).coordinate_version;
        let make_state = |pitch: Option<f32>| {
            let mut preview = Preview::new(&baseline);
            if let Some(pitch) = pitch {
                baseline.preview(Observer::Mixed, pitch, &[(0, 1.)], &mut preview);
            } else {
                preview.p.clone_from(&baseline.p);
                preview.mass = baseline.mass;
                preview.focus = baseline.focus;
                preview.conflict = baseline.conflict;
                preview.errors = baseline.numerical_errors();
            }
            let value = |v, e| Value {
                value: v,
                error: e,
                valid: true,
                bound: true,
            };
            (
                Acoustic {
                    coordinate_version,
                    mass: value(preview.mass, preview.errors.mass),
                    focus: value(preview.focus, preview.errors.focus),
                    competition: value(preview.conflict, preview.errors.conflict),
                    ..Default::default()
                },
                preview
                    .p
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &p)| {
                        (p > 0.).then_some(Bin {
                            index,
                            mass: p as f32,
                        })
                    })
                    .collect::<Vec<_>>(),
                preview.errors.shape,
            )
        };
        let states = [
            make_state(None),
            make_state(Some(pitches[0])),
            make_state(Some(pitches[1])),
        ];
        let events = [Event {
            source: None,
            flags: OPPORTUNITY,
        }];
        let boundaries = [Event {
            source: None,
            flags: OPPORTUNITY | BOUNDARY,
        }];
        let observation = |index: usize, time: f64, boundary: bool| Observation {
            time,
            timing_error: 0.,
            hop: true,
            acoustic: Acoustic {
                sampled_at: time,
                snapshot: (time * 48000.) as u64,
                ..states[index].0
            },
            support: Support {
                bins: &states[index].1,
                residual: 0.,
                error: states[index].2,
                bound: true,
                valid: true,
            },
            trajectories: &[],
            events: if boundary { &boundaries } else { &events },
        };
        let acquire = |order: &[usize], freeze: bool, remove: bool, episodes: bool| {
            let coordinates = (0..baseline.p.len())
                .map(|i| {
                    f64::from(baseline.geometry.root_start_log2)
                        + i as f64 * f64::from(baseline.geometry.space.step())
                })
                .collect();
            let mut m = Model::with_limits(
                Load {
                    sample_rate: 48000,
                    hop: 512,
                    sources: 0,
                    coordinates,
                    sparse_entries: baseline.p.len(),
                },
                Limits {
                    seconds: 600.,
                    records: 64,
                },
            )
            .unwrap();
            if freeze {
                m.freeze_learning();
            }
            for (i, &index) in order.iter().enumerate() {
                m.observe(observation(
                    index,
                    i as f64,
                    if episodes {
                        i % 2 == 1
                    } else {
                        i + 1 == order.len()
                    },
                ))
                .unwrap();
            }
            // Hold both sensory-context inputs fixed while the graph ordering differs.
            m.local = Accumulator::new(baseline.p.len(), 0.1);
            m.global = Accumulator::new(baseline.p.len(), 1.5);
            m.local.advance(&observation(0, 20., false), 1.);
            m.global.advance(&observation(0, 20., false), 1.);
            if remove {
                m.remove_correspondences();
            }
            let p = m
                .predict(&[observation(0, 20., false)], 71)
                .unwrap()
                .clone();
            let mut use_ = seed.clone();
            use_.decision = Some(Arc::new(Decision::new(
                p.clone(),
                p.variances,
                p.shape_variance,
            )));
            use_.set_sample(20 * 48000);
            let likelihoods = pitches.map(|pitch| use_.likelihood_pitch(pitch));
            let choice = use_.choose_scored(2, 1, |i| (Some(pitches[i]), use_.source(), 0., false));
            m.observe(observation(0, 20., false)).unwrap();
            m.observe(observation(1, 21., false)).unwrap();
            (choice, likelihoods, p.unknown, m.contexts.unknown)
        };
        let a = acquire(&[0, 1, 2], false, false, false);
        let b = acquire(&[0, 2, 1], false, false, false);
        assert_ne!(a.0, b.0);
        assert_ne!(a.1, b.1);
        assert_ne!(a.3, b.3);
        for (freeze, remove) in [(true, false), (false, true)] {
            let a = acquire(&[0, 1, 2], freeze, remove, false);
            let b = acquire(&[0, 2, 1], freeze, remove, false);
            assert_eq!((a.0, a.1, a.2), (b.0, b.1, b.2));
            assert_eq!(a.3, b.3);
            assert_eq!(a.0, 1);
            assert_eq!(a.2, 1.);
        }
        let a = acquire(&[0, 1, 2, 0], false, false, true);
        let b = acquire(&[2, 0, 0, 1], false, false, true);
        assert_eq!(a.0, b.0);
        for (x, y) in a.1.into_iter().zip(b.1) {
            assert!((x - y).abs() <= 16. * f64::EPSILON);
        }
    }

    #[test]
    fn outside_body_support_is_unknown_but_known_source_pitch_remains_usable() {
        use super::super::Trajectory;
        let (mut use_, pitches) = test_use(1., true);
        use_.template = Some(vec![(use_.baseline.p.len() as isize * 10, 1.)].into());
        assert_eq!(use_.likelihood_pitch(pitches[1]), 1.);
        let source = Source {
            kind: 1,
            id: 4,
            generation: 8,
        };
        use_.set_source(Some(source));
        let prediction = &mut Arc::make_mut(use_.decision.as_mut().unwrap()).prediction;
        prediction.atoms[0].trajectories = vec![Trajectory {
            source,
            pitch: Value::exact(f64::from(pitches[0])),
            ..Default::default()
        }];
        assert_eq!(use_.likelihood_pitch(pitches[0]), 1.);
        assert_eq!(use_.likelihood_pitch(pitches[1]), 1. - 1. / 5.);
    }

    #[test]
    fn one_sided_empirical_references_are_neutral_and_partial_pairs_keep_their_mass() {
        use super::super::Trajectory;
        let (mut use_, pitches) = test_use(1., true);
        use_.template = Some(vec![(use_.baseline.p.len() as isize * 10, 1.)].into());
        let a = Source {
            kind: 1,
            id: 1,
            generation: 1,
        };
        let b = Source { id: 2, ..a };
        let trajectory = |source| Trajectory {
            source,
            pitch: Value::exact(f64::from(pitches[0])),
            ..Default::default()
        };
        let prediction = &mut Arc::make_mut(use_.decision.as_mut().unwrap()).prediction;
        let mut atom = prediction.atoms[0].clone();
        atom.weight = 0.25;
        atom.trajectories = vec![trajectory(a)];
        prediction.atoms[0] = atom.clone();
        atom.trajectories = vec![trajectory(b)];
        prediction.atoms[1] = atom.clone();
        prediction.len = 2;
        prediction.unknown = 0.5;
        for (source, reference) in [(a, b), (b, a)] {
            let mut work = use_.work.borrow_mut();
            work.references = [None; 2];
            assert_eq!(
                use_.consequence(
                    0,
                    Some(pitches[1]),
                    Some(source),
                    Some(pitches[0]),
                    Some(reference),
                    &mut work
                ),
                0.
            );
        }
        atom.trajectories = vec![trajectory(a), trajectory(b)];
        let prediction = &mut Arc::make_mut(use_.decision.as_mut().unwrap()).prediction;
        prediction.atoms[2] = atom.clone();
        prediction.len = 3;
        prediction.unknown = 0.25;
        let paired = {
            let mut work = use_.work.borrow_mut();
            work.references = [None; 2];
            use_.consequence(
                0,
                Some(pitches[1]),
                Some(b),
                Some(pitches[0]),
                Some(a),
                &mut work,
            )
        };
        assert!(paired < 0.);
        let prediction = &mut Arc::make_mut(use_.decision.as_mut().unwrap()).prediction;
        prediction.atoms[0] = atom;
        prediction.len = 1;
        prediction.unknown = 0.75;
        let mut work = use_.work.borrow_mut();
        work.references = [None; 2];
        assert_eq!(
            paired,
            use_.consequence(
                0,
                Some(pitches[1]),
                Some(b),
                Some(pitches[0]),
                Some(a),
                &mut work
            )
        );
    }

    #[test]
    fn transported_residual_is_not_counted_twice_after_prediction_publication() {
        let (mut use_, pitches) = test_use(1., true);
        let prediction = &mut Arc::make_mut(use_.decision.as_mut().unwrap()).prediction;
        prediction.unknown = 0.5;
        prediction.atoms[0].weight = 0.5;
        let without_residual = pitches.map(|pitch| use_.likelihood_pitch(pitch));
        Arc::make_mut(use_.decision.as_mut().unwrap())
            .prediction
            .atoms[0]
            .residual = 0.5;
        assert_eq!(
            pitches.map(|pitch| use_.likelihood_pitch(pitch)),
            without_residual
        );
    }

    #[test]
    fn competing_lane_choices_use_each_candidates_generation() {
        use super::super::Trajectory;
        let (mut use_, pitches) = test_use(1., true);
        let sources = [
            Source {
                kind: 1,
                id: 77,
                generation: 1,
            },
            Source {
                kind: 1,
                id: 77,
                generation: 2,
            },
        ];
        let prediction = &mut Arc::make_mut(use_.decision.as_mut().unwrap()).prediction;
        prediction.atoms[0].trajectories = sources
            .into_iter()
            .enumerate()
            .map(|(i, source)| Trajectory {
                source,
                pitch: Value::exact(f64::from(pitches[i])),
                ..Default::default()
            })
            .collect();
        use_.set_source(Some(sources[1]));
        // The same acoustic candidate isolates identity from support compatibility.
        assert_eq!(
            use_.choose_scored(2, 1, |_| (Some(pitches[0]), use_.source(), 0., false)),
            1
        );
        assert_eq!(
            use_.choose_scored(2, 1, |i| (Some(pitches[0]), Some(sources[i]), 0., false)),
            0
        );
        assert_eq!(
            use_.choose_scored(2, 0, |i| (
                Some(pitches[0]),
                Some(sources[1 - i]),
                0.,
                false
            )),
            1
        );
        assert_eq!(use_.source(), Some(sources[1]));
    }

    #[test]
    fn action_compatibility_changes_choices_and_keeps_a_separate_random_stream() {
        let (use_, pitches) = test_use(1., true);
        assert!(use_.likelihood_pitch(pitches[0]) > use_.likelihood_pitch(pitches[1]));
        assert_eq!(
            use_.choose_scored(2, 1, |i| (Some(pitches[i]), use_.source(), 0., false)),
            0
        );
        let mut weights = [0.2, 0.8];
        use_.mix(&mut weights, 1, |i| Some(pitches[i]));
        assert!(weights[0] > 0.2);
        assert!((weights.iter().sum::<f64>() - 1.).abs() <= 2. * f64::EPSILON);
        assert_eq!(use_.dedicated_draw(5), use_.dedicated_draw(5));
        assert_ne!(use_.dedicated_draw(5), use_.dedicated_draw(6));
        assert!(use_.allow_now(48000));
        let mut use_ = use_;
        use_.continuation = 1.;
        assert!(!use_.allow_now(48000));
        assert!(use_.allow_now(512));
    }
    #[test]
    fn signed_columns_preserve_native_costs_and_separate_action_class() {
        let (mut use_, pitches) = test_use(1., true);
        let row = |i: usize| (Some(pitches[i]), None, 0., false);
        assert_eq!(use_.choose_scored(2, 1, row), 0);
        use_.weight = -1.;
        assert_eq!(use_.choose_scored(2, 0, row), 1);
        use_.weight = 1.;
        // A bounded disposition cannot discard a larger native utility gap.
        assert_eq!(
            use_.choose_scored(2, 1, |i| (
                Some(pitches[i]),
                None,
                if i == 1 { 2. } else { 0. },
                false
            )),
            1
        );
        assert_eq!(
            use_.choose_scored(2, 1, |i| (Some(pitches[i]), None, 0., i == 0)),
            1
        );
        assert!(use_.allow_now(48000));
        use_.weight = 0.;
        use_.continuation = 1.;
        assert!(!use_.allow_now(48000));
        use_.continuation = -1.;
        assert!(use_.allow_now(48000));
    }

    #[test]
    fn softmax_relation_preserves_baseline_action_class_and_unknown_scores() {
        let (use_, pitches) = test_use(1., true);
        let mut weights = [0.2, 0.3, 0.5];
        use_.softmax_scores(&mut weights, 1, 0.5, |i| {
            (
                Some(if i == 0 { pitches[0] } else { pitches[1] }),
                None,
                0.,
                i == 2,
            )
        });
        assert_eq!(weights[2], 0.);
        assert!(weights[0] > weights[1]);
        let (unknown, _) = test_use(-1., false);
        let before = weights;
        unknown.softmax_scores(&mut weights, 1, 0.5, |i| {
            (Some(pitches[i % 2]), None, 0., i == 2)
        });
        assert_eq!(weights, before);
    }
}
