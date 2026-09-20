use super::{Community, PhonationBatch};
use crate::core::timebase::Tick;
use crate::life::social_density::SocialDensityTrace;
use crate::life::voice::AnyArticulationCore;

pub(super) fn kuramoto_order_from_phases(phases: &[f32]) -> Option<f32> {
    let mut sum_cos = 0.0f32;
    let mut sum_sin = 0.0f32;
    let mut count = 0u32;
    for &phase in phases {
        if !phase.is_finite() {
            continue;
        }
        sum_cos += phase.cos();
        sum_sin += phase.sin();
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let n = count as f32;
    let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n;
    Some(r.clamp(0.0, 1.0))
}

pub(super) fn mix_pred_gate_gain(sync: f32, gain_raw: f32) -> f32 {
    let sync = sync.clamp(0.0, 1.0);
    let gain01 = 0.2 + 0.8 * gain_raw.powf(2.0);
    let gain = 1.0 + (gain01 - 1.0) * sync;
    if gain.is_finite() { gain.max(0.0) } else { 1.0 }
}

pub(super) fn build_social_trace_from_batches(
    phonation_batches: &[PhonationBatch],
    frame_end: Tick,
    hop_tick: Tick,
    bin_ticks: u32,
    smooth: f32,
    population_size: usize,
) -> SocialDensityTrace {
    let mut onset_ticks = Vec::new();
    for batch in phonation_batches {
        for onset in &batch.onsets {
            onset_ticks.push((onset.onset_tick.saturating_add(hop_tick), onset.strength));
        }
    }
    SocialDensityTrace::from_onsets(
        frame_end,
        frame_end.saturating_add(hop_tick),
        bin_ticks,
        smooth,
        population_size,
        &onset_ticks,
    )
}

pub(super) fn social_trace_params(hop_tick: Tick) -> (u32, f32) {
    let auto_bin = (hop_tick / 64).max(1);
    let bin_ticks = auto_bin.min(u32::MAX as Tick) as u32;
    (bin_ticks, 0.0)
}

pub(super) fn social_trace_enabled_from_couplings<I>(couplings: I) -> bool
where
    I: IntoIterator<Item = f32>,
{
    couplings.into_iter().any(|coupling| coupling != 0.0)
}

impl Community {
    /// Offset-removed entrainment phases of live voices, one per `Entrain` core.
    /// Used for the Kuramoto order parameter and the GUI phase circle.
    pub fn entrain_aligned_phases(&self) -> Vec<f32> {
        let mut phases = Vec::with_capacity(self.voices.len());
        for voice in &self.voices {
            if !voice.is_alive() {
                continue;
            }
            let AnyArticulationCore::Entrain(core) = &voice.articulation.core else {
                continue;
            };
            let aligned_phase =
                (core.rhythm_phase - core.phase_offset).rem_euclid(std::f32::consts::TAU);
            if aligned_phase.is_finite() {
                phases.push(aligned_phase);
            }
        }
        phases
    }

    pub fn kuramoto_order_parameter(&self) -> Option<(f32, usize)> {
        let phases = self.entrain_aligned_phases();
        let r = kuramoto_order_from_phases(&phases)?;
        Some((r, phases.len()))
    }

    /// Entrainment phases plus their Kuramoto order in a single voice scan, for
    /// the UI frame (avoids scanning + allocating twice per frame).
    pub fn entrain_phases_and_order(&self) -> (Vec<f32>, Option<f32>) {
        let phases = self.entrain_aligned_phases();
        let r = kuramoto_order_from_phases(&phases);
        (phases, r)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{spawn_spec_with_freq, test_pop};
    use super::*;
    use crate::core::landscape::LandscapeFrame;
    use crate::life::phonation_engine::OnsetEvent;
    use crate::scenario::Action;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn pred_gate_gain_sync_zero_is_unity() {
        let gain = mix_pred_gate_gain(0.0, 0.3);
        assert_eq!(gain, 1.0);
    }

    #[test]
    fn social_trace_is_delayed_by_one_hop() {
        let batch = PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: None,
            source_id: 1,
            source_generation: 0,
            routing: crate::scenario::control::Routing::default(),
            cmds: Vec::new(),
            tones: Vec::new(),
            onsets: vec![OnsetEvent {
                gate: 0,
                onset_tick: 90,
                strength: 1.0,
            }],
        };
        let trace = build_social_trace_from_batches(&[batch], 100, 10, 5, 0.0, 1);
        assert_eq!(trace.start_tick, 100);
        assert_eq!(trace.density_at(95), 0.0);
        assert_eq!(trace.density_at(100), 1.0);
    }

    #[test]
    fn social_trace_enabled_with_nonzero_coupling() {
        let couplings = vec![0.0, 1.0];
        assert!(social_trace_enabled_from_couplings(couplings));
    }

    #[test]
    fn kuramoto_order_parameter_is_bounded() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let phases: Vec<f32> = (0..256)
            .map(|_| rng.random_range(0.0..std::f32::consts::TAU))
            .collect();
        let r = kuramoto_order_from_phases(&phases).expect("non-empty");
        assert!((0.0..=1.0).contains(&r));
    }

    #[test]
    fn kuramoto_order_parameter_high_for_aligned_low_for_random() {
        let aligned = vec![0.0f32; 128];
        let aligned_r = kuramoto_order_from_phases(&aligned).expect("non-empty");
        assert!(aligned_r > 0.99, "aligned phase set should have high order");

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let random: Vec<f32> = (0..128)
            .map(|_| rng.random_range(0.0..std::f32::consts::TAU))
            .collect();
        let random_r = kuramoto_order_from_phases(&random).expect("non-empty");
        assert!(
            random_r < 0.35,
            "random phase set should have low order (got {random_r})"
        );
    }

    #[test]
    fn kuramoto_order_parameter_uses_relative_phase_offset() {
        let mut pop = test_pop();
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                population_id: 1,
                ids: vec![1, 2, 3],
                spec: spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        let shared_phase = 0.75;
        for (idx, voice) in pop.voices.iter_mut().enumerate() {
            let AnyArticulationCore::Entrain(core) = &mut voice.articulation.core else {
                panic!("expected entrain core");
            };
            core.phase_offset = idx as f32 * 2.0;
            core.rhythm_phase = shared_phase + core.phase_offset;
        }

        let (order, count) = pop.kuramoto_order_parameter().expect("order");

        assert_eq!(count, 3);
        assert!(order > 0.99);
    }
}
