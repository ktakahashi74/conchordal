use super::*;

impl PitchHillClimbPitchCore {
    pub(in super::super) fn set_reference_loo_mode(&mut self, mode: ReferenceLooMode) {
        self.reference_loo_mode = mode;
    }
}

pub(crate) fn aud_m1_hill_choice(
    landscape: &Landscape,
    anchor: f32,
    neighbors: &[f32],
) -> (f32, f32) {
    use rand::SeedableRng;
    let mut core = PitchHillClimbPitchCore::new(landscape.space.step() * 1200.0, anchor, 0.0);
    core.set_global_peaks(8, 20.0);
    core.set_ratio_candidates(true, 20);
    core.move_cost_coeff = 0.0;
    let choice = core
        .propose_with_scorer(
            anchor,
            anchor,
            landscape,
            neighbors,
            &mut rand::rngs::SmallRng::seed_from_u64(719),
            |p| landscape.evaluate_pitch_score_log2(p),
        )
        .target_pitch_log2;
    let mut ranked = core.scratch.borrow().scored.clone();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let margin = if ranked.len() > 1 {
        ranked[0].1 - ranked[1].1
    } else {
        f32::INFINITY
    };
    (choice, margin)
}
