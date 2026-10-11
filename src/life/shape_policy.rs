use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    Base,
    Connect,
}

#[derive(Debug)]
pub(crate) struct ShapePolicy {
    pub(crate) enabled: bool,
    pub(crate) state: Shape,
    pub(crate) veto_plan: bool,
    pub(crate) has_tone: bool,
    p: f32,
    fs: f32,
    rng: SmallRng,
    clears: Vec<u64>,
    clear_index: usize,
    #[cfg(test)]
    pub(crate) observations: Option<Vec<(u64, Shape)>>,
}

impl ShapePolicy {
    pub(crate) fn new(p: f32, seed: u64, fs: f32) -> Self {
        assert!(p > 0.0 && p <= 1.0);
        Self {
            enabled: true,
            state: Shape::Base,
            veto_plan: false,
            has_tone: false,
            p,
            fs,
            // Domain separation is reproducibility bookkeeping, not a musical weight.
            rng: SmallRng::seed_from_u64(seed ^ 0xA800_504F_4C49_4359),
            clears: Vec::new(),
            clear_index: 0,
            #[cfg(test)]
            observations: None,
        }
    }

    pub(crate) fn reserve(&mut self, control_capacity: usize, hop: usize) {
        let onset_capacity = (hop as f64 * f64::from(super::phonation_engine::MAX_ONSET_RATE_HZ)
            / f64::from(self.fs))
        .ceil() as usize;
        self.clears.reserve(control_capacity + onset_capacity + 4);
    }

    pub(crate) fn propose(&mut self, proposal: Shape, at: u64, locked: bool) {
        let before = self.state;
        let trial = !locked && proposal != before;
        if trial && self.rng.random::<f32>() >= self.p {
            self.state = proposal;
        }
        if before != self.state && self.state == Shape::Connect {
            // Release allows growth if the reserved budget is exceeded.
            debug_assert!(self.clears.len() < self.clears.capacity());
            self.clears.push(at);
        }
        #[cfg(test)]
        if before != self.state
            && let Some(observations) = self.observations.as_mut()
        {
            observations.push((at, self.state));
        }
    }

    pub(crate) fn clear_before(&mut self, at: u64) -> bool {
        let mut clear = false;
        while self.clears.get(self.clear_index).is_some_and(|t| *t <= at) {
            self.clear_index += 1;
            clear = true;
        }
        clear
    }

    pub(crate) fn finish_segment(&mut self) {
        // Release discards undrained clears and starts the next segment empty.
        debug_assert_eq!(self.clear_index, self.clears.len());
        self.clears.clear();
        self.clear_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn articulation_persistence_policy_same_proposal_and_locked_wait_do_not_consume_rng() {
        let mut a = ShapePolicy::new(0.5, 21, 48000.0);
        let mut b = ShapePolicy::new(0.5, 21, 48000.0);
        a.reserve(32, 512);
        b.reserve(32, 512);
        for at in 0..32 {
            a.propose(Shape::Base, at, false);
            a.propose(Shape::Connect, at, true);
        }
        assert_eq!(a.rng.random::<u64>(), b.rng.random::<u64>());
    }

    #[test]
    fn articulation_persistence_policy_counterpart_proposals_switch_and_clear_in_sample_order() {
        let mut p = ShapePolicy::new(0.5, 21, 48000.0);
        p.reserve(32, 512);
        let mut changes = 0;
        for at in 0..32 {
            let before = p.state;
            let proposal = if before == Shape::Base {
                Shape::Connect
            } else {
                Shape::Base
            };
            p.propose(proposal, at, false);
            if before != p.state {
                changes += 1;
            }
            assert_eq!(
                p.clear_before(at),
                before != p.state && p.state == Shape::Connect
            );
            p.finish_segment();
        }
        assert!(changes > 0 && changes < 32);
    }

    #[test]
    fn articulation_persistence_policy_same_event_stream_is_independent_of_hop_partition() {
        let mut a = ShapePolicy::new(0.75, 21, 48000.0);
        let mut b = ShapePolicy::new(0.75, 21, 48000.0);
        a.reserve(64, 512);
        b.reserve(64, 256);
        for at in 0..64 {
            let r = if at % 3 == 0 {
                Shape::Base
            } else {
                Shape::Connect
            };
            a.propose(r, at, at % 7 == 0);
            b.propose(r, at, at % 7 == 0);
            assert_eq!(a.state, b.state);
            assert_eq!(a.clear_before(at), b.clear_before(at));
            a.finish_segment();
            if at % 2 == 1 {
                b.finish_segment();
            }
        }
        b.finish_segment();
        assert_eq!(a.rng.random::<u64>(), b.rng.random::<u64>());
    }

    #[test]
    fn articulation_persistence_policy_after_birth_has_no_allocations() {
        let mut p = ShapePolicy::new(0.875, 21, 48000.0);
        p.reserve(64, 512);
        crate::runtime_profile::begin_allocations();
        for at in 0..64 {
            p.propose(
                if at % 2 == 0 {
                    Shape::Connect
                } else {
                    Shape::Base
                },
                at,
                false,
            );
            p.clear_before(at);
            p.finish_segment();
        }
        if let Some(counts) = crate::runtime_profile::finish_allocations() {
            assert_eq!(counts.count, 0);
            assert_eq!(counts.bytes, 0);
        }
    }
}
