use super::*;

#[derive(Default)]
pub(super) struct ClockOnset {
    pub(super) onset_comparison: Option<crate::config::TemporalOnsetComparisonConfig>,
    pub(super) footprint: crate::life::temporal_participation::FootprintTracker,
    pub(super) decision_trace: bool,
}

impl ClockOnset {
    pub(super) fn apply(&self, policy: &mut TemporalParticipation) {
        policy.set_onset_comparison(self.onset_comparison);
        policy.set_body_footprint(self.footprint.selected());
        policy.set_current_identity(self.footprint.identity());
        policy.set_decision_trace(self.decision_trace);
    }
}

impl ParticipationClock {
    pub(super) fn footprint_hold(&self) -> Option<(f32, Option<crate::life::sound::ToneAdsr>)> {
        self.comparison
            .onset_comparison
            .map(|_| (self.hold_theta / self.base_rate_hz, self.adsr))
    }
}

impl PhonationEngine {
    pub(crate) fn set_onset_comparison(
        &mut self,
        config: Option<crate::config::TemporalOnsetComparisonConfig>,
    ) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.comparison.onset_comparison = config;
        }
    }

    pub(crate) fn set_decision_trace(&mut self, enabled: bool) {
        if let PhonationClock::Participation(clock) = &mut self.clock {
            clock.comparison.decision_trace = enabled;
        }
    }

    pub(crate) fn drain_participation_decisions(
        &mut self,
    ) -> impl Iterator<Item = crate::life::temporal_participation::ParticipationDecision> + '_ {
        match &mut self.clock {
            PhonationClock::Participation(clock) => clock.policy.as_mut(),
            _ => None,
        }
        .into_iter()
        .flat_map(|policy| policy.drain_decisions())
    }

    pub(crate) fn footprint_hold(&self) -> Option<(f32, Option<crate::life::sound::ToneAdsr>)> {
        match &self.clock {
            PhonationClock::Participation(clock) => clock.footprint_hold(),
            _ => None,
        }
    }

    /// I11-1 §4.2 hop step: adopt the identity and send at most one request.
    /// Returns true when the supplier had already been asked for this identity.
    pub(crate) fn footprint_hop(
        &mut self,
        identity: crate::life::onset_footprint::Identity,
        now: Tick,
        recipe: &crate::life::onset_footprint::Recipe,
        send: impl FnOnce(crate::life::onset_footprint::Request) -> bool,
    ) -> bool {
        match &mut self.clock {
            PhonationClock::Participation(clock) => {
                clock.comparison.footprint.hop(identity, now, recipe, send)
            }
            _ => false,
        }
    }

    /// Returns true when this Voice was waiting for the dropped reply.
    pub(crate) fn footprint_release(
        &mut self,
        identity: crate::life::onset_footprint::Identity,
    ) -> bool {
        match &mut self.clock {
            PhonationClock::Participation(clock) => clock.comparison.footprint.release(identity),
            _ => false,
        }
    }

    /// Returns true when the record was discarded because its request was superseded.
    pub(crate) fn footprint_receive(
        &mut self,
        record: &crate::life::onset_footprint::Record,
    ) -> bool {
        match &mut self.clock {
            PhonationClock::Participation(clock) => clock.comparison.footprint.receive(record),
            _ => false,
        }
    }
}
