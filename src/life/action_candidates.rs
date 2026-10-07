//! Frozen bodily inputs, not acoustic predictions or permission to execute them.
//! Public for the offline temporal_action_profiles example; live policy remains local.

use crate::core::timebase::Tick;

pub(crate) mod energy;
pub(crate) mod live;

/// Facts passed to the phonation engine, not permission for a future candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct PolicySnapshot {
    pub at: Tick,
    pub is_alive: bool,
    pub gate_allows_onset: bool,
}

/// Frozen clock state, not a reservation or future permission to emit sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(serde::Deserialize))]
pub enum OpportunityBasis {
    ParticipationDue,
    ParticipationPlanned,
    CouplingProjection,
    ThetaProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct Opportunity {
    pub issued_at: Tick,
    pub at: Tick,
    pub basis: OpportunityBasis,
}

/// A granted clock candidate, captured before onset emission.
/// Participation's intrinsic due time is read before resolving the selected plan.
/// Attached to its one materialized ToneSpec; later reports are receipts, not pending actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct OnsetOpportunity {
    pub issued_at: Tick,
    pub at: Tick,
    pub gate: u64,
    pub intrinsic_due_at: Option<Tick>,
    pub intrinsic_period_ticks: Option<Tick>,
    pub planned_release_at: Option<Tick>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    OnsetNow,
    DelayedOnset,
    Wait,
    Skip,
    Continue,
    Release,
    Gap,
}

#[derive(Clone, Copy)]
pub struct BodyState {
    /// The caller evaluates the existing class-specific body/rate constraints.
    /// Unknown permission is not sufficient to construct an input.
    pub permits_action: Option<bool>,
    pub active_at_candidate: Option<bool>,
    pub pending_opportunity: bool,
    pub due_unconsumed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Input {
    pub class: Class,
    pub issued_at: Tick,
    pub at: Tick,
    pub excitation_at: Option<Tick>,
    pub release_at: Option<Tick>,
    pub reconsider_at: Option<Tick>,
    pub consumes_due_opportunity: bool,
    pub withhold_until: Option<Tick>,
}

impl Class {
    pub fn input(
        self,
        issued_at: Tick,
        at: Tick,
        period: Option<Tick>,
        sample_rate: u32,
        body: BodyState,
    ) -> Option<Input> {
        if sample_rate == 0 || at < issued_at || body.permits_action != Some(true) {
            return None;
        }
        let mut input = Input {
            class: self,
            issued_at,
            at,
            excitation_at: None,
            release_at: None,
            reconsider_at: None,
            consumes_due_opportunity: false,
            withhold_until: None,
        };
        match self {
            Self::OnsetNow if at == issued_at => input.excitation_at = Some(at),
            Self::DelayedOnset if at > issued_at => input.excitation_at = Some(at),
            Self::Wait
                if body.pending_opportunity
                    && at - issued_at >= u64::from(sample_rate).div_ceil(20) =>
            {
                input.reconsider_at = Some(at);
            }
            Self::Skip if at == issued_at && body.pending_opportunity && body.due_unconsumed => {
                input.consumes_due_opportunity = true;
            }
            Self::Continue if at == issued_at && body.active_at_candidate == Some(true) => {}
            Self::Release if body.active_at_candidate == Some(true) => input.release_at = Some(at),
            Self::Gap => {
                input.release_at = body.active_at_candidate?.then_some(at);
                input.withhold_until = Some(at.checked_add(period.filter(|p| *p > 0)?)?);
            }
            _ => return None,
        }
        Some(input)
    }
}

/// Uniform offsets use integer floor; keep the unchanged default even outside the span.
/// Extra points require caller-established support and calibration; no CDF is invented here.
pub(crate) fn times(
    issued_at: Tick,
    span: Option<Tick>,
    body_default: Tick,
    additional: [Option<Tick>; 3],
) -> Option<([Tick; 16], usize)> {
    if body_default < issued_at {
        return None;
    }
    let mut times = [body_default; 16];
    let mut count = 1;
    if let Some(span) = span {
        for k in 0..12 {
            let Some(at) = issued_at.checked_add((u128::from(span) * k / 11) as u64) else {
                continue;
            };
            if !times[..count].contains(&at) {
                times[count] = at;
                count += 1;
            }
        }
        for at in additional.into_iter().flatten() {
            if at >= issued_at && at - issued_at <= span && !times[..count].contains(&at) {
                times[count] = at;
                count += 1;
            }
        }
    }
    times[..count].sort_unstable();
    Some((times, count))
}

#[cfg(test)]
mod tests;
