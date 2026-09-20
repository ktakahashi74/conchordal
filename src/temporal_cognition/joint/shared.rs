//! Shared transition admission, independent of observation log-potentials.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Known {
    // This identifies a complete retained path, not its current context label.
    pub path: u64,
    pub raw_score: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Choice {
    pub path: Option<u64>,
    pub log_transition: f64,
}

/// Slots are stay, unknown, retrieved, new/contrasting; empty slots are not refilled.
pub(super) fn build(
    parent: Option<u64>,
    stay: Option<Known>,
    retrieved: Option<Known>,
    novel: Option<Known>,
    dt: f64,
    observed: bool,
) -> Result<[Option<Choice>; 4], &'static str> {
    if !dt.is_finite() || dt < 0. {
        return Err("invalid shared proposal clock");
    }
    let inputs = [stay, None, retrieved, novel];
    if inputs
        .iter()
        .flatten()
        .any(|p| !p.raw_score.is_finite() || p.raw_score < 0.)
        || stay.is_some_and(|p| Some(p.path) != parent)
    {
        return Err("invalid shared proposal score or stay identity");
    }
    let mut admitted = [None; 4];
    for (slot, input) in inputs.into_iter().enumerate() {
        let Some(input) = input else { continue };
        // Missing input retains known-parent priors but cannot recover an unknown parent.
        if input.raw_score == 0. || (!observed && parent.is_none()) {
            continue;
        }
        // A repeated identity keeps the first role's score, never a second vote.
        if admitted[..slot]
            .iter()
            .flatten()
            .any(|p: &Known| p.path == input.path)
        {
            continue;
        }
        admitted[slot] = Some(input);
    }
    let maximum = admitted
        .iter()
        .flatten()
        .map(|p| p.raw_score)
        .fold(0., f64::max);
    let total: f64 = admitted
        .iter()
        .flatten()
        .map(|p| p.raw_score / maximum)
        .sum();
    let mut output = [None; 4];
    for (slot, p) in admitted
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.map(|p| (i, p)))
    {
        output[slot] = Some(Choice {
            path: Some(p.path),
            log_transition: p.raw_score.ln() - maximum.ln() - total.ln() - dt / 120.,
        });
    }
    output[1] = Some(Choice {
        path: None,
        log_transition: if maximum == 0. {
            0.
        } else {
            (-(-dt / 120.).exp_m1()).ln()
        },
    });
    Ok(output)
}

#[cfg(test)]
mod tests;

mod state;
pub(super) use state::{Group, Section, State, Support};

pub(super) mod producer;
