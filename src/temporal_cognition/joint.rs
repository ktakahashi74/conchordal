//! Enumerated conditional-state normalization, before identity-losing beam pruning.

mod acoustics;
pub(super) mod correspondence;
mod grouping;
mod owner;
mod phrase_inputs;
mod producer;
pub(super) mod proposals;
pub(super) mod section;
mod shared;

#[derive(Clone, Copy, Debug)]
pub(super) struct Pair {
    pub parent: u8,
    pub extension: u8,
    pub resolved: bool,
    pub log_prior: f64,
    pub log_transition: f64,
    pub log_potential: f64,
}

pub(super) struct Shared<'a> {
    pub pair: Pair,
    pub groups: &'a [&'a [Pair]],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Weight {
    pub parent: u8,
    pub extension: u8,
    pub mass: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Local {
    pub rows: [Option<Weight>; 15],
    pub explicit_unknown: f64,
    pub pruned_mass: f64,
    pub pruned_count: usize,
    pub enumerated: usize,
    // log Z = shift[0] + shift[1]; retain both to preserve small log-prior differences.
    pub partition: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Context {
    pub weight: Weight,
    pub groups: [Local; 8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Posterior {
    pub contexts: [Option<Context>; 7],
    pub explicit_unknown: f64,
    pub pruned_mass: f64,
    pub pruned_count: usize,
    pub shared_enumerated: usize,
    pub local_enumerated: usize,
    pub group_count: usize,
    pub log_evidence: Option<f64>,
}

pub(super) struct Normalizer {
    groups: Box<[[Local; 8]; 32]>,
    weights: [f64; 256],
    order: [usize; 256],
    output: Posterior,
}

pub(super) struct Candidate {
    pub shared_pair: (u8, u8),
    pub local_pair: (u8, u8),
    pub delta_log_score: Option<f64>,
    pub heads: [Option<super::ratings::WeightedRating>; 3],
}

impl Posterior {
    pub(super) fn preview(
        &self,
        group: usize,
        candidates: &[Candidate],
        coverage: [f64; 3],
        priors: [[f64; 5]; 3],
        temperatures: [f64; 3],
    ) -> Option<super::consequence::Projection> {
        use super::consequence::Alternative;
        if group >= self.group_count || candidates.len() > 7 * 15 {
            return None;
        }
        let mut alternatives = [Alternative {
            weight: 0.,
            delta_log_score: None,
            heads: [None; 3],
        }; 7 * 15];
        for (c, context) in self
            .contexts
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
        {
            for (l, local) in context.groups[group]
                .rows
                .iter()
                .enumerate()
                .filter_map(|(i, l)| l.map(|l| (i, l)))
            {
                alternatives[c * 15 + l].weight = context.weight.mass * local.mass;
            }
        }
        let mut seen = [false; 7 * 15];
        for input in candidates {
            let (c, context) = self
                .contexts
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|c| (i, c)))
                .find(|(_, c)| (c.weight.parent, c.weight.extension) == input.shared_pair)?;
            let (l, _) = context.groups[group]
                .rows
                .iter()
                .enumerate()
                .filter_map(|(i, l)| l.map(|l| (i, l)))
                .find(|(_, l)| (l.parent, l.extension) == input.local_pair)?;
            let index = c * 15 + l;
            if seen[index] {
                return None;
            }
            seen[index] = true;
            alternatives[index].delta_log_score = input.delta_log_score;
            alternatives[index].heads = input.heads;
        }
        super::consequence::project(&alternatives, coverage, priors, temperatures)
    }
}

fn validate(
    pairs: impl Iterator<Item = Pair>,
    parents: usize,
    children: usize,
) -> Result<(), &'static str> {
    let mut priors = [None; 16];
    let mut transitions = [0.; 16];
    let mut unknowns = [0; 16];
    let mut seen = [0_u64; 4];
    let mut count = 0;
    for pair in pairs {
        let p = usize::from(pair.parent);
        let c = usize::from(pair.extension);
        if p >= parents
            || c >= children
            || pair.log_prior.is_nan()
            || pair.log_prior > 0.
            || pair.log_transition.is_nan()
            || pair.log_transition > 0.
            || !pair.log_potential.is_finite()
        {
            return Err("invalid joint proposal slot or log score");
        }
        let key = p * children + c;
        if seen[key / 64] & (1 << (key % 64)) != 0 {
            return Err("duplicate joint parent/extension");
        }
        seen[key / 64] |= 1 << (key % 64);
        if priors[p].is_some_and(|q| q != pair.log_prior) {
            return Err("joint parent prior changed between extensions");
        }
        priors[p] = Some(pair.log_prior);
        transitions[p] += pair.log_transition.exp();
        unknowns[p] += usize::from(!pair.resolved);
        count += 1;
    }
    if count == 0 || (priors.iter().flatten().map(|q| q.exp()).sum::<f64>() - 1.).abs() > 1e-12 {
        return Err("joint parent priors must sum to one");
    }
    for p in 0..parents {
        if priors[p].is_some() && ((transitions[p] - 1.).abs() > 1e-12 || unknowns[p] != 1) {
            return Err("joint parent needs normalized transitions and one unknown extension");
        }
    }
    Ok(())
}

impl Normalizer {
    pub(super) fn new() -> Self {
        Self {
            groups: Box::new([[Local::default(); 8]; 32]),
            weights: [0.; 256],
            order: [0; 256],
            output: Posterior::default(),
        }
    }

    pub(super) fn normalize(
        &mut self,
        shared: &[Shared<'_>],
        observed: bool,
    ) -> Result<&Posterior, &'static str> {
        if shared.is_empty() || shared.len() > 32 {
            return Err("joint shared pair capacity");
        }
        let group_count = shared[0].groups.len();
        // With no live groups, the empty local product leaves only shared transitions and evidence.
        if group_count > 8 || shared.iter().any(|c| c.groups.len() != group_count) {
            return Err("joint group inventory mismatch");
        }
        validate(shared.iter().map(|s| s.pair), 8, 4)?;
        let mut local_enumerated = 0;
        for (context, input) in shared.iter().enumerate() {
            self.groups[context][group_count..].fill(Local::default());
            for (group, pairs) in input.groups.iter().enumerate() {
                if pairs.len() > 256 {
                    return Err("joint local pair capacity");
                }
                validate(pairs.iter().copied(), 16, 16)?;
                local_enumerated += pairs.len();
                let shift = pairs
                    .iter()
                    .filter(|p| p.log_prior.is_finite() && p.log_transition.is_finite())
                    .map(|p| if observed { p.log_potential } else { 0. })
                    .fold(f64::NEG_INFINITY, f64::max);
                let scores = &mut self.weights[..pairs.len()];
                for (score, pair) in scores.iter_mut().zip(*pairs) {
                    if !pair.log_prior.is_finite() || !pair.log_transition.is_finite() {
                        *score = f64::NEG_INFINITY;
                        continue;
                    }
                    *score = pair.log_prior
                        + pair.log_transition
                        + (if observed { pair.log_potential } else { 0. } - shift);
                }
                let pivot = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if !pivot.is_finite() {
                    return Err("unrepresentable joint local relative scores");
                }
                let total = scores.iter().map(|v| (v - pivot).exp()).sum::<f64>();
                let log_total = total.ln();
                let local = &mut self.groups[context][group];
                *local = Local {
                    enumerated: pairs.len(),
                    partition: [shift, pivot + log_total],
                    ..Local::default()
                };
                for (index, (score, pair)) in scores.iter_mut().zip(*pairs).enumerate() {
                    *score = ((*score - pivot) - log_total).exp();
                    self.order[index] = index;
                    if !pair.resolved {
                        local.explicit_unknown += *score;
                    }
                }
                self.order[..pairs.len()].sort_unstable_by(|&a, &b| {
                    scores[b].total_cmp(&scores[a]).then(
                        (pairs[a].parent, pairs[a].extension)
                            .cmp(&(pairs[b].parent, pairs[b].extension)),
                    )
                });
                let mut kept = 0;
                for &index in &self.order[..pairs.len()] {
                    let pair = pairs[index];
                    if !pair.resolved || scores[index] == 0. {
                        continue;
                    }
                    if kept < 15 {
                        local.rows[kept] = Some(Weight {
                            parent: pair.parent,
                            extension: pair.extension,
                            mass: scores[index],
                        });
                        kept += 1;
                    } else {
                        local.pruned_mass += scores[index];
                        local.pruned_count += 1;
                    }
                }
            }
        }
        let shared_shift = shared
            .iter()
            .filter(|s| s.pair.log_prior.is_finite() && s.pair.log_transition.is_finite())
            .map(|s| if observed { s.pair.log_potential } else { 0. })
            .fold(f64::NEG_INFINITY, f64::max);
        let group_shifts: [f64; 8] = std::array::from_fn(|g| {
            if g >= group_count {
                return 0.;
            }
            shared
                .iter()
                .enumerate()
                .filter(|(_, c)| c.pair.log_prior.is_finite() && c.pair.log_transition.is_finite())
                .map(|(i, _)| self.groups[i][g].partition[0])
                .fold(f64::NEG_INFINITY, f64::max)
        });
        for (index, input) in shared.iter().enumerate() {
            let pair = input.pair;
            self.order[index] = index;
            if !pair.log_prior.is_finite() || !pair.log_transition.is_finite() {
                self.weights[index] = f64::NEG_INFINITY;
                continue;
            }
            let mut score = pair.log_prior
                + pair.log_transition
                + (if observed { pair.log_potential } else { 0. } - shared_shift);
            for (local, &shift) in self.groups[index]
                .iter()
                .zip(&group_shifts)
                .take(group_count)
            {
                score += (local.partition[0] - shift) + local.partition[1];
            }
            self.weights[index] = score;
        }
        let scores = &mut self.weights[..shared.len()];
        let pivot = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !pivot.is_finite() {
            return Err("unrepresentable joint shared relative scores");
        }
        let log_total = scores.iter().map(|v| (v - pivot).exp()).sum::<f64>().ln();
        let log_evidence =
            shared_shift + group_shifts[..group_count].iter().sum::<f64>() + pivot + log_total;
        let mut output = Posterior {
            shared_enumerated: shared.len(),
            local_enumerated,
            group_count,
            log_evidence: log_evidence.is_finite().then_some(log_evidence),
            ..Posterior::default()
        };
        for (weight, input) in scores.iter_mut().zip(shared) {
            *weight = ((*weight - pivot) - log_total).exp();
            if !input.pair.resolved {
                output.explicit_unknown += *weight;
            }
        }
        self.order[..shared.len()].sort_unstable_by(|&a, &b| {
            scores[b].total_cmp(&scores[a]).then(
                (shared[a].pair.parent, shared[a].pair.extension)
                    .cmp(&(shared[b].pair.parent, shared[b].pair.extension)),
            )
        });
        let mut kept = 0;
        for &index in &self.order[..shared.len()] {
            let pair = shared[index].pair;
            if !pair.resolved || scores[index] == 0. {
                continue;
            }
            if kept < 7 {
                output.contexts[kept] = Some(Context {
                    weight: Weight {
                        parent: pair.parent,
                        extension: pair.extension,
                        mass: scores[index],
                    },
                    groups: self.groups[index],
                });
                kept += 1;
            } else {
                output.pruned_mass += scores[index];
                output.pruned_count += 1;
            }
        }
        self.output = output;
        Ok(&self.output)
    }
}

#[cfg(test)]
mod tests;
