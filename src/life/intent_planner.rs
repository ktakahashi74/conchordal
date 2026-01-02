use crate::core::timebase::Tick;
use crate::life::intent::Intent;

pub fn choose_onset_by_density(candidates: &[Tick], intents: &[Intent], eps: Tick) -> Option<Tick> {
    if candidates.is_empty() {
        return None;
    }
    let mut best_tick = candidates[0];
    let mut best_score = 0usize;
    for &cand in candidates {
        let min = cand.saturating_sub(eps);
        let max = cand.saturating_add(eps);
        let mut score = 0usize;
        for intent in intents {
            if intent.onset >= min && intent.onset <= max {
                score += 1;
            }
        }
        if score > best_score || (score == best_score && cand < best_tick) {
            best_score = score;
            best_tick = cand;
        }
    }
    Some(best_tick)
}
