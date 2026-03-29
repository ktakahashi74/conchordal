use crate::life::control::{DEFAULT_TIMBRE_SPREAD, DEFAULT_TIMBRE_UNISON, MAX_TIMBRE_UNISON};
use crate::life::sound::modal_engine::ModalMode;

pub(crate) fn modal_tilt_from_brightness(brightness: f32) -> f32 {
    brightness.clamp(0.0, 1.0)
}

pub(crate) fn brightness_from_modal_tilt(modal_tilt: f32) -> f32 {
    modal_tilt.clamp(0.0, 1.0)
}

const MAX_CLUSTER_SPREAD_CENTS: f32 = 24.0;
const CLUSTER_SPREAD_EPS_CENTS: f32 = 1.0e-4;

pub(crate) fn cluster_spread_cents_from_public(spread: f32) -> f32 {
    let spread = if spread.is_finite() {
        spread.clamp(0.0, 1.0)
    } else {
        DEFAULT_TIMBRE_SPREAD
    };
    spread * MAX_CLUSTER_SPREAD_CENTS
}

pub(crate) fn public_spread_from_cluster_spread_cents(spread_cents: f32) -> f32 {
    if !spread_cents.is_finite() || MAX_CLUSTER_SPREAD_CENTS <= 0.0 {
        return DEFAULT_TIMBRE_SPREAD;
    }
    (spread_cents / MAX_CLUSTER_SPREAD_CENTS).clamp(0.0, 1.0)
}

pub(crate) fn sanitize_cluster_unison(unison: usize) -> usize {
    unison.clamp(DEFAULT_TIMBRE_UNISON, MAX_TIMBRE_UNISON)
}

pub(crate) fn active_cluster_unison(spread_cents: f32, unison: usize) -> usize {
    let unison = sanitize_cluster_unison(unison);
    if !spread_cents.is_finite() || spread_cents.abs() <= CLUSTER_SPREAD_EPS_CENTS {
        DEFAULT_TIMBRE_UNISON
    } else {
        unison
    }
}

pub(crate) fn cluster_detune_mul(spread_cents: f32, unison: usize, unison_idx: usize) -> f32 {
    let unison = active_cluster_unison(spread_cents, unison);
    if unison <= 1 {
        return 1.0;
    }
    let center = (unison.saturating_sub(1)) as f32 * 0.5;
    let denom = center.max(0.5);
    let pos = unison_idx as f32 - center;
    let offset_cents = (pos / denom) * spread_cents;
    2.0f32.powf(offset_cents / 1200.0)
}

pub(crate) fn cluster_gain(base_gain: f32, spread_cents: f32, unison: usize) -> f32 {
    base_gain / active_cluster_unison(spread_cents, unison) as f32
}

pub(crate) fn sanitize_mode_ratios(mut ratios: Vec<f32>) -> Vec<f32> {
    ratios.retain(|ratio| ratio.is_finite() && *ratio > 0.0);
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ratios.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-4);
    if ratios.is_empty() { vec![1.0] } else { ratios }
}

pub(crate) fn modal_modes_from_ratios(
    ratios: &[f32],
    modal_tilt: f32,
    cluster_spread_cents: f32,
    cluster_unison: usize,
) -> Vec<ModalMode> {
    let modal_tilt = modal_tilt.clamp(0.0, 1.0);
    let tilt_exp = (1.85 - modal_tilt * 1.45).clamp(0.12, 2.2);
    let active_cluster_unison = active_cluster_unison(cluster_spread_cents, cluster_unison);
    let mut base_modes = Vec::with_capacity(ratios.len().max(1));
    for (idx, ratio) in ratios.iter().copied().enumerate() {
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let k = (idx + 1) as f32;
        let gain = 1.0 / k.powf(tilt_exp);
        let t60_s = ((0.35 + modal_tilt * 1.4) / (1.0 + 0.09 * k)).max(0.03);
        let in_gain = (1.0 / (1.0 + 0.04 * k)).max(0.02);
        base_modes.push(ModalMode {
            ratio,
            t60_s,
            gain,
            in_gain,
        });
    }

    if base_modes.is_empty() {
        base_modes.push(ModalMode {
            ratio: 1.0,
            t60_s: 0.8,
            gain: 1.0,
            in_gain: 1.0,
        });
    }
    normalize_modal_gains(&mut base_modes);
    if active_cluster_unison <= 1 {
        return base_modes;
    }

    let mut clustered = Vec::with_capacity(base_modes.len() * active_cluster_unison);
    for mode in base_modes {
        let gain = cluster_gain(mode.gain, cluster_spread_cents, active_cluster_unison);
        for unison_idx in 0..active_cluster_unison {
            let detune =
                cluster_detune_mul(cluster_spread_cents, active_cluster_unison, unison_idx);
            let ratio = mode.ratio * detune;
            if !ratio.is_finite() || ratio <= 0.0 {
                continue;
            }
            clustered.push(ModalMode {
                ratio,
                t60_s: mode.t60_s,
                gain,
                in_gain: mode.in_gain,
            });
        }
    }
    clustered.sort_by(|a, b| {
        a.ratio
            .partial_cmp(&b.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    clustered
}

fn normalize_modal_gains(modes: &mut [ModalMode]) {
    let peak = modes
        .iter()
        .map(|mode| mode.gain)
        .fold(0.0f32, |acc, gain| acc.max(gain));
    if peak <= 0.0 || !peak.is_finite() {
        return;
    }
    for mode in modes.iter_mut() {
        mode.gain /= peak;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_tilt_public_mapping_is_identity() {
        for value in [0.0, 0.25, 0.6, 1.0] {
            assert!((modal_tilt_from_brightness(value) - value).abs() <= 1e-6);
            assert!((brightness_from_modal_tilt(value) - value).abs() <= 1e-6);
        }
    }

    #[test]
    fn brighter_modal_tilt_boosts_upper_modes() {
        let ratios = [1.0, 2.0, 3.0, 4.0];
        let dark = modal_modes_from_ratios(&ratios, modal_tilt_from_brightness(0.0), 0.0, 1);
        let bright = modal_modes_from_ratios(&ratios, modal_tilt_from_brightness(1.0), 0.0, 1);

        assert!(bright[3].gain > dark[3].gain);
        assert!(bright[3].t60_s > dark[3].t60_s);
    }

    #[test]
    fn clustered_modes_expand_symmetrically() {
        let modes = modal_modes_from_ratios(
            &[1.0],
            modal_tilt_from_brightness(0.5),
            cluster_spread_cents_from_public(0.5),
            3,
        );

        assert_eq!(modes.len(), 3);
        assert!(modes[0].ratio < 1.0);
        assert!((modes[1].ratio - 1.0).abs() <= 1.0e-6);
        assert!(modes[2].ratio > 1.0);
        let total_gain: f32 = modes.iter().map(|mode| mode.gain).sum();
        assert!((total_gain - 1.0).abs() <= 1.0e-5);
    }
}
