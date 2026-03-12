use crate::life::sound::modal_engine::ModalMode;

pub(crate) fn modal_tilt_from_brightness(brightness: f32) -> f32 {
    brightness.clamp(0.0, 1.0)
}

pub(crate) fn brightness_from_modal_tilt(modal_tilt: f32) -> f32 {
    modal_tilt.clamp(0.0, 1.0)
}

pub(crate) fn sanitize_mode_ratios(mut ratios: Vec<f32>) -> Vec<f32> {
    ratios.retain(|ratio| ratio.is_finite() && *ratio > 0.0);
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ratios.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-4);
    if ratios.is_empty() { vec![1.0] } else { ratios }
}

pub(crate) fn modal_modes_from_ratios(ratios: &[f32], modal_tilt: f32) -> Vec<ModalMode> {
    let modal_tilt = modal_tilt.clamp(0.0, 1.0);
    let tilt_exp = (1.85 - modal_tilt * 1.45).clamp(0.12, 2.2);
    let mut modes = Vec::with_capacity(ratios.len().max(1));
    for (idx, ratio) in ratios.iter().copied().enumerate() {
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let k = (idx + 1) as f32;
        let gain = 1.0 / k.powf(tilt_exp);
        let t60_s = ((0.35 + modal_tilt * 1.4) / (1.0 + 0.09 * k)).max(0.03);
        let in_gain = (1.0 / (1.0 + 0.04 * k)).max(0.02);
        modes.push(ModalMode {
            ratio,
            t60_s,
            gain,
            in_gain,
        });
    }

    if modes.is_empty() {
        modes.push(ModalMode {
            ratio: 1.0,
            t60_s: 0.8,
            gain: 1.0,
            in_gain: 1.0,
        });
    }
    normalize_modal_gains(&mut modes);
    modes
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
        let dark = modal_modes_from_ratios(&ratios, modal_tilt_from_brightness(0.0));
        let bright = modal_modes_from_ratios(&ratios, modal_tilt_from_brightness(1.0));

        assert!(bright[3].gain > dark[3].gain);
        assert!(bright[3].t60_s > dark[3].t60_s);
    }
}
