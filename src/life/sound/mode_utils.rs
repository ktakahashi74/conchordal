use crate::life::sound::modal_engine::ModalMode;

pub(crate) fn sanitize_mode_ratios(mut ratios: Vec<f32>) -> Vec<f32> {
    ratios.retain(|ratio| ratio.is_finite() && *ratio > 0.0);
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ratios.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-4);
    if ratios.is_empty() { vec![1.0] } else { ratios }
}

pub(crate) fn modal_modes_from_ratios(
    ratios: &[f32],
    brightness: f32,
    width: f32,
) -> Vec<ModalMode> {
    let brightness = brightness.clamp(0.0, 1.0);
    let width = width.clamp(0.0, 1.0);
    let detuned = detuned_ratios_for_width(ratios, width);

    let tilt_exp = (1.85 - brightness * 1.45).clamp(0.12, 2.2);
    let mut modes = Vec::with_capacity(detuned.len().max(1));
    for (idx, ratio) in detuned.iter().copied().enumerate() {
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let k = (idx + 1) as f32;
        let gain = 1.0 / k.powf(tilt_exp);
        let t60_s = ((0.35 + brightness * 1.4) / (1.0 + 0.09 * k)).max(0.03);
        let in_gain = ((1.0 - width * 0.45) / (1.0 + 0.04 * k)).max(0.02);
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

fn detuned_ratios_for_width(ratios: &[f32], width: f32) -> Vec<f32> {
    let width = width.clamp(0.0, 1.0);
    let max_cents = width * 18.0;
    if max_cents <= 0.0 {
        return ratios.to_vec();
    }

    let mut out = Vec::with_capacity(ratios.len());
    for (idx, ratio) in ratios.iter().copied().enumerate() {
        let shape = match idx % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
        let cents = shape * max_cents * (1.0 + idx as f32 * 0.015);
        let detune = 2.0f32.powf(cents / 1200.0);
        out.push(ratio * detune);
    }
    out
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
