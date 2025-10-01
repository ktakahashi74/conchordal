use crate::core::gammatone::gammatone_filterbank;
use crate::core::hilbert::hilbert_envelope;
use crate::core::gammatone::erb_hz;

/// R（ラフネス）を計算する
/// input: 波形
/// center_freqs: ガンマトーン中心周波数リスト
/// sample_rate: サンプリング周波数
pub fn roughness_r(input: &[f32], center_freqs: &[f32], sample_rate: f32) -> f32 {
    // 1. ガンマトーンフィルタバンク
    let channels = gammatone_filterbank(input, center_freqs, sample_rate);

    // 2. 包絡に変換
    let envelopes: Vec<Vec<f32>> = channels
        .iter()
        .map(|ch| hilbert_envelope(ch))
        .collect();

    // 3. 各チャネルの平均エネルギー
    let means: Vec<f32> = envelopes
        .iter()
        .map(|env| env.iter().map(|v| v.abs()).sum::<f32>() / env.len() as f32)
        .collect();

    // 4. ペアごとの相互作用を合計
    let mut r_total = 0.0;
    for i in 0..center_freqs.len() {
        for j in (i+1)..center_freqs.len() {
            let fc = 0.5 * (center_freqs[i] + center_freqs[j]);
            let erb = erb_hz(fc);
            let df = (center_freqs[i] - center_freqs[j]).abs();
            let w = (-0.5 * (df / erb).powi(2)).exp();
            r_total += w * (means[i] * means[j]);
        }
    }

    r_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::roughness::roughness_r;

    fn sine(fs: f32, f: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin()).collect()
    }

    #[test]
    fn roughness_is_low_for_single_tone() {
        let fs = 16000.0;
        let x = sine(fs, 440.0, 4096);
        let cfs = vec![440.0, 880.0, 1320.0];
        let r = roughness_r(&x, &cfs, fs);
        assert!(r < 0.1, "R = {}", r);
    }

    #[test]
    fn roughness_increases_for_close_tones() {
        let fs = 16000.0;
        let n = 4096;
        let x1 = sine(fs, 440.0, n);
        let x2 = sine(fs, 450.0, n);
        let x: Vec<f32> = x1.iter().zip(x2.iter()).map(|(a, b)| a + b).collect();
        let cfs = vec![440.0, 450.0, 1000.0];
        let r = roughness_r(&x, &cfs, fs);
        assert!(r > 0.1, "R = {}", r);
    }
}
