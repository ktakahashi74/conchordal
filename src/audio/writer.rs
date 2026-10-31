use std::sync::Arc;

use crossbeam_channel::Receiver;
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::audio::limiter::{Limiter, LimiterMeter, LimiterMode, LimiterStats};

pub struct WavOutput {
    // Writer is kept alive in the thread
}

impl WavOutput {
    pub fn run(
        rx: Receiver<Arc<[f32]>>,
        path: String,
        sample_rate: u32,
        guard_mode: LimiterMode,
        guard_meter: Option<Arc<LimiterMeter>>,
        output_gain: f32,
    ) -> std::thread::JoinHandle<Result<(), String>> {
        std::thread::spawn(move || {
            Self::write(
                rx,
                path,
                sample_rate,
                guard_mode,
                guard_meter,
                output_gain,
                u32::MAX,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        rx: Receiver<Arc<[f32]>>,
        path: String,
        sample_rate: u32,
        guard_mode: LimiterMode,
        guard_meter: Option<Arc<LimiterMeter>>,
        output_gain: f32,
        size_limit: u32,
    ) -> Result<(), String> {
        (|| -> hound::Result<()> {
            let mut guard = Limiter::new(guard_mode, sample_rate, 1).with_input_gain(output_gain);
            if let Some(meter) = guard_meter {
                guard = guard.with_meter(meter);
            }
            let mut scratch: Vec<f32> = Vec::new();
            let spec = WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };
            let mut writer = WavWriter::create(path, spec)?;
            // Hound uses PCM headers for <=16-bit mono/stereo, extensible otherwise.
            let riff_overhead = if spec.channels > 2 || spec.bits_per_sample > 16 {
                60
            } else {
                36
            };
            let bytes_per_frame =
                u64::from(spec.bits_per_sample.div_ceil(8)) * u64::from(spec.channels);
            let max_frames = u64::from(size_limit.saturating_sub(riff_overhead)) / bytes_per_frame;
            let mut frames_written = 0u64;
            let mut total = LimiterStats::default();

            while let Ok(samples) = rx.recv() {
                if scratch.len() != samples.len() {
                    scratch.resize(samples.len(), 0.0);
                }
                scratch.copy_from_slice(&samples);
                guard.process_interleaved(&mut scratch, 1);
                let stats = guard.stats();
                total.max_abs_in = total.max_abs_in.max(stats.max_abs_in);
                total.max_abs_out = total.max_abs_out.max(stats.max_abs_out);
                total.max_reduction_db = total.max_reduction_db.max(stats.max_reduction_db);
                total.num_over += stats.num_over;
                for &s in scratch.iter() {
                    if frames_written >= max_frames {
                        writer.finalize()?;
                        return Err(hound::Error::IoError(std::io::Error::other(format!(
                            "WAV size limit reached at t={:.6}s",
                            frames_written as f64 / sample_rate as f64,
                        ))));
                    }
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    writer.write_sample(v)?;
                    frames_written += 1;
                }
            }

            writer.finalize()?;
            tracing::info!(
                output_gain,
                max_abs_in = total.max_abs_in,
                max_abs_out = total.max_abs_out,
                max_reduction_db = total.max_reduction_db,
                num_over = total.num_over,
                "WAV output peaks"
            );
            Ok(())
        })()
        .map_err(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_limit_finalizes_valid_header_before_the_overflowing_sample() {
        let path =
            std::env::temp_dir().join(format!("conchordal-ipu3-wav-{}.wav", std::process::id()));
        let (tx, rx) = crossbeam_channel::bounded(2);
        tx.send(Arc::from([0.1, -0.1])).unwrap();
        tx.send(Arc::from([0.2, -0.2, 0.3])).unwrap();
        drop(tx);
        // RIFF overhead 36 + four complete 16-bit mono frames.
        let error = WavOutput::write(
            rx,
            path.to_string_lossy().into_owned(),
            48_000,
            LimiterMode::None,
            None,
            1.0,
            44,
        )
        .unwrap_err();
        assert_eq!(error, "WAV size limit reached at t=0.000083s");
        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.duration(), 4);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(
            reader
                .samples::<i16>()
                .map(Result::unwrap)
                .collect::<Vec<_>>(),
            [3276, -3276, 6553, -6553]
        );
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 44);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 8);
        assert_eq!(bytes.len(), 52);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wav_exact_limit_succeeds_and_io_errors_are_returned() {
        let path =
            std::env::temp_dir().join(format!("conchordal-ipu3-exact-{}.wav", std::process::id()));
        let (tx, rx) = crossbeam_channel::bounded(1);
        tx.send(Arc::from([0.0; 4])).unwrap();
        drop(tx);
        WavOutput::write(
            rx,
            path.to_string_lossy().into_owned(),
            48_000,
            LimiterMode::None,
            None,
            1.0,
            44,
        )
        .unwrap();
        assert_eq!(hound::WavReader::open(&path).unwrap().duration(), 4);
        std::fs::remove_file(&path).unwrap();
        let (tx, rx) = crossbeam_channel::bounded(1);
        drop(tx);
        assert!(
            WavOutput::write(
                rx,
                path.join("missing.wav").to_string_lossy().into_owned(),
                48_000,
                LimiterMode::None,
                None,
                1.0,
                u32::MAX
            )
            .is_err()
        );
    }
}
