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
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
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
            let mut writer = WavWriter::create(path, spec).expect("create wav");
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
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    writer.write_sample(v).unwrap();
                }
            }

            writer.finalize().unwrap();
            tracing::info!(
                output_gain,
                max_abs_in = total.max_abs_in,
                max_abs_out = total.max_abs_out,
                max_reduction_db = total.max_reduction_db,
                num_over = total.num_over,
                "WAV output peaks"
            );
        })
    }
}
