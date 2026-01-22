use std::sync::Arc;

use crossbeam_channel::Receiver;
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::audio::limiter::{Limiter, LimiterMeter, LimiterMode};

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
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut guard = Limiter::new(guard_mode, sample_rate, 1);
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

            while let Ok(samples) = rx.recv() {
                if scratch.len() != samples.len() {
                    scratch.resize(samples.len(), 0.0);
                }
                scratch.copy_from_slice(&samples);
                guard.process_interleaved(&mut scratch, 1);
                for &s in scratch.iter() {
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    writer.write_sample(v).unwrap();
                }
            }

            writer.finalize().unwrap();
        })
    }
}
