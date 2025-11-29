use crossbeam_channel::Receiver;
use hound::{SampleFormat, WavSpec, WavWriter};

pub struct WavOutput {
    // Writer is kept alive in the thread
}

impl WavOutput {
    pub fn run(
        rx: Receiver<Vec<f32>>,
        path: String,
        sample_rate: u32,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let spec = WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };
            let mut writer = WavWriter::create(path, spec).expect("create wav");

            while let Ok(samples) = rx.recv() {
                for s in samples {
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    writer.write_sample(v).unwrap();
                }
            }

            writer.finalize().unwrap();
        })
    }
}
