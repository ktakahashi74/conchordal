use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::audio::limiter::{Limiter, LimiterMeter, LimiterMode};

use anyhow::{Context, ensure};
use ringbuf::traits::*;
use ringbuf::{HeapCons, HeapProd, HeapRb};
use serde::Serialize;
use tracing::{debug, info};

const PREFERRED_OUTPUT_SAMPLE_RATE: u32 = 48_000;

fn buffer_capacity_frames(
    sample_rate: u32,
    latency_ms: f32,
    hop_size: usize,
    min_frames: usize,
) -> anyhow::Result<usize> {
    ensure!(sample_rate > 0, "audio sample rate must be positive");
    ensure!(
        latency_ms.is_finite() && latency_ms > 0.0,
        "audio.latency_ms must be finite and positive"
    );
    ensure!(hop_size > 0, "analysis.hop_size must be positive");
    let target_frames = (sample_rate as f64 * latency_ms as f64 / 1000.0)
        .round()
        .max(1.0);
    let max_frames = isize::MAX as usize / std::mem::size_of::<f32>();
    ensure!(
        target_frames < (max_frames / 2) as f64,
        "audio.latency_ms exceeds the addressable audio buffer size"
    );
    let capacity = (target_frames as usize)
        .checked_mul(2)
        .context("audio buffer capacity overflow")?
        .max(min_frames)
        .max(hop_size);
    ensure!(capacity <= max_frames, "audio buffer capacity is too large");
    Ok(capacity)
}

fn fill_output(
    cons: &mut HeapCons<f32>,
    data: &mut [f32],
    channels: usize,
    underrun_frames: &AtomicU64,
) {
    let mut missing = 0u64;
    for frame in data.chunks_exact_mut(channels) {
        let sample = cons.try_pop().unwrap_or_else(|| {
            missing += 1;
            0.0
        });
        frame.fill(sample);
    }
    if missing > 0 {
        underrun_frames.fetch_add(missing, Ordering::Relaxed);
    }
}

fn select_output_config(device: &cpal::Device) -> anyhow::Result<cpal::SupportedStreamConfig> {
    let default_config = device
        .default_output_config()
        .context("No default config")?;
    if default_config.sample_rate() == PREFERRED_OUTPUT_SAMPLE_RATE {
        return Ok(default_config);
    }

    let default_format = default_config.sample_format();
    let default_channels = default_config.channels();

    let preferred = match device.supported_output_configs() {
        Ok(configs) => configs
            .filter(|range| range.sample_format() == default_format)
            .filter(|range| {
                let min = range.min_sample_rate();
                let max = range.max_sample_rate();
                min <= PREFERRED_OUTPUT_SAMPLE_RATE && PREFERRED_OUTPUT_SAMPLE_RATE <= max
            })
            .max_by_key(|range| {
                (
                    u8::from(range.channels() == default_channels),
                    range.channels(),
                )
            })
            .map(|range| range.with_sample_rate(PREFERRED_OUTPUT_SAMPLE_RATE)),
        Err(err) => {
            debug!("Could not enumerate supported output configs: {err}");
            None
        }
    };

    if let Some(config) = preferred {
        info!(
            "Audio output config: preferring {} Hz over default {} Hz (ch={} fmt={:?})",
            PREFERRED_OUTPUT_SAMPLE_RATE,
            default_config.sample_rate(),
            config.channels(),
            config.sample_format()
        );
        return Ok(config);
    }

    info!(
        "Audio output config: preferred {} Hz unavailable; using default {} Hz (ch={} fmt={:?})",
        PREFERRED_OUTPUT_SAMPLE_RATE,
        default_config.sample_rate(),
        default_channels,
        default_format
    );
    Ok(default_config)
}

/// Module for connecting to the output device.
pub struct AudioOutput {
    stream: Option<cpal::Stream>,
    pub config: cpal::StreamConfig,
    device_info: AudioDeviceInfo,
    counters: Arc<AudioCallbackCounters>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AudioDeviceInfo {
    pub(crate) backend: String,
    pub(crate) device_name: String,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) ring_capacity_frames: usize,
}

#[derive(Default)]
pub(crate) struct AudioCallbackCounters {
    pub(crate) callback_count: AtomicU64,
    pub(crate) callback_frames_total: AtomicU64,
    pub(crate) underrun_frames: Arc<AtomicU64>,
    pub(crate) callback_errors_total: AtomicU64,
}

impl AudioOutput {
    /// Start AudioOutput and return a Producer for the worker loop.
    pub fn new(
        latency_ms: f32,
        hop_size: usize,
        guard_mode: LimiterMode,
        guard_meter: Option<Arc<LimiterMeter>>,
    ) -> anyhow::Result<(Self, HeapProd<f32>)> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No default output device")?;
        let device_name = device
            .description()
            .context("No audio device description")?
            .name()
            .to_string();

        let supported_config = select_output_config(&device)?;
        let sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();
        ensure!(channels > 0, "audio output device has no channels");

        let config = cpal::StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let min_frames = match supported_config.buffer_size() {
            cpal::SupportedBufferSize::Range { min, .. } => (*min as usize)
                .checked_mul(2)
                .context("audio device buffer size overflow")?,
            cpal::SupportedBufferSize::Unknown => 512,
        };
        let capacity_frames =
            buffer_capacity_frames(sample_rate, latency_ms, hop_size, min_frames)?;
        info!(
            "Audio buffer config: sr={} ch={} latency_ms={} hop_size={} capacity_frames={}",
            sample_rate, channels, latency_ms, hop_size, capacity_frames
        );
        let rb =
            HeapRb::<f32>::try_new(capacity_frames).context("Failed to allocate audio buffer")?;
        let (prod, mut cons): (HeapProd<f32>, HeapCons<f32>) = rb.split();
        let counters = Arc::new(AudioCallbackCounters::default());
        let callback_counters = Arc::clone(&counters);
        let error_counters = Arc::clone(&counters);

        let mut guard = Limiter::new(guard_mode, sample_rate, channels as usize);
        if let Some(meter) = guard_meter {
            guard = guard.with_meter(meter);
        }

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    callback_counters
                        .callback_count
                        .fetch_add(1, Ordering::Relaxed);
                    callback_counters
                        .callback_frames_total
                        .fetch_add((data.len() / channels as usize) as u64, Ordering::Relaxed);
                    fill_output(
                        &mut cons,
                        data,
                        channels as usize,
                        &callback_counters.underrun_frames,
                    );
                    guard.process_interleaved(data, channels as usize);
                },
                move |err| {
                    error_counters
                        .callback_errors_total
                        .fetch_add(1, Ordering::Relaxed);
                    eprintln!("Stream error: {:?}", err);
                },
                None,
            )
            .context("Failed to build output stream")?;
        stream.play().context("Failed to start output stream")?;

        Ok((
            Self {
                stream: Some(stream),
                config,
                device_info: AudioDeviceInfo {
                    backend: format!("{:?}", host.id()),
                    device_name,
                    sample_rate,
                    channels,
                    ring_capacity_frames: capacity_frames,
                },
                counters,
            },
            prod,
        ))
    }

    pub(crate) fn underrun_frames(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.counters.underrun_frames)
    }

    pub(crate) fn device_info(&self) -> AudioDeviceInfo {
        self.device_info.clone()
    }

    pub(crate) fn counters(&self) -> Arc<AudioCallbackCounters> {
        Arc::clone(&self.counters)
    }

    /// Worker loop pushes new samples.
    pub fn push_samples(prod: &mut HeapProd<f32>, samples: &[f32]) {
        let mut offset = 0;
        while offset < samples.len() {
            let written = prod.push_slice(&samples[offset..]);
            offset += written;

            if offset < samples.len() {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
    }
}

impl Drop for AudioOutput {
    fn drop(&mut self) {
        if self.stream.is_some() {
            debug!("AudioOutput drop: stopping CPAL stream.");
        }
        self.stream.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_latency_buffer_accepts_a_complete_worker_hop() {
        let capacity = buffer_capacity_frames(48_000, 0.1, 4096, 512).unwrap();
        assert_eq!(capacity, 4096);
        let (mut prod, _cons) = HeapRb::<f32>::new(capacity).split();
        assert_eq!(prod.push_slice(&[0.25; 4096]), 4096);
    }

    #[test]
    fn buffer_capacity_respects_latency_and_device_minimum() {
        assert_eq!(
            buffer_capacity_frames(48_000, 50.0, 512, 256).unwrap(),
            4800
        );
        assert_eq!(
            buffer_capacity_frames(48_000, 0.1, 256, 1024).unwrap(),
            1024
        );
    }

    #[test]
    fn invalid_buffer_dimensions_return_errors() {
        for latency in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(buffer_capacity_frames(48_000, latency, 512, 512).is_err());
        }
        assert!(buffer_capacity_frames(0, 50.0, 512, 512).is_err());
        assert!(buffer_capacity_frames(48_000, 50.0, 0, 512).is_err());
        assert!(buffer_capacity_frames(48_000, 50.0, usize::MAX, 512).is_err());
        assert!(buffer_capacity_frames(48_000, 50.0, 512, usize::MAX).is_err());
    }

    #[test]
    fn callback_counts_missing_frames_and_duplicates_channels() {
        let (mut prod, mut cons) = HeapRb::<f32>::new(4).split();
        assert_eq!(prod.push_slice(&[0.25, -0.5]), 2);
        let missing = AtomicU64::new(0);
        let mut data = [9.0; 8];
        fill_output(&mut cons, &mut data, 2, &missing);
        assert_eq!(data, [0.25, 0.25, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(missing.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn callback_does_not_count_queued_silence_as_underrun() {
        let (mut prod, mut cons) = HeapRb::<f32>::new(4).split();
        assert_eq!(prod.push_slice(&[0.0; 4]), 4);
        let missing = AtomicU64::new(0);
        let mut data = [9.0; 4];
        fill_output(&mut cons, &mut data, 1, &missing);
        assert_eq!(data, [0.0; 4]);
        assert_eq!(missing.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn callback_counter_can_be_drained_between_callbacks() {
        let (_prod, mut cons) = HeapRb::<f32>::new(4).split();
        let missing = AtomicU64::new(0);
        let mut data = [9.0; 6];
        fill_output(&mut cons, &mut data, 2, &missing);
        fill_output(&mut cons, &mut data, 2, &missing);
        assert_eq!(missing.swap(0, Ordering::Relaxed), 6);
        fill_output(&mut cons, &mut data, 2, &missing);
        assert_eq!(missing.swap(0, Ordering::Relaxed), 3);
    }
}
