use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;

use crate::audio::limiter::{Limiter, LimiterMeter, LimiterMode};

use anyhow::Context;
use ringbuf::traits::*;
use ringbuf::{HeapCons, HeapProd, HeapRb};
use tracing::{debug, info};

/// Module for connecting to the output device.
pub struct AudioOutput {
    stream: Option<cpal::Stream>,
    pub config: cpal::StreamConfig,
}

impl AudioOutput {
    /// Start AudioOutput and return a Producer for the worker loop.
    pub fn new(
        latency_ms: f32,
        guard_mode: LimiterMode,
        guard_meter: Option<Arc<LimiterMeter>>,
    ) -> anyhow::Result<(Self, HeapProd<f32>)> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No default output device")?;

        let supported_config = device
            .default_output_config()
            .context("No default config")?;
        let sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();

        let config = cpal::StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let target_frames = (sample_rate as f32 * latency_ms / 1000.0).round().max(1.0) as usize;
        let min_frames = match supported_config.buffer_size() {
            cpal::SupportedBufferSize::Range { min, .. } => (*min as usize) * 2,
            cpal::SupportedBufferSize::Unknown => 512,
        };
        let capacity_frames = (target_frames * 2).max(min_frames);
        if capacity_frames > target_frames * 2 {
            info!(
                "Audio buffer raised to min frames: target={} actual={} (sr={} ch={})",
                target_frames * 2,
                capacity_frames,
                sample_rate,
                channels
            );
        }
        info!(
            "Audio buffer config: sr={} ch={} target_frames={} capacity_frames={}",
            sample_rate, channels, target_frames, capacity_frames
        );
        let rb = HeapRb::<f32>::new(capacity_frames);
        let (prod, mut cons): (HeapProd<f32>, HeapCons<f32>) = rb.split();

        let mut guard = Limiter::new(guard_mode, sample_rate, channels as usize);
        if let Some(meter) = guard_meter {
            guard = guard.with_meter(meter);
        }

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let n_frames = data.len() / channels as usize;

                    for frame in 0..n_frames {
                        // Pop one frame of samples.
                        let s = cons.try_pop().unwrap_or(0.0);

                        // Mono: copy to all channels.
                        for ch in 0..channels {
                            data[frame * channels as usize + ch as usize] = s;
                        }
                    }

                    guard.process_interleaved(data, channels as usize);
                },
                |err| eprintln!("Stream error: {:?}", err),
                None,
            )
            .context("Failed to build output stream")?;
        stream.play().context("Failed to start output stream")?;

        Ok((
            Self {
                stream: Some(stream),
                config,
            },
            prod,
        ))
    }

    pub fn stop(&mut self) {
        self.stream.take(); // take and Drop
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

    // pub fn buffered_samples(&self) -> usize {
    //     self._rb.occupied_len()
    // }

    // pub fn capacity(&self) -> usize {
    //     self._rb.capacity().get()
    // }
}

impl Drop for AudioOutput {
    fn drop(&mut self) {
        if self.stream.is_some() {
            debug!("AudioOutput drop: stopping CPAL stream.");
        }
        self.stream.take();
    }
}
