use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use ringbuf::{HeapRb, HeapProd, HeapCons};
use ringbuf::traits::*;



/// 出力デバイスに接続するモジュール
pub struct AudioOutput {
    stream: Option<cpal::Stream>,
    capacity: usize,
    pub config: cpal::StreamConfig,
}

impl AudioOutput {
    /// AudioOutput を開始し、ワーカーループ側が push できる Producer を返す
    pub fn new(latency_ms: f32) -> (Self, HeapProd<f32>) {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("No output device");
	
        let supported_config = device.default_output_config().expect("No default config");
        let sample_rate = supported_config.sample_rate().0;
        let channels = supported_config.channels();

	let config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let capacity = (sample_rate as f32 * latency_ms / 1000.0) as usize;
        let rb = HeapRb::<f32>::new(capacity * channels as usize * 10);
        let (prod, mut cons): (HeapProd<f32>, HeapCons<f32>) = rb.split();

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
		    eprintln!("CPAL callback: requested {} samples", data.len());

		    let mut frames_filled = 0;
		    let n_frames = data.len() / channels as usize;
		    
		    for frame in 0..n_frames {
			// 1フレーム分のサンプル値を取り出す
			let s = cons.try_pop().unwrap_or(0.0);
			
			// モノラル: 全チャンネルに同じ値を複製
			for ch in 0..channels {
			    data[frame * channels as usize + ch as usize] = s;
			}
			
			frames_filled += 1;
		    }
		},

                |err| eprintln!("Stream error: {:?}", err),
                None,
            )
            .unwrap();
        stream.play().unwrap();

        (
            Self {
                stream: Some(stream),
		capacity,
		config
            },
            prod,
        )
    }

    pub fn stop(&mut self) {
	self.stream.take(); // take and Drop
    }
    
    /// ワーカーループが新しいサンプルを push
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


impl Drop for  AudioOutput {
    fn drop(&mut self) {
	if self.stream.is_some() {
	    eprintln!("AudioOutput drop: stopping CPAL stream.");
	}
	self.stream.take();
    }
}
