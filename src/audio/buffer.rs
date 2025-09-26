// Generic audio buffer type to support mono/stereo interleaved data
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    pub fn mono(samples: Vec<f32>) -> Self {
        Self { channels: 1, samples }
    }

    pub fn stereo(left: Vec<f32>, right: Vec<f32>) -> Self {
        assert_eq!(left.len(), right.len());
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for (l, r) in left.into_iter().zip(right.into_iter()) {
            interleaved.push(l);
            interleaved.push(r);
        }
        Self { channels: 2, samples: interleaved }
    }
}
