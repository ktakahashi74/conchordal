use super::individual::PureTone;

#[derive(Clone, Debug)]
pub struct PopulationParams {
    pub initial_tones_hz: Vec<f32>,
    pub amplitude: f32,
}

#[derive(Clone, Debug)]
pub struct Population {
    pub tones: Vec<PureTone>,
}

impl Population {
    pub fn new(p: PopulationParams) -> Self {
        let tones = p
            .initial_tones_hz
            .into_iter()
            .map(|f| PureTone::new(f, p.amplitude))
            .collect();
        Self { tones }
    }

    /// Project tones to magnitude spectrum bins using simple triangular (soft) binning.
    pub fn project_spectrum(&self, n_bins: usize, fs: f32, n_fft: usize) -> Vec<f32> {
        let mut amps = vec![0.0f32; n_bins];

	for t in &self.tones {
            let bin_f = t.freq_hz * n_fft as f32 / fs;
            let k = bin_f.round() as isize;
            if k >= 0 && (k as usize) < n_bins {
		amps[k as usize] += t.amp;
            }
	}
	amps
    }
}
