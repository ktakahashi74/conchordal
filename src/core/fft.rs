use rustfft::{num_complex::Complex32, FftPlanner};

pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::TAU * i as f32 / n as f32;
            0.5 * (1.0 - f32::cos(x))
        })
        .collect()
}

pub struct ISTFT {
    pub n: usize,
    pub hop: usize,          // = n/2 を想定
    pub window: Vec<f32>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    tmp: Vec<Complex32>,
    ola_buffer: Vec<f32>,  // n サンプルぶん確保
    write_pos: usize,      // 次に書き込む位置
}

impl ISTFT {
    pub fn new(n: usize, hop: usize) -> Self {
        assert!(hop == n / 2);
        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(n);
        let window = hann_window(n);

	
        Self {
            n,
            hop,
            window,
            ifft,
	    tmp: vec![Complex32::new(0.0, 0.0); n],
            ola_buffer: vec![0.0; n],
	    write_pos: 0,
        }
    }

    pub fn process(&mut self, spec_half: &[Complex32]) -> Vec<f32> {

	// for (i, c) in spec_half.iter().enumerate().take(32) { // 最初の32ビンだけ
	//     println!("bin {:3} {:7.1} Hz: |X|={:.3}", i, i as f32 * 48000.0 / self.n as f32, c.norm());
	// }

        let n = self.n;
        assert_eq!(spec_half.len(), n / 2 + 1);

        // 1) Hermitian 対称スペクトルを構成
        for k in 0..=n / 2 {
            self.tmp[k] = spec_half[k];
        }
        for k in 1..n / 2 {
            self.tmp[n - k] = self.tmp[k].conj();
        }

        // 2) IFFT
        self.ifft.process(&mut self.tmp);
        let inv_n = 1.0 / (n as f32);

        // 3) 窓を掛ける
        let mut win_frame = vec![0.0f32; n];
        for i in 0..n {
            win_frame[i] = (self.tmp[i].re * inv_n) * self.window[i];
        }

        for i in 0..self.n {
            let idx = (self.write_pos + i) % self.n;
            self.ola_buffer[idx] += win_frame[i];
        }
        self.write_pos = (self.write_pos + self.hop) % self.n;

        let mut out = vec![0.0; self.hop];
        for i in 0..self.hop {
            out[i] = self.ola_buffer[(self.write_pos + i) % self.n];
            self.ola_buffer[(self.write_pos + i) % self.n] = 0.0; // 読み出したぶんをクリア
        }
        out
    }
}


pub fn bin_freqs_hz(fs: f32, n: usize) -> Vec<f32> {
    (0..=n / 2).map(|k| k as f32 * fs / n as f32).collect()
}
