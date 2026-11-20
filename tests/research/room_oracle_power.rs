// Included only in native loo-diagnostic tests.
#[derive(Clone)]
struct PowerOracle {
    own: Bank,
    guard: bool,
}

impl AuditoryFrontEnd {
    pub(crate) fn oracle_own(&mut self, pcm: &[f32], guard: bool) {
        let oracle = self.oracle.get_or_insert_with(|| {
            let mut own = self.bank.clone();
            own.clear(); // A newborn has no pre-birth sound.
            Box::new(PowerOracle { own, guard })
        });
        assert_eq!(oracle.guard, guard);
        oracle
            .own
            .read_boxcar(pcm.iter().map(|&x| Complex32::new(x, 0.0)));
    }
}

impl PowerOracle {
    fn subtract(&self, mix: &mut Bank, samples: usize) {
        let mut totals = [0.; 4];
        for c in 0..mix.centers.len() {
            mix.energy[c] = (mix.energy[c] - self.own.energy[c]).max(0.);
            mix.correlation[c] -= self.own.correlation[c];
            let j = mix.correlation[c];
            if j == Complex32::default() {
                continue;
            }
            let f = mix.fs * j.arg() / TAU;
            let outside =
                f <= 0. || (erbn(f64::from(f)) - erbn(f64::from(mix.centers[c]))).abs() > 1.;
            let hops = mix.statistics.as_ref().unwrap().counts[c];
            let power = 2. * f64::from(j.norm()) / (samples * hops) as f64;
            totals[0] += 1.;
            totals[1] += power;
            if outside && self.guard {
                totals[2] += 1.;
                totals[3] += power;
                mix.correlation[c] = Complex32::default();
            }
        }
        crate::runtime::room_performance::oracle::rejections(totals);
    }
}
