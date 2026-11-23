use super::*;

impl SineForecast {
    pub(crate) fn at(self, tick: u64) -> Option<[f64; 3]> {
        let elapsed = tick.checked_sub(self.first_sample)?;
        let [x, y] = self.state.map(f64::from);
        let [c, s] = self.rotation.map(f64::from);
        let omega = s.atan2(c);
        let steps = elapsed as f64 + 1.;
        let (sin, cos) = (omega * steps).sin_cos();
        // The renderer periodically restores unit radius; coefficient roundoff
        // is not a physical growth/decay rate to extrapolate over the horizon.
        let gain = 1. + f64::from(self.boost) * f64::from(self.boost_decay).powf(elapsed as f64);
        let result = [
            (x * cos - y * sin) * gain,
            (x * sin + y * cos) * gain,
            omega,
        ];
        result.iter().all(|x| x.is_finite()).then_some(result)
    }
}
