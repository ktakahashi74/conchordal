// Source-equation witness retained from FB-1.
use rustfft::num_traits::Float;
// Centered moments avoid subtracting large unwrapped phase coordinates.
struct Moments<T> {
    rate: T,
    delta: T,
    second: T,
}

impl<T: Float> Moments<T> {
    fn variance_about(&self, displacement: T) -> T {
        self.second - (T::one() + T::one()) * displacement * self.delta
            + displacement * displacement
    }
}

fn moments<T: Float>(
    mu: T,
    variance: T,
    background: T,
    peaks: impl Iterator<Item = (T, T, T)>,
) -> Moments<T> {
    let two = T::one() + T::one();
    let mut rate = background;
    let mut delta = T::zero();
    let mut second = background * variance;
    for (mean, width, strength) in peaks {
        let sum = variance + width;
        let distance = mean - mu;
        let intensity = strength * (-(distance * distance) / (two * sum)).exp()
            / (T::from(std::f64::consts::TAU).unwrap() * sum).sqrt();
        let shift = distance * variance / sum;
        let k = variance * width / sum;
        rate = rate + intensity;
        delta = delta + intensity * shift;
        second = second + intensity * (k + shift * shift);
    }
    Moments {
        rate,
        delta: delta / rate,
        second: second / rate,
    }
}

