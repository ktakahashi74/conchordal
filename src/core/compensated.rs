//! Retained f64 sums, including cancellation-safe differences of their parts.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Sum {
    pub(crate) hi: f64,
    pub(crate) lo: f64,
}

impl Sum {
    pub(crate) const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    pub(crate) fn add(&mut self, value: f64) {
        let sum = self.hi + value;
        self.lo += if self.hi.abs() >= value.abs() {
            (self.hi - sum) + value
        } else {
            (value - sum) + self.hi
        };
        self.hi = sum;
    }

    pub(crate) fn total(self) -> f64 {
        self.hi + self.lo
    }

    pub(crate) fn difference(self, other: Self) -> f64 {
        // Grow an exact four-term expansion before the one final f64 rounding.
        let mut expansion = [0.0; 4];
        for mut value in [self.hi, -other.hi, self.lo, -other.lo] {
            for part in &mut expansion {
                let sum = *part + value;
                let tail = sum - *part;
                value = (*part - (sum - tail)) + (value - tail);
                *part = sum;
            }
        }
        expansion.into_iter().rev().sum()
    }
}
