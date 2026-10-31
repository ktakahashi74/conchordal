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

    /// Accumulate a product of raw f32 differences in double-double precision.
    pub(crate) fn add_product(&mut self, left: [f32; 2], right: [f32; 2]) {
        let two_sum = |a: f64, b: f64| {
            let hi = a + b;
            let tail = hi - a;
            Self {
                hi,
                lo: (a - (hi - tail)) + (b - tail),
            }
        };
        // Scalar products retain normalized DD pairs. After the leading
        // error-free sum, FastTwoSum renormalizes its accumulated low part.
        let quick_two_sum = |a: f64, b: f64| {
            let hi = a + b;
            Self {
                hi,
                lo: b - (hi - a),
            }
        };
        // A finite f32 difference needs at most 24 + exponent span bits,
        // plus one for opposite signs. Zero and subnormals need no more.
        // If the two widths total <= 53, both differences and their product
        // are exact f64 values. Only the DD sum then needs rounding repair.
        let width = |[a, b]: [f32; 2]| {
            let a = a.to_bits();
            let b = b.to_bits();
            let ea = (a >> 23) & 255;
            let eb = (b >> 23) & 255;
            if ea == 255 || eb == 255 {
                return 512;
            }
            if a & 0x7fff_ffff == 0 || b & 0x7fff_ffff == 0 {
                return 24;
            }
            24 + ea.max(1).abs_diff(eb.max(1)) + ((a ^ b) >> 31)
        };
        if width(left) + width(right) <= 53 {
            let a = left[0] as f64 - left[1] as f64;
            let b = right[0] as f64 - right[1] as f64;
            let sum = two_sum(self.hi, a * b);
            *self = quick_two_sum(sum.hi, sum.lo + self.lo);
            return;
        }
        let a = two_sum(left[0] as f64, -(left[1] as f64));
        let b = two_sum(right[0] as f64, -(right[1] as f64));
        let product = a.hi * b.hi;
        let error = a.hi.mul_add(b.hi, -product);
        let p = two_sum(product, error + (a.hi * b.lo + a.lo * b.hi));
        let p = two_sum(p.hi, p.lo + a.lo * b.lo);
        // Renormalize both retained parts at every step, not just their total.
        let s = two_sum(self.hi, p.hi);
        let t = two_sum(self.lo, p.lo);
        let u = quick_two_sum(s.hi, s.lo + t.hi);
        *self = quick_two_sum(u.hi, u.lo + t.lo);
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
