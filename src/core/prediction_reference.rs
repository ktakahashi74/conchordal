//! Test-only independent double-double reference with propagated absolute bounds.
//! Hooks receive raw operands, never production sufficient statistics or factors.

use super::{FEATURES, FORECAST_LEN, RIDGE};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};

const EPS: f64 = 4.930380657631324e-32; // 2^-104, including DD operation roundoff.

fn upper(value: f64) -> f64 {
    // Each nonnegative error expression below has fewer than 16 f64 operations.
    // Cover their own rounding before moving the endpoint outward.
    (value * (1.0 + 16.0 * f64::EPSILON)).next_up()
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Dd(f64, f64);

fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let v = s - a;
    Dd(s, (a - (s - v)) + (b - v))
}
impl Add for Dd {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        let s = two_sum(self.0, b.0);
        let t = two_sum(self.1, b.1);
        let u = two_sum(s.0, s.1 + t.0);
        two_sum(u.0, u.1 + t.1)
    }
}
impl Neg for Dd {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}
impl Sub for Dd {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl Mul for Dd {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let p = self.0 * b.0;
        let e = self.0.mul_add(b.0, -p);
        let cross = self.0 * b.1 + self.1 * b.0;
        let s = two_sum(p, e + cross);
        s + Dd(self.1 * b.1, 0.0)
    }
}
impl Div for Dd {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let q = Dd(self.0 / b.0, 0.0);
        let r = self - b * q;
        let q2 = Dd(r.0 / b.0, 0.0);
        let r2 = r - b * q2;
        q + q2 + Dd(r2.0 / b.0, 0.0)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Bound {
    v: Dd,
    u: f64,
}
impl Bound {
    fn exact(x: f64) -> Self {
        Self {
            v: Dd(x, 0.0),
            u: 0.0,
        }
    }
    fn center(self) -> f64 {
        self.v.0 + self.v.1
    }
    fn magnitude(self) -> f64 {
        self.v.0.abs() + self.v.1.abs()
    }
    fn sqrt(self) -> Self {
        let x = self.center().sqrt();
        if !(x > 0.0 && self.center() > self.u) {
            return Self {
                v: Dd(f64::NAN, 0.0),
                u: f64::INFINITY,
            };
        }
        let v = Dd(x, 0.0) + (self.v - Dd(x, 0.0) * Dd(x, 0.0)) / Dd(2.0 * x, 0.0);
        Self {
            v,
            u: upper(self.u / (self.center().next_down() - self.u).next_down().sqrt() + EPS * x),
        }
    }
    fn clamp(self, lo: f64, hi: f64) -> Self {
        if (self.v - Dd(lo, 0.0)).0 <= -self.u {
            return Self::exact(lo);
        }
        if hi.is_finite() && (self.v - Dd(hi, 0.0)).0 >= self.u {
            return Self::exact(hi);
        }
        Self {
            v: if self.center() < lo {
                Dd(lo, 0.0)
            } else if self.center() > hi {
                Dd(hi, 0.0)
            } else {
                self.v
            },
            u: self.u,
        }
    }
}
impl Add for Bound {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            u: upper(self.u + b.u + EPS * (self.magnitude() + b.magnitude())),
        }
    }
}
impl Neg for Bound {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            u: self.u,
        }
    }
}
impl Sub for Bound {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl Mul for Bound {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            u: upper(
                self.magnitude() * b.u
                    + b.magnitude() * self.u
                    + self.u * b.u
                    + EPS * self.magnitude() * b.magnitude(),
            ),
        }
    }
}
impl Div for Bound {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let denominator = (b.center().abs().next_down() - b.u).next_down();
        if denominator <= 0.0 || !denominator.is_finite() {
            return Self {
                v: Dd(f64::NAN, 0.0),
                u: f64::INFINITY,
            };
        }
        let v = self.v / b.v;
        Self {
            v,
            u: upper(
                (self.u + (v.0.abs() + v.1.abs()) * b.u) / denominator
                    + EPS * (v.0.abs() + v.1.abs()),
            ),
        }
    }
}

// Four-component expansion for independent residual and scalar certification.
// Cascaded TwoSum and FMA products expose every discarded term explicitly.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Quad {
    parts: [f64; 4],
    u: f64,
}
impl Quad {
    fn exact(value: f64) -> Self {
        Self {
            parts: [value, 0.0, 0.0, 0.0],
            u: 0.0,
        }
    }
    fn magnitude(self) -> f64 {
        upper(self.parts.iter().map(|x| x.abs()).sum::<f64>())
    }
    fn add_term(&mut self, mut value: f64) {
        for part in &mut self.parts {
            let sum = two_sum(*part, value);
            *part = sum.0;
            value = sum.1;
        }
        if value != 0.0 {
            self.u = (self.u + value.abs()).next_up();
        }
    }
    fn normalized(mut self) -> Self {
        // TwoSum preserves the exact expansion sum, including cancellation
        // between leading components. Four sweeps move any surviving term up.
        for _ in 0..4 {
            for i in (1..4).rev() {
                let sum = two_sum(self.parts[i - 1], self.parts[i]);
                self.parts[i - 1] = sum.0;
                self.parts[i] = sum.1;
            }
        }
        self
    }
    fn sqrt(self) -> Self {
        let bound = self.bound();
        let lower = (bound.center().next_down() - bound.u)
            .next_down()
            .sqrt()
            .next_down();
        if lower <= 0.0 || !lower.is_finite() {
            return Self {
                parts: [f64::NAN; 4],
                u: f64::INFINITY,
            };
        }
        let mut root = Self::exact(bound.center().sqrt());
        for _ in 0..2 {
            root = (root + self / root) * Self::exact(0.5);
            root.u = 0.0;
        }
        let residual = self - root * root;
        root.u = upper((residual.magnitude() + residual.u) / lower);
        root
    }

    fn bound(self) -> Bound {
        let mut value = Dd(0.0, 0.0);
        let mut error = self.u;
        for part in self.parts {
            let next = value + Dd(part, 0.0);
            // Two retained leading components suffice for the comparison; the
            // original expansion certifies exact zero after cancellation.
            if value.1 != 0.0 && part != 0.0 {
                error = (error + EPS * (value.0.abs() + part.abs())).next_up();
            }
            value = next;
        }
        Bound { v: value, u: error }
    }
}
impl Add for Quad {
    type Output = Self;
    fn add(mut self, b: Self) -> Self {
        for value in b.parts.into_iter().filter(|value| *value != 0.0) {
            self.add_term(value);
        }
        if b.u != 0.0 {
            self.u = (self.u + b.u).next_up();
        }
        self
    }
}
impl Neg for Quad {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            parts: self.parts.map(|x| -x),
            u: self.u,
        }
    }
}
impl Sub for Quad {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl Mul for Quad {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let mut result = Self::default();
        for a in self.parts.into_iter().filter(|value| *value != 0.0) {
            for b in b.parts.into_iter().filter(|value| *value != 0.0) {
                let product = a * b;
                result.add_term(product);
                result.add_term(a.mul_add(b, -product));
            }
        }
        let propagated = self.u * b.magnitude() + b.u * self.magnitude() + self.u * b.u;
        if propagated != 0.0 {
            result.u = upper(result.u + upper(propagated));
        }
        result
    }
}
impl Div for Quad {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let mut result = Self::default();
        let b = b.normalized();
        let mut residual = self;
        for _ in 0..5 {
            residual = residual.normalized();
            let quotient = residual.parts[0] / b.parts[0];
            result.add_term(quotient);
            residual = residual - b * Self::exact(quotient);
        }
        let bound = b.bound();
        let denominator = (bound.center().abs().next_down() - bound.u).next_down();
        if denominator <= 0.0 || !denominator.is_finite() {
            return Self {
                parts: [f64::NAN; 4],
                u: f64::INFINITY,
            };
        }
        let remainder = residual.normalized().magnitude() + residual.u;
        if remainder != 0.0 {
            result.u = upper(result.u + upper(remainder) / denominator);
        }
        result
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Scalar {
    dd: Bound,
    quad: Quad,
}
impl Scalar {
    fn add_product(&mut self, left: [f32; 2], right: [f32; 2]) {
        let dd = |[a, b]: [f32; 2]| Bound::exact(a as f64) - Bound::exact(b as f64);
        let quad = |[a, b]: [f32; 2]| Quad::exact(a as f64) - Quad::exact(b as f64);
        self.dd = self.dd + dd(left) * dd(right);
        self.quad = self.quad + quad(left) * quad(right);
    }

    fn add(&mut self, term: f64) {
        self.dd = self.dd + Bound::exact(term);
        self.quad.add_term(term);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counts {
    pub passed: [u64; 3],
    pub failed: [u64; 3],
    pub inconclusive: [u64; 3],
    pub condition_upper: f64,
    pub scaled_condition_upper: f64,
    pub dd_inconclusive: u64,
    pub uncertified: [u64; 3],
}
impl Counts {
    pub(crate) fn assert_passed(self) {
        assert!(
            self.passed.iter().all(|n| *n > 0),
            "missing outputs: {self:?}"
        );
        assert_eq!(self.failed, [0; 3], "{self:?}");
        assert_eq!(
            self.inconclusive, [0; 3],
            "uncertified values need higher precision: {self:?}"
        );
    }
    fn check(&mut self, kind: usize, prod: f32, reference: Bound, scale: f64) -> bool {
        assert!(prod.is_finite(), "nonfinite production output kind={kind}");
        if !reference.center().is_finite() || !reference.u.is_finite() || !scale.is_finite() {
            self.inconclusive[kind] += 1;
            self.uncertified[kind] += 1;
            eprintln!(
                "reference INCONCLUSIVE kind={kind} reason=nonfinite_certificate prod={prod:e}"
            );
            return false;
        }
        // Subtract in DD, so an f64-rounded reference cannot conceal cancellation.
        let difference = (Dd(prod as f64, 0.0) - reference.v).0.abs();
        let uncertainty = reference.u + f64::EPSILON * difference;
        let magnitude = reference.center().abs();
        let ulp = |v: f64| {
            let exact = v.max(f32::MIN_POSITIVE as f64);
            let rounded = exact as f32;
            let floor = if rounded as f64 > exact {
                rounded.next_down()
            } else {
                rounded
            };
            (floor.next_up() - floor) as f64
        };
        let low = ulp((magnitude - uncertainty).max(0.0)) + 2.0 * f64::EPSILON * scale;
        let high = ulp(magnitude + uncertainty) + 2.0 * f64::EPSILON * scale;
        if let Ok(path) = std::env::var("IP_V7_VALUES") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            let decision = if difference + uncertainty <= low {
                "pass"
            } else if difference - uncertainty > high {
                "fail"
            } else {
                "inconclusive"
            };
            serde_json::to_writer(
                &mut file,
                &serde_json::json!({
                    "kind": kind, "prod": prod, "v_ref": reference.v, "u_ref": reference.u,
                    "d": difference, "tol_min": low, "tol_max": high, "scale": scale,
                    "decision": decision,
                }),
            )
            .unwrap();
            writeln!(file).unwrap();
        }
        if difference + uncertainty <= low {
            self.passed[kind] += 1;
            true
        } else if difference - uncertainty > high {
            self.failed[kind] += 1;
            eprintln!(
                "reference FAIL kind={kind} prod={prod:e} ref={reference:?} d={difference:e} tol={low:e}..{high:e} S={scale:e}"
            );
            true
        } else {
            self.inconclusive[kind] += 1;
            eprintln!(
                "reference INCONCLUSIVE kind={kind} prod={prod:e} ref={reference:?} d={difference:e} tol={low:e}..{high:e} S={scale:e}"
            );
            false
        }
    }

    fn scalar(&mut self, kind: usize, prod: f32, sums: [Scalar; 2], complement: bool) {
        let value = if sums[0].dd.center() > 0.0 {
            let numerator = if complement {
                sums[0].dd - sums[1].dd
            } else {
                sums[1].dd
            };
            (numerator / sums[0].dd).clamp(0.0, 1.0)
        } else {
            Bound::exact(f64::from(complement))
        };
        if !self.check(kind, prod, value, 0.0) {
            self.dd_inconclusive += 1;
            self.inconclusive[kind] -= 1;
            let numerator = if complement {
                sums[0].quad - sums[1].quad
            } else {
                sums[1].quad
            };
            let refined = (numerator / sums[0].quad).bound().clamp(0.0, 1.0);
            self.check(kind, prod, refined, 0.0);
        }
    }
}

#[derive(Clone)]
struct Reference {
    gram: Vec<Vec<Bound>>,
    quad_gram: Vec<Vec<Quad>>,
    quad_response: Vec<[[Quad; 3]; FEATURES]>,
    response: Vec<[[Bound; 3]; FEATURES]>,
    comparison: Vec<[[Scalar; 2]; 3]>,
    issues: u64,
    last_forecast: Option<u64>,
    last_weights: Option<u64>,
    checkpoint_start: Counts,
    checkpoint_counts: Counts,
    condition_upper: f64,
    scaled_condition_upper: f64,
}
impl Reference {
    fn new() -> Self {
        let mut gram = vec![vec![Bound::default(); FEATURES]; FEATURES];
        for (i, row) in gram.iter_mut().enumerate() {
            row[i] = Bound::exact(RIDGE);
        }
        let mut quad_gram = vec![vec![Quad::default(); FEATURES]; FEATURES];
        for (i, row) in quad_gram.iter_mut().enumerate() {
            row[i] = Quad::exact(RIDGE);
        }
        Self {
            gram,
            quad_gram,
            quad_response: vec![[[Quad::default(); 3]; FEATURES]; FORECAST_LEN],
            response: vec![[[Bound::default(); 3]; FEATURES]; FORECAST_LEN],
            comparison: vec![[[Scalar::default(); 2]; 3]; FORECAST_LEN],
            issues: 0,
            last_forecast: None,
            last_weights: None,
            checkpoint_start: Counts::default(),
            checkpoint_counts: Counts::default(),
            condition_upper: 0.0,
            scaled_condition_upper: 0.0,
        }
    }
    fn factor(&self) -> Vec<Vec<Bound>> {
        let mut lower = vec![vec![Bound::default(); FEATURES]; FEATURES];
        for i in 0..FEATURES {
            for j in 0..=i {
                let mut v = self.gram[i][j];
                v.u = 0.0;
                for k in 0..j {
                    v = v - lower[i][k] * lower[j][k];
                    v.u = 0.0;
                }
                lower[i][j] = if i == j { v.sqrt() } else { v / lower[j][j] };
                // Certify the computed factor a posteriori, avoiding interval dependency inflation.
                lower[i][j].u = 0.0;
            }
        }
        lower
    }
    fn solve(lower: &[Vec<Bound>], x: &[f64; FEATURES]) -> [Bound; FEATURES] {
        let mut result = x.map(Bound::exact);
        for i in 0..FEATURES {
            for j in 0..i {
                result[i] = result[i] - lower[i][j] * result[j];
            }
            result[i] = result[i] / lower[i][i];
            result[i].u = 0.0;
        }
        for i in (0..FEATURES).rev() {
            for j in i + 1..FEATURES {
                result[i] = result[i] - lower[j][i] * result[j];
            }
            result[i] = result[i] / lower[i][i];
            result[i].u = 0.0;
        }
        result
    }
    fn quad_projection(&self, x: &[f64; FEATURES]) -> [Quad; FEATURES] {
        // The factor and solves supply candidates only. Discard their interval
        // widths; the independent residual below certifies the final candidate.
        let mut lower = vec![vec![Quad::default(); FEATURES]; FEATURES];
        for i in 0..FEATURES {
            for j in 0..=i {
                let mut value = self.quad_gram[i][j];
                value.u = 0.0;
                for k in 0..j {
                    value = value - lower[i][k] * lower[j][k];
                    value.u = 0.0;
                }
                lower[i][j] = if i == j {
                    value.sqrt()
                } else {
                    value / lower[j][j]
                };
                lower[i][j].u = 0.0;
            }
        }
        let solve = |mut result: [Quad; FEATURES]| {
            for i in 0..FEATURES {
                for j in 0..i {
                    result[i] = result[i] - lower[i][j] * result[j];
                    result[i].u = 0.0;
                }
                result[i] = result[i] / lower[i][i];
                result[i].u = 0.0;
            }
            for i in (0..FEATURES).rev() {
                for j in i + 1..FEATURES {
                    result[i] = result[i] - lower[j][i] * result[j];
                    result[i].u = 0.0;
                }
                result[i] = result[i] / lower[i][i];
                result[i].u = 0.0;
            }
            result
        };
        let residual = |p: &[Quad; FEATURES]| -> [Quad; FEATURES] {
            std::array::from_fn(|i| {
                let mut r = Quad::exact(x[i]);
                for (j, &p) in p.iter().enumerate() {
                    r = r - self.quad_gram[i.max(j)][i.min(j)] * p;
                }
                r
            })
        };
        let mut projected = solve(x.map(Quad::exact));
        // Two corrections recover the candidate's accuracy after cancellation;
        // their success is not assumed by the a posteriori certificate.
        for _ in 0..2 {
            let correction = solve(residual(&projected).map(|mut r| {
                r.u = 0.0;
                r
            }));
            for (p, correction) in projected.iter_mut().zip(correction) {
                *p = *p + correction;
                p.u = 0.0;
            }
        }
        let mut norm = 0.0;
        for r in residual(&projected) {
            let magnitude = upper(r.normalized().magnitude() + r.u);
            norm = upper(norm + upper(magnitude * magnitude));
        }
        // Exact raw G = RIDGE I + sum xx^T: ||G^-1||2 <= 1/RIDGE.
        // Quad interval residual includes both accumulation and arithmetic error.
        let error = upper(norm.sqrt() / RIDGE.next_down());
        for p in &mut projected {
            p.u = error;
        }
        projected
    }

    fn scaled_condition(&self) -> f64 {
        // D is an exact diagonal of the chosen f64 scale factors. Certify the
        // inverse of D G D by its residual; failure means unknown, not a gate.
        let d: [f64; FEATURES] = std::array::from_fn(|i| 1.0 / self.gram[i][i].center().sqrt());
        let a: Vec<Vec<Bound>> = (0..FEATURES)
            .map(|i| {
                (0..FEATURES)
                    .map(|j| {
                        self.gram[i.max(j)][i.min(j)] * Bound::exact(d[i]) * Bound::exact(d[j])
                    })
                    .collect()
            })
            .collect();
        let mut lower = vec![vec![Bound::default(); FEATURES]; FEATURES];
        for i in 0..FEATURES {
            for j in 0..=i {
                let mut v = a[i][j];
                v.u = 0.0;
                for k in 0..j {
                    v = v - lower[i][k] * lower[j][k];
                    v.u = 0.0;
                }
                lower[i][j] = if i == j { v.sqrt() } else { v / lower[j][j] };
                lower[i][j].u = 0.0;
            }
        }
        let mut inverse = vec![[Bound::default(); FEATURES]; FEATURES];
        for j in 0..FEATURES {
            let mut unit = [0.0; FEATURES];
            unit[j] = 1.0;
            let column = Self::solve(&lower, &unit);
            for i in 0..FEATURES {
                inverse[i][j] = column[i];
            }
        }
        let mut defect: f64 = 0.0;
        let mut matrix_norm: f64 = 0.0;
        let mut inverse_norm: f64 = 0.0;
        for i in 0..FEATURES {
            let mut erow = 0.0;
            let mut arow = 0.0;
            let mut zrow = 0.0;
            for j in 0..FEATURES {
                let mut e = Bound::exact(f64::from(i == j));
                for k in 0..FEATURES {
                    e = e - a[i][k] * inverse[k][j];
                }
                if !e.center().is_finite() || !e.u.is_finite() {
                    return f64::INFINITY;
                }
                erow = upper(erow + e.magnitude() + e.u);
                arow = upper(arow + a[i][j].magnitude() + a[i][j].u);
                zrow = upper(zrow + inverse[i][j].magnitude());
            }
            defect = defect.max(erow);
            matrix_norm = matrix_norm.max(arow);
            inverse_norm = inverse_norm.max(zrow);
        }
        if defect < 1.0 {
            upper(matrix_norm * inverse_norm / (1.0 - defect).next_down())
        } else {
            f64::INFINITY
        }
    }

    fn certified_projection(
        &self,
        lower: &[Vec<Bound>],
        x: &[f64; FEATURES],
    ) -> ([Bound; FEATURES], f64) {
        let mut projected = Self::solve(lower, x);
        let mut residual_norm = 0.0;
        let mut gram_norm: f64 = 0.0;
        for (i, &x) in x.iter().enumerate() {
            let mut residual = Bound::exact(x);
            let mut row_norm = 0.0;
            for (j, &p) in projected.iter().enumerate() {
                let g = self.gram[i.max(j)][i.min(j)];
                residual = residual - g * p;
                row_norm = upper(row_norm + g.magnitude() + g.u);
            }
            let magnitude = upper(residual.magnitude() + residual.u);
            residual_norm = upper(residual_norm + upper(magnitude * magnitude));
            gram_norm = gram_norm.max(row_norm);
        }
        for p in &mut projected {
            p.u = upper(residual_norm.sqrt() / RIDGE.next_down());
        }
        (projected, upper(gram_norm / RIDGE.next_down()))
    }
}

impl Reference {
    fn evaluate(
        &mut self,
        id: u64,
        x: &[f64; FEATURES],
        energy: [f32; 3],
        output: &[[f32; 3]; FORECAST_LEN],
        counts: &mut Counts,
    ) {
        let lower = self.factor();
        let (projected, condition) = self.certified_projection(&lower, x);
        self.condition_upper = condition;
        self.scaled_condition_upper = self.scaled_condition();
        counts.scaled_condition_upper = counts
            .scaled_condition_upper
            .max(self.scaled_condition_upper);
        counts.condition_upper = counts.condition_upper.max(condition);
        {
            println!(
                "reference observer={id} issue={} kappa2_upper={condition:e} scaled_kappa2_upper={:e}",
                self.issues, self.scaled_condition_upper
            );
        }
        let mut quad_projected = None;
        for (lead, out) in output.iter().enumerate() {
            for band in 0..3 {
                let mut correction = Bound::default();
                for (row, p) in self.response[lead].iter().zip(projected) {
                    correction = correction + row[band] * p;
                }
                let value = Bound::exact(energy[band] as f64) + correction;
                if !counts.check(
                    0,
                    out[band],
                    value.clamp(0.0, f64::INFINITY),
                    energy[band].abs() as f64 + correction.magnitude() + correction.u,
                ) {
                    counts.dd_inconclusive += 1;
                    counts.inconclusive[0] -= 1;
                    let projected = quad_projected.get_or_insert_with(|| self.quad_projection(x));
                    let mut correction = Quad::default();
                    for (row, &p) in self.quad_response[lead].iter().zip(projected.iter()) {
                        correction = correction + row[band] * p;
                    }
                    let value = (Quad::exact(energy[band] as f64) + correction).bound();
                    counts.check(
                        0,
                        out[band],
                        value.clamp(0.0, f64::INFINITY),
                        energy[band].abs() as f64 + correction.magnitude() + correction.u,
                    );
                }
            }
        }
    }
}

// Numeric JSON round-trips every f64 component (serde_json/float_roundtrip).
// Rows are flushed before certification so a failed certificate loses no input.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind")]
enum Snapshot {
    Matrix {
        observer: u64,
        issue: u64,
        gram: Vec<Vec<Bound>>,
        quad_gram: Vec<Vec<Quad>>,
        response: Vec<Vec<[Bound; 3]>>,
        quad_response: Vec<Vec<[Quad; 3]>>,
        x: Vec<f64>,
        energy: [f32; 3],
        output: Vec<[f32; 3]>,
    },
    Scalar {
        observer: u64,
        issue: u64,
        lead: usize,
        band: usize,
        output_kind: usize,
        sums: [Scalar; 2],
        output: f32,
    },
}

fn save_snapshot(snapshot: &Snapshot) {
    use std::io::Write;
    let Ok(directory) = std::env::var("IP_V7_SNAPSHOTS") else {
        return;
    };
    let issue = match snapshot {
        Snapshot::Matrix { issue, .. } | Snapshot::Scalar { issue, .. } => issue,
    };
    let path = std::path::Path::new(&directory).join(format!("checkpoint-{issue}.jsonl"));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(&mut file, snapshot).unwrap();
    writeln!(file).unwrap();
    file.sync_data().unwrap();
}

#[test]
#[ignore = "offline rejudgment of IP_V7_SNAPSHOT; no renderer or audio device"]
fn rejudge_precision_snapshot() {
    use std::io::BufRead;
    let file = std::fs::File::open(std::env::var("IP_V7_SNAPSHOT").unwrap()).unwrap();
    let mut counts = Counts::default();
    let mut identities = std::collections::BTreeSet::new();
    let mut checkpoint = None;
    for line in std::io::BufReader::new(file).lines() {
        let snapshot: Snapshot = serde_json::from_str(&line.unwrap()).unwrap();
        let (issue, key) = match &snapshot {
            Snapshot::Matrix {
                observer, issue, ..
            } => (*issue, (0, *observer, 0, 0)),
            Snapshot::Scalar {
                observer,
                issue,
                lead,
                band,
                output_kind,
                ..
            } => (*issue, (*output_kind, *observer, *lead, *band)),
        };
        assert_eq!(*checkpoint.get_or_insert(issue), issue, "mixed checkpoints");
        assert!(identities.insert(key), "duplicate checkpoint output");
        match snapshot {
            Snapshot::Matrix {
                observer,
                issue,
                gram,
                quad_gram,
                response,
                quad_response,
                x,
                energy,
                output,
            } => {
                let mut reference = Reference::new();
                reference.gram = gram;
                reference.quad_gram = quad_gram;
                reference.response = response
                    .into_iter()
                    .map(|row| row.try_into().unwrap())
                    .collect();
                reference.quad_response = quad_response
                    .into_iter()
                    .map(|row| row.try_into().unwrap())
                    .collect();
                reference.issues = issue;
                let x = x.try_into().unwrap();
                let output = output.try_into().unwrap();
                reference.evaluate(observer, &x, energy, &output, &mut counts);
            }
            Snapshot::Scalar {
                sums,
                output,
                output_kind,
                ..
            } => {
                counts.scalar(output_kind, output, sums, output_kind == 2);
            }
        }
    }
    println!("rejudgment {counts:?}");
    assert_eq!(
        std::array::from_fn::<_, 3, _>(|i| counts.passed[i]
            + counts.failed[i]
            + counts.inconclusive[i]),
        [1206, 1206, 603],
        "incomplete checkpoint snapshot"
    );
    counts.assert_passed();
}

#[derive(Default)]
struct Capture {
    histories: BTreeMap<u64, Reference>,
    learning: BTreeMap<u64, Vec<[[Scalar; 2]; 3]>>,
    learning_issues: BTreeMap<u64, u64>,
    counts: Counts,
    history_limit: usize,
    checkpoints: Vec<u64>,
}
thread_local! { static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) }; }
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(crate) fn next_id() -> u64 {
    NEXT.fetch_add(1, Ordering::Relaxed)
}
pub(crate) fn start() {
    start_selected(usize::MAX, Vec::new());
}

/// Long acquisitions select the shared history and a long-lived clone, and
/// explicitly query at their registered issue checkpoints. No raw trace is retained.
pub(crate) fn start_selected(history_limit: usize, checkpoints: Vec<u64>) {
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(Capture {
            history_limit,
            checkpoints,
            ..Capture::default()
        })
    });
}

impl Capture {
    fn history(&mut self, id: u64) -> Option<&mut Reference> {
        if self.histories.contains_key(&id) || self.histories.len() < self.history_limit {
            Some(self.histories.entry(id).or_insert_with(Reference::new))
        } else {
            None
        }
    }
}

pub(crate) fn counts() -> Counts {
    CAPTURE.with(|c| c.borrow().as_ref().unwrap().counts)
}

pub(crate) fn select_history(id: u64) {
    capture(|c| {
        assert!(c.history(id).is_some());
    });
}

pub(crate) fn checkpoint_due(id: u64) -> bool {
    CAPTURE.with(|c| {
        c.borrow().as_ref().is_some_and(|c| {
            !c.checkpoints.is_empty()
                && c.histories.get(&id).is_some_and(|r| {
                    c.checkpoints.contains(&r.issues) && r.last_forecast != Some(r.issues)
                })
        })
    })
}

pub(crate) fn checkpoint_result(id: u64) -> (u64, Option<u64>, Counts) {
    CAPTURE.with(|c| {
        let c = c.borrow();
        let r = &c.as_ref().unwrap().histories[&id];
        (r.issues, r.last_weights, r.checkpoint_counts)
    })
}
pub(crate) fn stop() {
    CAPTURE.with(|c| *c.borrow_mut() = None);
}
fn capture(f: impl FnOnce(&mut Capture)) {
    CAPTURE.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut() {
            f(c);
        }
    });
}

pub(crate) fn clone_history(parent: u64, child: u64) {
    capture(|c| {
        if c.histories.len() < c.history_limit
            && let Some(r) = c.histories.get(&parent).cloned()
        {
            c.histories.insert(child, r);
        }
    });
}
pub(crate) fn reset(id: u64) {
    capture(|c| {
        c.histories.remove(&id);
    });
}
pub(crate) fn issue(id: u64, x: &[f64; FEATURES]) {
    capture(|c| {
        let Some(r) = c.history(id) else {
            return;
        };
        for i in 0..FEATURES {
            for j in 0..=i {
                r.gram[i][j] = r.gram[i][j] + Bound::exact(x[i]) * Bound::exact(x[j]);
                r.quad_gram[i][j] = r.quad_gram[i][j] + Quad::exact(x[i]) * Quad::exact(x[j]);
            }
        }
        r.issues += 1;
    });
}
pub(crate) fn response(id: u64, lead: usize, x: &[f64; FEATURES], residual: [f64; 3]) {
    capture(|c| {
        if let Some(r) = c.histories.get_mut(&id) {
            for (row, &x) in r.quad_response[lead].iter_mut().zip(x) {
                for (v, residual) in row.iter_mut().zip(residual) {
                    *v = *v + Quad::exact(x) * Quad::exact(residual);
                }
            }
            for (row, &x) in r.response[lead].iter_mut().zip(x) {
                for (v, residual) in row.iter_mut().zip(residual) {
                    *v = *v + Bound::exact(x) * Bound::exact(residual);
                }
            }
        }
    });
}
pub(crate) fn comparison(id: u64, lead: usize, band: usize, operands: [f32; 3]) {
    capture(|c| {
        let Some(r) = c.history(id) else {
            return;
        };
        let sums = &mut r.comparison[lead][band];
        let [history, recurrence, observed] = operands;
        sums[0].add_product([history, recurrence], [history, recurrence]);
        sums[1].add_product([history, recurrence], [observed, recurrence]);
    });
}
pub(crate) fn forecast(
    id: u64,
    x: &[f64; FEATURES],
    energy: [f32; 3],
    output: &[[f32; 3]; FORECAST_LEN],
) {
    capture(|c| {
        let Some(r) = c.histories.get_mut(&id) else {
            return;
        };
        let due = if c.checkpoints.is_empty() {
            r.last_forecast.is_none_or(|last| r.issues >= last + 100)
        } else {
            c.checkpoints.contains(&r.issues) && r.last_forecast != Some(r.issues)
        };
        if !due {
            return;
        }
        r.last_forecast = Some(r.issues);
        r.checkpoint_start = c.counts;
        if !c.checkpoints.is_empty() {
            save_snapshot(&Snapshot::Matrix {
                observer: id,
                issue: r.issues,
                gram: r.gram.clone(),
                quad_gram: r.quad_gram.clone(),
                response: r.response.iter().map(|row| row.to_vec()).collect(),
                quad_response: r.quad_response.iter().map(|row| row.to_vec()).collect(),
                x: x.to_vec(),
                energy,
                output: output.to_vec(),
            });
        }
        r.evaluate(id, x, energy, output, &mut c.counts);
    });
}
pub(crate) fn weights(id: u64, output: &[[f32; 3]; FORECAST_LEN]) {
    capture(|c| {
        let Some(r) = c.histories.get_mut(&id) else {
            return;
        };
        let due = if c.checkpoints.is_empty() {
            r.last_weights.is_none_or(|last| r.issues >= last + 100)
        } else {
            c.checkpoints.contains(&r.issues) && r.last_weights != Some(r.issues)
        };
        if !due {
            return;
        }
        r.last_weights = Some(r.issues);
        if !c.checkpoints.is_empty() {
            println!("reference comparison observer={id} issue={}", r.issues);
        }
        for (lead, (row, out)) in r.comparison.iter().zip(output).enumerate() {
            for (band, (sums, &out)) in row.iter().zip(out).enumerate() {
                if !c.checkpoints.is_empty() {
                    save_snapshot(&Snapshot::Scalar {
                        observer: id,
                        issue: r.issues,
                        lead,
                        band,
                        output_kind: 1,
                        sums: *sums,
                        output: out,
                    });
                }
                c.counts.scalar(1, out, *sums, false);
            }
        }
        r.checkpoint_counts = Counts {
            passed: std::array::from_fn(|i| c.counts.passed[i] - r.checkpoint_start.passed[i]),
            failed: std::array::from_fn(|i| c.counts.failed[i] - r.checkpoint_start.failed[i]),
            inconclusive: std::array::from_fn(|i| {
                c.counts.inconclusive[i] - r.checkpoint_start.inconclusive[i]
            }),
            dd_inconclusive: c.counts.dd_inconclusive - r.checkpoint_start.dd_inconclusive,
            uncertified: std::array::from_fn(|i| {
                c.counts.uncertified[i] - r.checkpoint_start.uncertified[i]
            }),
            condition_upper: r.condition_upper,
            scaled_condition_upper: r.scaled_condition_upper,
        };
    });
}
pub(crate) fn learning_reset(id: u64) {
    capture(|c| {
        c.learning.remove(&id);
        c.learning_issues.remove(&id);
    });
}
pub(crate) fn learning(
    id: u64,
    lead: usize,
    band: usize,
    candidates: [f32; 2],
    observed: Option<f32>,
    output: Option<f32>,
) {
    capture(|c| {
        let issues = c.learning_issues.entry(id).or_default();
        if observed.is_none() && lead == 0 && band == 0 {
            *issues += 1;
            if c.checkpoints.contains(issues) {
                println!("reference learning observer={id} issue={issues}");
            }
        }
        let due = c.checkpoints.is_empty() || c.checkpoints.contains(issues);
        let rows = c
            .learning
            .entry(id)
            .or_insert_with(|| vec![[[Scalar::default(); 2]; 3]; FORECAST_LEN]);
        let sums = &mut rows[lead][band];
        let i = usize::from(observed.is_some());
        let right = [observed.unwrap_or(candidates[0]), candidates[1]];
        sums[i].add_product(candidates, right);
        if let Some(out) = output.filter(|_| due) {
            if !c.checkpoints.is_empty() {
                save_snapshot(&Snapshot::Scalar {
                    observer: id,
                    issue: *issues,
                    lead,
                    band,
                    output_kind: 2,
                    sums: *sums,
                    output: out,
                });
            }
            c.counts.scalar(2, out, *sums, true);
        }
    });
}

#[test]
fn exact_f32_product_ranges_preserve_scalar_weight_acceptance() {
    use crate::core::compensated::Sum;
    let mut counts = Counts::default();
    // Exercise the 53-bit boundary, its wider fallback, signs, zero, and
    // both ends of the finite f32 range against independent raw statistics.
    for [candidate, baseline, observed] in [
        [1.0_f32.next_up(), 0.25, 0.5],
        [1.0_f32.next_up(), 0.125, 0.5],
        [1.0_f32.next_up(), 0.0625, 0.5],
        [1.0_f32.next_up(), -0.25, 0.125],
        [1.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MIN_POSITIVE / 2.0,
        ],
        [f32::MAX, f32::MAX / 4.0, f32::MAX / 2.0],
    ] {
        let mut raw = [Scalar::default(); 2];
        let mut production = [Sum::ZERO; 2];
        for _ in 0..64 {
            let delta = [candidate, baseline];
            production[0].add_product(delta, delta);
            production[1].add_product(delta, [observed, baseline]);
            raw[0].add_product(delta, delta);
            raw[1].add_product(delta, [observed, baseline]);
            let square = production[0].total();
            let (weight, complement) = if square > 0.0 {
                (
                    (production[1].total() / square).clamp(0.0, 1.0) as f32,
                    (production[0].difference(production[1]) / square).clamp(0.0, 1.0) as f32,
                )
            } else {
                (0.0, 1.0)
            };
            counts.scalar(1, weight, raw, false);
            counts.scalar(2, complement, raw, true);
        }
    }
    println!("numerical exact product ranges {counts:?}");
    assert_eq!(counts.passed, [0, 512, 512]);
    assert_eq!(counts.failed, [0; 3]);
    assert_eq!(counts.inconclusive, [0; 3]);
}

#[test]
fn raw_scalar_products_distinguish_rounded_products_at_zero_complement() {
    let mut raw = [Scalar::default(); 2];
    let mut rounded = [crate::core::compensated::Sum::ZERO; 2];
    let unit = 2.0_f32.powi(-23);
    for (baseline, observed) in [
        (2.0_f32.powi(-30), 1.0 + unit),
        (2.0_f32.powi(-31), 1.0 - unit),
    ] {
        raw[0].add_product([1.0, baseline], [1.0, baseline]);
        raw[1].add_product([1.0, baseline], [observed, baseline]);
        let delta = 1.0 - baseline as f64;
        rounded[0].add(delta * delta);
        rounded[1].add(delta * (observed as f64 - baseline as f64));
    }
    // Independent algebra: u * (2^-30 - 2^-31) = 2^-54.
    assert_eq!((raw[0].dd - raw[1].dd).center(), 2.0_f64.powi(-54));
    assert_eq!(
        (raw[0].quad - raw[1].quad).bound().center(),
        2.0_f64.powi(-54)
    );
    let old = (rounded[0].difference(rounded[1]) / rounded[0].total()).clamp(0.0, 1.0) as f32;
    assert_eq!(old, 0.0);
    let mut counts = Counts::default();
    counts.scalar(2, old, raw, true);
    assert_eq!(counts.failed, [0, 0, 1]);
    assert_eq!(counts.inconclusive, [0; 3]);
    // Differences themselves must also be formed before f64 discards low bits.
    let mut tiny = Scalar::default();
    tiny.add_product([1.0, 2.0_f32.powi(-80)], [1.0, 0.0]);
    assert_eq!(tiny.dd.v.1, -2.0_f64.powi(-80));
    assert_eq!(tiny.quad.bound().v.1, -2.0_f64.powi(-80));
}

#[test]
fn reference_detects_sub_ulp_cancellation_and_interval_boundaries() {
    let a = Bound::exact(1e20) + Bound::exact(1.0) - Bound::exact(1e20);
    assert_eq!(a.center(), 1.0);
    assert!(a.u < 1e-10);
    let mut counts = Counts::default();
    counts.check(0, 1.0, Bound::exact(1.0), 0.0);
    counts.check(0, 1.01, Bound::exact(1.0), 0.0);
    counts.check(
        0,
        1.0,
        Bound {
            v: Dd(1.0, 0.0),
            u: 1e-6,
        },
        0.0,
    );
    let mut square = Scalar::default();
    square.add(1.0);
    let mut cross = Scalar::default();
    cross.add(0.1);
    cross.add(-0.1);
    counts.scalar(1, 0.0, [square, cross], false);
    assert_eq!(counts.dd_inconclusive, 1);
    assert_eq!(
        (counts.passed[1], counts.failed[1], counts.inconclusive[1]),
        (1, 0, 0)
    );
    assert_eq!(
        (counts.passed[0], counts.failed[0], counts.inconclusive[0]),
        (1, 1, 1)
    );
}

#[test]
fn matrix_inconclusive_is_recomputed_from_raw_quad_statistics() {
    start();
    let mut x = [0.0; FEATURES];
    x[0] = 1.0;
    issue(0, &x);
    response(0, 0, &x, [0.1; 3]);
    capture(|c| {
        // Simulate a DD certificate too wide to decide; the independent raw
        // quad accumulation remains intact and must supply the decision.
        c.histories.get_mut(&0).unwrap().response[0][0][0].u = 0.1;
    });
    let mut output = [[0.0; 3]; FORECAST_LEN];
    output[0] = [(0.1 / 1.01) as f32; 3];
    forecast(0, &x, [0.0; 3], &output);
    let result = counts();
    stop();
    assert_eq!(result.dd_inconclusive, 1);
    assert_eq!(result.passed[0], (3 * FORECAST_LEN) as u64);
    assert_eq!(result.failed, [0; 3]);
    assert_eq!(result.inconclusive, [0; 3]);
}

#[test]
fn ridge_residual_certifies_collinear_million_issue_scale() {
    let mut reference = Reference::new();
    let mut x = [1.0; FEATURES];
    x[1..28].fill(20_000.0);
    // A repeated, exactly represented raw row has an analytic Sherman-Morrison
    // solution. Large energy and constant/coverage collinearity reproduce V7's scale.
    for i in 0..FEATURES {
        for j in 0..=i {
            let term = Quad::exact(1_000_000.0) * Quad::exact(x[i]) * Quad::exact(x[j]);
            reference.quad_gram[i][j] = reference.quad_gram[i][j] + term;
            reference.gram[i][j] = reference.quad_gram[i][j].bound();
        }
    }
    // A conservative, valid input enclosure at the constant-feature entry.
    // It is far below the ridge, but factor interval dependency amplifies it
    // by the square of the large feature. The residual uses its actual action
    // on the candidate instead of amplifying it through every pivot.
    reference.quad_gram[0][0].u = 1e-10;
    let mut old_lower = vec![vec![Quad::default(); FEATURES]; FEATURES];
    let mut old_certified = true;
    'factor: for i in 0..FEATURES {
        for j in 0..=i {
            let mut value = reference.quad_gram[i][j];
            for k in 0..j {
                value = value - old_lower[i][k] * old_lower[j][k];
            }
            old_lower[i][j] = if i == j {
                value.sqrt()
            } else {
                value / old_lower[j][j]
            };
            if !old_lower[i][j].u.is_finite() {
                old_certified = false;
                break 'factor;
            }
        }
    }
    assert!(!old_certified, "old interval pivot unexpectedly certified");
    let projected = reference.quad_projection(&x);
    let mut result = Quad::default();
    for (p, x) in projected.into_iter().zip(x) {
        result = result + p * Quad::exact(x);
    }
    let norm: f64 = x.iter().map(|x| x * x).sum();
    let exact = norm / (RIDGE + 1_000_000.0 * norm);
    let mut counts = Counts::default();
    counts.check(0, exact as f32, result.bound(), exact);
    counts.check(0, (exact * 1.001) as f32, result.bound(), exact);
    assert_eq!(counts.passed[0], 1);
    assert_eq!(counts.failed[0], 1);
    assert_eq!(counts.inconclusive[0], 0);
}

#[test]
fn uncertifiable_reference_counts_and_continues_but_nonfinite_product_fails() {
    let mut r = Reference::new();
    r.gram[0][0] = Bound::exact(-1.0);
    r.quad_gram[0][0] = Quad::exact(-1.0);
    let mut counts = Counts::default();
    let mut x = [0.0; FEATURES];
    x[0] = 1.0;
    r.evaluate(0, &x, [1.0; 3], &[[1.0; 3]; FORECAST_LEN], &mut counts);
    assert_eq!(counts.inconclusive[0], (3 * FORECAST_LEN) as u64);
    assert_eq!(counts.passed[0], 0);
    assert!(counts.uncertified[0] > 0);
    // The following checkpoint still runs and is decidable.
    Reference::new().evaluate(0, &x, [1.0; 3], &[[1.0; 3]; FORECAST_LEN], &mut counts);
    assert_eq!(counts.passed[0], (3 * FORECAST_LEN) as u64);
    assert!(
        std::panic::catch_unwind(|| {
            Counts::default().check(0, f32::NAN, Bound::exact(0.0), 0.0);
        })
        .is_err()
    );
}

#[test]
fn snapshot_roundtrip_preserves_expansions_and_rejudgment() {
    let mut reference = Reference::new();
    reference.gram[1][0] = Bound {
        v: Dd(1.0, 2.0_f64.powi(-80)),
        u: 1e-31,
    };
    reference.quad_response[7][5][2] = Quad {
        parts: [1e20, -0.125, 2.0_f64.powi(-100), -2.0_f64.powi(-180)],
        u: 1e-80,
    };
    let snapshot = Snapshot::Matrix {
        observer: 3,
        issue: 1000000,
        gram: reference.gram.clone(),
        quad_gram: reference.quad_gram.clone(),
        response: reference.response.iter().map(|row| row.to_vec()).collect(),
        quad_response: reference
            .quad_response
            .iter()
            .map(|row| row.to_vec())
            .collect(),
        x: vec![0.0; FEATURES],
        energy: [1.0; 3],
        output: vec![[1.0; 3]; FORECAST_LEN],
    };
    let bytes = serde_json::to_vec(&snapshot).unwrap();
    let decoded: Snapshot = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
    let Snapshot::Matrix {
        gram,
        quad_response,
        ..
    } = decoded
    else {
        unreachable!()
    };
    assert_eq!(gram[1][0].v.1.to_bits(), reference.gram[1][0].v.1.to_bits());
    for (a, b) in quad_response[7][5][2]
        .parts
        .into_iter()
        .zip(reference.quad_response[7][5][2].parts)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    println!("snapshot zero-dominated matrix bytes={}", bytes.len());
}
