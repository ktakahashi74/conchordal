//! Test-only independent double-double reference with propagated absolute bounds.
//! Hooks receive raw operands, never production sufficient statistics or factors.

use super::{FEATURES, FORECAST_LEN, RIDGE};
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

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Copy, Debug, Default)]
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
        assert!(
            x > 0.0 && self.center() > self.u,
            "uncertified reference pivot"
        );
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
        assert!(denominator > 0.0, "uncertified reference denominator");
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

// Four-component expansion, retained only for scalar inconclusive fallback.
// Cascaded TwoSum and FMA products expose every discarded term explicitly.
#[derive(Clone, Copy, Default)]
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
        self.parts.iter().map(|x| x.abs()).sum::<f64>().next_up()
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
    fn sqrt(self) -> Self {
        let bound = self.bound();
        let lower = (bound.center().next_down() - bound.u)
            .next_down()
            .sqrt()
            .next_down();
        assert!(lower > 0.0, "uncertified quad reference pivot");
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
        for value in b.parts {
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
        for a in self.parts {
            for b in b.parts {
                let product = a * b;
                result.add_term(product);
                result.add_term(a.mul_add(b, -product));
            }
        }
        let propagated = self.u * b.magnitude() + b.u * self.magnitude() + self.u * b.u;
        if propagated != 0.0 {
            result.u = (result.u + propagated).next_up();
        }
        result
    }
}
impl Div for Quad {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let mut result = Self::default();
        let mut residual = self;
        for _ in 0..5 {
            let quotient = residual.parts[0] / b.parts[0];
            result.add_term(quotient);
            residual = residual - b * Self::exact(quotient);
        }
        let denominator = b.bound().center().abs() - b.u;
        assert!(denominator > 0.0);
        let remainder = residual.magnitude() + residual.u;
        if remainder != 0.0 {
            result.u = (result.u + remainder / denominator).next_up();
        }
        result
    }
}

#[derive(Clone, Copy, Default)]
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
    pub dd_inconclusive: u64,
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
        if difference + uncertainty <= low {
            self.passed[kind] += 1;
            true
        } else if difference - uncertainty > high {
            self.failed[kind] += 1;
            if self.failed[kind] <= 4 {
                eprintln!(
                    "reference FAIL kind={kind} prod={prod:e} ref={reference:?} d={difference:e} tol={low:e}..{high:e} S={scale:e}"
                );
            }
            true
        } else {
            self.inconclusive[kind] += 1;
            if self.inconclusive[kind] <= 3 {
                eprintln!(
                    "reference INCONCLUSIVE kind={kind} prod={prod:e} ref={reference:?} d={difference:e} tol={low:e}..{high:e} S={scale:e}"
                );
            }
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
        let mut lower = vec![vec![Quad::default(); FEATURES]; FEATURES];
        for i in 0..FEATURES {
            for j in 0..=i {
                let mut value = self.quad_gram[i][j];
                for k in 0..j {
                    value = value - lower[i][k] * lower[j][k];
                }
                lower[i][j] = if i == j {
                    value.sqrt()
                } else {
                    value / lower[j][j]
                };
            }
        }
        let mut result = x.map(Quad::exact);
        for i in 0..FEATURES {
            for j in 0..i {
                result[i] = result[i] - lower[i][j] * result[j];
            }
            result[i] = result[i] / lower[i][i];
        }
        for i in (0..FEATURES).rev() {
            for j in i + 1..FEATURES {
                result[i] = result[i] - lower[j][i] * result[j];
            }
            result[i] = result[i] / lower[i][i];
        }
        result
    }

    fn certified_projection(
        &self,
        lower: &[Vec<Bound>],
        x: &[f64; FEATURES],
    ) -> ([Bound; FEATURES], f64) {
        let mut inverse = vec![[Bound::default(); FEATURES]; FEATURES];
        for i in 0..FEATURES {
            let mut unit = [0.0; FEATURES];
            unit[i] = 1.0;
            let column = Self::solve(lower, &unit);
            for j in 0..FEATURES {
                inverse[j][i] = column[j];
            }
        }
        let mut defect_norm: f64 = 0.0;
        let mut inverse_norm: f64 = 0.0;
        let mut gram_norm: f64 = 0.0;
        for i in 0..FEATURES {
            let mut defect_row = 0.0;
            let mut inverse_row = 0.0;
            let mut gram_row = 0.0;
            for j in 0..FEATURES {
                let mut defect = Bound::exact(f64::from(i == j));
                for (k, row) in inverse.iter().enumerate() {
                    defect = defect - self.gram[i.max(k)][i.min(k)] * row[j];
                }
                defect_row = (defect_row + defect.magnitude() + defect.u).next_up();
                inverse_row = (inverse_row + inverse[i][j].magnitude()).next_up();
                let g = self.gram[i.max(j)][i.min(j)];
                gram_row = (gram_row + g.magnitude() + g.u).next_up();
            }
            defect_norm = defect_norm.max(defect_row);
            inverse_norm = inverse_norm.max(inverse_row);
            gram_norm = gram_norm.max(gram_row);
        }
        // E = I-GZ: ||E||inf < 1 certifies invertibility and
        // ||G^-1||inf <= ||Z||inf/(1-||E||inf), including accumulated G error.
        assert!(defect_norm < 1.0, "reference inverse uncertified");
        let inverse_bound = (inverse_norm / (1.0 - defect_norm).next_down()).next_up();
        let mut projected = Self::solve(lower, x);
        let mut residual_norm: f64 = 0.0;
        for (i, &x) in x.iter().enumerate() {
            let mut residual = Bound::exact(x);
            for (j, &p) in projected.iter().enumerate() {
                residual = residual - self.gram[i.max(j)][i.min(j)] * p;
            }
            residual_norm = residual_norm.max((residual.magnitude() + residual.u).next_up());
        }
        for p in &mut projected {
            p.u = (inverse_bound * residual_norm).next_up();
        }
        // Symmetric G: the infinity-norm condition product bounds kappa_2.
        (projected, (gram_norm * inverse_bound).next_up())
    }
}

#[derive(Default)]
struct Capture {
    histories: BTreeMap<u64, Reference>,
    learning: BTreeMap<u64, Vec<[[Scalar; 2]; 3]>>,
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
        let lower = r.factor();
        let (projected, condition) = r.certified_projection(&lower, x);
        c.counts.condition_upper = c.counts.condition_upper.max(condition);
        if !c.checkpoints.is_empty() {
            println!(
                "reference observer={id} issue={} kappa2_upper={condition:e}",
                r.issues
            );
        }
        let mut quad_projected = None;
        for (lead, out) in output.iter().enumerate() {
            for band in 0..3 {
                let mut correction = Bound::default();
                for (row, p) in r.response[lead].iter().zip(projected) {
                    correction = correction + row[band] * p;
                }
                let value = Bound::exact(energy[band] as f64) + correction;
                if !c.counts.check(
                    0,
                    out[band],
                    value.clamp(0.0, f64::INFINITY),
                    energy[band].abs() as f64 + correction.magnitude() + correction.u,
                ) {
                    c.counts.dd_inconclusive += 1;
                    c.counts.inconclusive[0] -= 1;
                    let projected = quad_projected.get_or_insert_with(|| r.quad_projection(x));
                    let mut correction = Quad::default();
                    for (row, &p) in r.quad_response[lead].iter().zip(projected.iter()) {
                        correction = correction + row[band] * p;
                    }
                    let value = (Quad::exact(energy[band] as f64) + correction).bound();
                    c.counts.check(
                        0,
                        out[band],
                        value.clamp(0.0, f64::INFINITY),
                        energy[band].abs() as f64 + correction.magnitude() + correction.u,
                    );
                }
            }
        }
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
        for (row, out) in r.comparison.iter().zip(output) {
            for (sums, &out) in row.iter().zip(out) {
                c.counts.scalar(1, out, *sums, false);
            }
        }
    });
}
pub(crate) fn learning_reset(id: u64) {
    capture(|c| {
        c.learning.remove(&id);
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
        let rows = c
            .learning
            .entry(id)
            .or_insert_with(|| vec![[[Scalar::default(); 2]; 3]; FORECAST_LEN]);
        let sums = &mut rows[lead][band];
        let i = usize::from(observed.is_some());
        let right = [observed.unwrap_or(candidates[0]), candidates[1]];
        sums[i].add_product(candidates, right);
        if let Some(out) = output {
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
