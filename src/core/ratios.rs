// src/core/ratios.rs

/// Ratio candidate (m:n with normalized PLV)
#[derive(Debug, Clone, PartialEq)]
pub struct Ratio {
    pub m: u32,
    pub n: u32,
    pub plv: f32,
}

/// Simple gcd (Euclidean algorithm)
pub const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let tmp = b;
        b = a % b;
        a = tmp;
    }
    a
}

/// Generate ratio set up to constraints:
/// - 1 <= m <= m_max
/// - 1 <= n <= n_max
/// - m >= n (normalize orientation)
/// - gcd(m,n) == 1 (irreducible)
/// - m+n <= sum_max
/// - PLV normalized: PLV(1:1) = 1.0, formula (2/(m+n))^alpha
pub fn generate_ratios(m_max: u32, n_max: u32, sum_max: u32, alpha: f32) -> Vec<Ratio> {
    let mut ratios = Vec::new();

    for m in 1..=m_max {
        for n in 1..=n_max {
            if m < n {
                continue; // enforce m >= n
            }
            if gcd(m, n) != 1 {
                continue; // skip reducible
            }
            if m + n > sum_max {
                continue;
            }

            // normalized PLV: 1:1 = 1.0
            let norm = (2.0f32 / (m + n) as f32).powf(alpha);

            ratios.push(Ratio { m, n, plv: norm });
        }
    }

    // sort by descending PLV, then by fraction value
    ratios.sort_by(|a, b| {
        b.plv
            .partial_cmp(&a.plv)
            .unwrap()
            .then_with(|| (a.m * b.n).cmp(&(b.m * a.n)))
    });

    ratios
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ratios_basic() {
        let rs = generate_ratios(9, 9, 20, 1.0);

        // 1:1 must be included
        assert!(rs.iter().any(|x| x.m == 1 && x.n == 1));

        // 2:3 must not exist (m<n discarded)
        assert!(!rs.iter().any(|x| x.m == 2 && x.n == 3));

        // Check that gcd filtering works (2:2 should not be present)
        assert!(!rs.iter().any(|x| x.m == 2 && x.n == 2));

        // Check PLV normalization: 1:1 should be 1.0
        let one_one = rs.iter().find(|x| x.m == 1 && x.n == 1).unwrap();
        assert!((one_one.plv - 1.0).abs() < 1e-6);

        // Spot check: 2:1 should be ~0.67
        let two_one = rs.iter().find(|x| x.m == 2 && x.n == 1).unwrap();
        assert!((two_one.plv - 0.67).abs() < 0.02);

        // Count of ratios under these limits
        assert_eq!(rs.len(), 28);
    }

    #[test]
    fn test_alpha_effect() {
        let rs1 = generate_ratios(9, 9, 20, 1.0);
        let rs2 = generate_ratios(9, 9, 20, 2.0);

        let r31_a1 = rs1.iter().find(|x| x.m == 3 && x.n == 1).unwrap().plv;
        let r31_a2 = rs2.iter().find(|x| x.m == 3 && x.n == 1).unwrap().plv;

        assert!(r31_a2 < r31_a1); // larger alpha => stronger decay
    }
}
