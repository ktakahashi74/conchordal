/// Per-voice life-history accumulator and death record.
///
/// Mirrors the paper's E3DeathRecord / E3LifeState with adaptations
/// for the real-time instrument context.

#[derive(Debug, Clone)]
pub struct LifeRecord {
    pub voice_id: u64,
    pub group_id: u64,
    pub birth_frame: u64,
    pub death_frame: u64,
    pub lifetime_ticks: u32,
    pub c_level_birth: f32,
    pub c_level_firstk_mean: f32,
    pub c_level_lifetime_mean: f32,
    pub c_level_lifetime_std: f32,
    pub c_level_attack_mean: f32,
    pub attack_count: u32,
    pub plv_at_death: Option<f32>,
    pub generation: u32,
}

#[derive(Debug, Clone)]
pub struct LifeAccumulator {
    first_k: u32,
    birth_frame: u64,
    c_level_birth: f32,
    // first-k tracking
    firstk_sum: f32,
    firstk_count: u32,
    // lifetime tracking
    lifetime_sum: f32,
    lifetime_sum_sq: f32,
    lifetime_count: u32,
    // attack-time tracking
    attack_sum: f32,
    attack_count: u32,
}

impl LifeAccumulator {
    pub fn new(birth_frame: u64, first_k: u32, c_level_birth: f32) -> Self {
        Self {
            first_k,
            birth_frame,
            c_level_birth,
            firstk_sum: 0.0,
            firstk_count: 0,
            lifetime_sum: 0.0,
            lifetime_sum_sq: 0.0,
            lifetime_count: 0,
            attack_sum: 0.0,
            attack_count: 0,
        }
    }

    pub fn accumulate_tick(&mut self, c_level: f32) {
        self.lifetime_sum += c_level;
        self.lifetime_sum_sq += c_level * c_level;
        self.lifetime_count += 1;
        if self.firstk_count < self.first_k {
            self.firstk_sum += c_level;
            self.firstk_count += 1;
        }
    }

    pub fn accumulate_attack(&mut self, c_level: f32) {
        self.attack_sum += c_level;
        self.attack_count += 1;
    }

    pub fn finalize(
        &self,
        voice_id: u64,
        group_id: u64,
        death_frame: u64,
        plv: Option<f32>,
        generation: u32,
    ) -> LifeRecord {
        let n = self.lifetime_count;
        let (mean, std) = if n > 0 {
            let mean = self.lifetime_sum / n as f32;
            let var = (self.lifetime_sum_sq / n as f32 - mean * mean).max(0.0);
            (mean, var.sqrt())
        } else {
            (0.0, 0.0)
        };
        let firstk_mean = if self.firstk_count > 0 {
            self.firstk_sum / self.firstk_count as f32
        } else {
            0.0
        };
        let attack_mean = if self.attack_count > 0 {
            self.attack_sum / self.attack_count as f32
        } else {
            0.0
        };
        LifeRecord {
            voice_id,
            group_id,
            birth_frame: self.birth_frame,
            death_frame,
            lifetime_ticks: n,
            c_level_birth: self.c_level_birth,
            c_level_firstk_mean: firstk_mean,
            c_level_lifetime_mean: mean,
            c_level_lifetime_std: std,
            c_level_attack_mean: attack_mean,
            attack_count: self.attack_count,
            plv_at_death: plv,
            generation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_computes_mean_and_std() {
        let mut acc = LifeAccumulator::new(100, 3, 0.5);
        // Push values: 1.0, 2.0, 3.0, 4.0, 5.0
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            acc.accumulate_tick(v);
        }
        acc.accumulate_attack(2.0);
        acc.accumulate_attack(4.0);

        let rec = acc.finalize(42, 7, 200, Some(0.9), 0);
        assert_eq!(rec.voice_id, 42);
        assert_eq!(rec.group_id, 7);
        assert_eq!(rec.birth_frame, 100);
        assert_eq!(rec.death_frame, 200);
        assert_eq!(rec.lifetime_ticks, 5);
        assert!((rec.c_level_birth - 0.5).abs() < 1e-6);
        // first-k mean: (1+2+3)/3 = 2.0
        assert!((rec.c_level_firstk_mean - 2.0).abs() < 1e-6);
        // lifetime mean: (1+2+3+4+5)/5 = 3.0
        assert!((rec.c_level_lifetime_mean - 3.0).abs() < 1e-6);
        // lifetime std: sqrt(var) where var = E[x^2] - E[x]^2
        //   E[x^2] = (1+4+9+16+25)/5 = 11, E[x]^2 = 9, var = 2
        let expected_std = 2.0f32.sqrt();
        assert!((rec.c_level_lifetime_std - expected_std).abs() < 1e-5);
        // attack mean: (2+4)/2 = 3.0
        assert!((rec.c_level_attack_mean - 3.0).abs() < 1e-6);
        assert_eq!(rec.attack_count, 2);
        assert_eq!(rec.plv_at_death, Some(0.9));
    }

    #[test]
    fn finalize_empty() {
        let acc = LifeAccumulator::new(0, 5, 0.0);
        let rec = acc.finalize(1, 1, 0, None, 0);
        assert_eq!(rec.lifetime_ticks, 0);
        assert_eq!(rec.c_level_lifetime_mean, 0.0);
        assert_eq!(rec.c_level_lifetime_std, 0.0);
        assert_eq!(rec.c_level_firstk_mean, 0.0);
        assert_eq!(rec.c_level_attack_mean, 0.0);
        assert_eq!(rec.plv_at_death, None);
    }
}
