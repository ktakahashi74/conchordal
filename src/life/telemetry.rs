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
    pub c_level_firstk_mean: f32,
    pub plv_at_death: Option<f32>,
    pub generation: u32,
}

#[derive(Debug, Clone)]
pub struct LifeAccumulator {
    first_k: u32,
    birth_frame: u64,
    // first-k tracking
    firstk_sum: f32,
    firstk_count: u32,
    lifetime_count: u32,
}

impl LifeAccumulator {
    pub fn new(birth_frame: u64, first_k: u32) -> Self {
        Self {
            first_k,
            birth_frame,
            firstk_sum: 0.0,
            firstk_count: 0,
            lifetime_count: 0,
        }
    }

    pub fn accumulate_tick(&mut self, c_level: f32) {
        self.lifetime_count += 1;
        if self.firstk_count < self.first_k {
            self.firstk_sum += c_level;
            self.firstk_count += 1;
        }
    }

    pub fn finalize(
        &self,
        voice_id: u64,
        group_id: u64,
        death_frame: u64,
        plv: Option<f32>,
        generation: u32,
    ) -> LifeRecord {
        let firstk_mean = if self.firstk_count > 0 {
            self.firstk_sum / self.firstk_count as f32
        } else {
            0.0
        };
        LifeRecord {
            voice_id,
            group_id,
            birth_frame: self.birth_frame,
            death_frame,
            lifetime_ticks: self.lifetime_count,
            c_level_firstk_mean: firstk_mean,
            plv_at_death: plv,
            generation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_computes_firstk_mean() {
        let mut acc = LifeAccumulator::new(100, 3);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            acc.accumulate_tick(v);
        }

        let rec = acc.finalize(42, 7, 200, Some(0.9), 0);
        assert_eq!(rec.voice_id, 42);
        assert_eq!(rec.group_id, 7);
        assert_eq!(rec.birth_frame, 100);
        assert_eq!(rec.death_frame, 200);
        assert_eq!(rec.lifetime_ticks, 5);
        // first-k mean: (1+2+3)/3 = 2.0
        assert!((rec.c_level_firstk_mean - 2.0).abs() < 1e-6);
        assert_eq!(rec.plv_at_death, Some(0.9));
    }

    #[test]
    fn finalize_empty() {
        let acc = LifeAccumulator::new(0, 5);
        let rec = acc.finalize(1, 1, 0, None, 0);
        assert_eq!(rec.lifetime_ticks, 0);
        assert_eq!(rec.c_level_firstk_mean, 0.0);
        assert_eq!(rec.plv_at_death, None);
    }
}
