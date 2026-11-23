use super::*;

impl ScheduleRenderer {
    pub(crate) fn phase3_enabled(&self) -> bool {
        self.prototype.enabled()
    }

    #[inline]
    pub(super) fn retire_prototype_sources(&mut self, now: Tick) {
        if self.phase3_enabled() {
            self.retire_closed_sources(now);
        }
    }

    #[inline]
    pub(super) fn enable_prototype_tone(&self, tone: &mut Tone) {
        if self.phase3_enabled() {
            tone.enable_phase3();
        }
    }

    pub(super) fn retire_closed_sources(&mut self, now: Tick) {
        #[cfg(test)]
        if self.sample_major_oracle {
            self.retire_closed_sources_oracle(now);
            return;
        }
        use std::ops::Bound::{Excluded, Unbounded};
        let mut after = None;
        while let Some((key, generation)) = self
            .tones
            .range((after.map_or(Unbounded, Excluded), Unbounded))
            .next()
            .map(|(k, _)| (*k, k.generation))
        {
            after = Some(key);
            let from = ToneKey {
                source_id: key.source_id,
                tone_id: 0,
                generation: 0,
            };
            let through = ToneKey {
                source_id: key.source_id,
                tone_id: u64::MAX,
                generation: u32::MAX,
            };
            // Generations may interleave in ToneKey order; evaluate each group once.
            if self
                .tones
                .range(from..key)
                .any(|(k, _)| k.generation == generation)
            {
                continue;
            }
            let residual = self
                .tones
                .range(from..=through)
                .filter(|(k, _)| k.generation == generation)
                .filter_map(|(_, rt)| rt.tone.residual_bound(now))
                .sum::<f64>();
            if residual <= f64::from(crate::life::voice::Voice::AMP_EPS) {
                for (k, rt) in self.tones.range_mut(from..=through) {
                    if k.generation == generation && rt.tone.residual_bound(now).is_some() {
                        rt.tone.retire_phase3();
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn retire_closed_sources_oracle(&mut self, now: Tick) {
        use std::ops::Bound::{Excluded, Unbounded};
        let mut after = None;
        while let Some((key, generation)) = self
            .tones
            .range((after.map_or(Unbounded, Excluded), Unbounded))
            .next()
            .map(|(k, _)| (*k, k.generation))
        {
            after = Some(key);
            let from = ToneKey {
                source_id: key.source_id,
                tone_id: 0,
                generation: 0,
            };
            let through = ToneKey {
                source_id: key.source_id,
                tone_id: u64::MAX,
                generation: u32::MAX,
            };
            let residual = self
                .tones
                .range(from..=through)
                .filter(|(k, _)| k.generation == generation)
                .filter_map(|(_, rt)| rt.tone.residual_bound(now))
                .sum::<f64>();
            if residual <= f64::from(crate::life::voice::Voice::AMP_EPS) {
                for (k, rt) in self.tones.range_mut(from..=through) {
                    if k.generation == generation && rt.tone.residual_bound(now).is_some() {
                        rt.tone.retire_phase3();
                    }
                }
            }
        }
    }
}
