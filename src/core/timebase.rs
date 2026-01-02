pub type Tick = u64;

#[derive(Clone, Copy, Debug)]
pub struct Timebase {
    pub fs: f32,
    pub hop: usize,
}

impl Timebase {
    pub fn tick_to_sec(&self, t: Tick) -> f32 {
        t as f32 / self.fs
    }

    pub fn sec_to_tick(&self, s: f32) -> Tick {
        if s <= 0.0 {
            return 0;
        }
        let tick = (s as f64 * self.fs as f64).round();
        tick as Tick
    }

    pub fn frame_start_tick(&self, frame_idx: u64) -> Tick {
        frame_idx.saturating_mul(self.hop as u64)
    }

    pub fn frame_end_tick(&self, frame_idx: u64) -> Tick {
        self.frame_start_tick(frame_idx)
            .saturating_add(self.hop as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::{Tick, Timebase};

    #[test]
    fn sec_tick_round_trip() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 512,
        };
        let t: Tick = 12_345;
        let sec = tb.tick_to_sec(t);
        let round = tb.sec_to_tick(sec);
        assert_eq!(round, t);
    }

    #[test]
    fn frame_bounds() {
        let tb = Timebase {
            fs: 48_000.0,
            hop: 256,
        };
        let start = tb.frame_start_tick(10);
        let end = tb.frame_end_tick(10);
        assert_eq!(start, 2560);
        assert_eq!(end, 2816);
    }
}
