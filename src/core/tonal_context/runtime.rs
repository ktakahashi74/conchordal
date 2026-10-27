use super::*;
use std::cell::{Cell, RefCell};

/// A continuous PCM owner. A gap invalidates this run rather than filling silence.
#[derive(Clone)]
pub(crate) struct Context {
    model: Model,
    decoder: Arc<Decoder>,
    epoch_start: u64,
}

impl Context {
    pub(crate) fn new(decoder: Arc<Decoder>, epoch_start: u64) -> Self {
        Self {
            model: Model::new(48000),
            decoder,
            epoch_start,
        }
    }

    pub(crate) fn fresh(&self, at: u64) -> Self {
        Self::new(self.decoder.clone(), at)
    }

    pub(crate) fn invalidate(&mut self) {
        self.model.valid = false;
    }

    pub(crate) fn process(&mut self, pcm: &[f32]) {
        for &x in pcm {
            self.model.push(f64::from(x));
        }
    }

    pub(crate) fn snapshot(&self) -> Option<Arc<Snapshot>> {
        (self.model.valid && self.model.pitch.frame > 0).then(|| {
            Arc::new(Snapshot {
                observed: pearson(&self.model.pitch.local, &self.model.pitch.glob),
                epoch_start: self.epoch_start,
                end: self.epoch_start + self.model.n0 as u64,
                frame: self.model.pitch.frame,
                local: self.model.pitch.local,
                glob: self.model.pitch.glob,
                decoder: self.decoder.clone(),
            })
        })
    }
}

/// Immutable decoder input; owner/bus identity is supplied by its observation.
#[derive(Debug)]
pub(crate) struct Snapshot {
    pub(crate) observed: Option<f64>,
    pub(crate) epoch_start: u64,
    pub(crate) end: u64,
    frame: usize,
    local: [f64; W],
    glob: [f64; W],
    decoder: Arc<Decoder>,
}

impl Snapshot {
    pub(crate) fn query(&self, hz: f32) -> Option<f64> {
        self.decoder.query_memory(
            (self.end - self.epoch_start) as usize,
            self.frame,
            &self.local,
            &self.glob,
            f64::from(hz),
        )
    }
}

/// One pinned context for dwell and selection, with exact-frequency query reuse.
#[derive(Debug, Clone)]
pub(crate) struct Use {
    pub(crate) snapshot: Option<Arc<Snapshot>>,
    pub(crate) weight: f64,
    pub(crate) bounds: (f32, f32),
    cache: RefCell<Vec<(u32, Option<f64>)>>,
    pub(crate) selection_known: Cell<Option<bool>>,
}

impl Use {
    pub(crate) fn new() -> Self {
        Self {
            snapshot: None,
            weight: 0.,
            bounds: (0., 0.),
            cache: RefCell::new(Vec::with_capacity(32)),
            selection_known: Cell::new(None),
        }
    }

    pub(crate) fn pin(&mut self, snapshot: Option<Arc<Snapshot>>, weight: f64, bounds: (f32, f32)) {
        if !self
            .snapshot
            .as_ref()
            .zip(snapshot.as_ref())
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
        {
            self.cache.get_mut().clear();
        }
        self.selection_known.set(None);
        self.snapshot = snapshot;
        self.weight = weight;
        self.bounds = bounds;
    }

    pub(crate) fn query(&self, hz: f32) -> Option<f64> {
        let snapshot = self.snapshot.as_ref()?;
        let key = hz.to_bits();
        let mut cache = self.cache.borrow_mut();
        if let Some((_, q)) = cache.iter().find(|(k, _)| *k == key) {
            return *q;
        }
        let q = snapshot.query(hz);
        cache.push((key, q));
        q
    }

    pub(crate) fn score(&self, old: f32, q: f64) -> f32 {
        (f64::from(old) + self.weight * q).clamp(-(f32::MAX as f64), f32::MAX as f64) as f32
    }
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Record {
    pub at: Option<u64>,
    pub epoch_start: Option<u64>,
    pub snapshot_end: Option<u64>,
    pub weight: f64,
    pub current_hz: f32,
    pub current_q: Option<f64>,
    pub selection_known: Option<bool>,
    pub target_hz: f32,
}

impl Use {
    pub(crate) fn record(&self, at: Option<u64>, current_hz: f32, target_hz: f32) -> Record {
        Record {
            at,
            epoch_start: self.snapshot.as_ref().map(|s| s.epoch_start),
            snapshot_end: self.snapshot.as_ref().map(|s| s.end),
            weight: self.weight,
            current_hz,
            current_q: self.query(current_hz),
            selection_known: self.selection_known.get(),
            target_hz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tonal_continuity_birth_gap_and_fresh_epoch() {
        let mut original = Context::new(Decoder::unqueried_for_test(), 512);
        assert!(original.snapshot().is_none());
        let pcm: Vec<_> = (0..12288)
            .map(|n| (0.1 * (n as f64 * std::f64::consts::TAU * 220. / 48000.).sin()) as f32)
            .collect();
        let mut split = original.clone();
        original.process(&pcm);
        for hop in pcm.chunks(257) {
            split.process(hop);
        }
        assert_eq!(original.model.pitch.local, split.model.pitch.local);
        assert_eq!(original.model.pitch.glob, split.model.pitch.glob);
        let snapshot = original.snapshot().unwrap();
        assert_eq!((snapshot.epoch_start, snapshot.end), (512, 12800));
        let mut child = original.clone();
        child.process(&[0.; 512]);
        assert_eq!(original.model.n0, 12288);
        assert_eq!(child.model.n0, 12800);
        original.invalidate();
        original.process(&pcm);
        assert!(original.snapshot().is_none());
        assert!(child.snapshot().is_some());
        let fresh = original.fresh(99999);
        assert_eq!(fresh.epoch_start, 99999);
        assert_eq!(fresh.model.n0, 0);
        assert!(fresh.snapshot().is_none());
    }
}
