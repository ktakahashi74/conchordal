//! Bounded approximate retrieval in the existing square-root support embedding.
use super::{SMAX, Support};

#[derive(Clone, Copy)]
struct Shaped {
    projection: f64,
    id: u64,
    time: f64,
}
#[derive(Clone, Copy)]
struct Record {
    id: u64,
    time: f64,
}

pub(super) struct Index {
    axis: Vec<f64>,
    shaped: Vec<Shaped>,
    missing: Vec<Record>,
    chronological: Vec<Record>,
}

impl Index {
    pub fn new(coordinates: &[f64]) -> Self {
        // The axis is the declared coordinate vector normalized to unit length.
        // It orders proposals only; full Hellinger comparison remains downstream.
        let norm = coordinates.iter().map(|x| x * x).sum::<f64>().sqrt();
        let axis = coordinates
            .iter()
            .map(|x| if norm > 0. { x / norm } else { 0. })
            .collect();
        Self {
            axis,
            shaped: Vec::new(),
            missing: Vec::new(),
            chronological: Vec::new(),
        }
    }

    pub fn bytes(&self) -> usize {
        self.axis.capacity() * size_of::<f64>()
            + self.shaped.capacity() * size_of::<Shaped>()
            + (self.missing.capacity() + self.chronological.capacity()) * size_of::<Record>()
    }

    pub fn project(&self, support: Support<'_>) -> Option<f64> {
        if !support.valid || support.residual != 0. {
            return None;
        }
        let total = support.bins.iter().map(|b| f64::from(b.mass)).sum::<f64>();
        if total <= 0. {
            return None;
        }
        Some(
            support
                .bins
                .iter()
                .map(|b| (f64::from(b.mass) / total).sqrt() * self.axis[b.index])
                .sum(),
        )
    }

    pub fn insert(&mut self, id: u64, time: f64, projection: Option<f64>) {
        debug_assert!(
            self.chronological
                .last()
                .is_none_or(|last| last.id < id && last.time < time)
        );
        let record = Record { id, time };
        self.chronological.push(record);
        if let Some(projection) = projection {
            let entry = Shaped {
                projection,
                id,
                time,
            };
            let at = self.shaped.partition_point(|old| {
                old.projection
                    .total_cmp(&projection)
                    .then(old.id.cmp(&id))
                    .is_lt()
            });
            self.shaped.insert(at, entry);
        } else {
            self.missing.push(record);
        }
    }

    /// Expiration removes an ID prefix. Compact each index at most once per batch.
    pub fn expire_before(&mut self, first_live_id: u64) {
        let removed = self.chronological.partition_point(|r| r.id < first_live_id);
        if removed == 0 {
            return;
        }
        if removed == self.chronological.len() {
            self.chronological.clear();
            self.shaped.clear();
            self.missing.clear();
            return;
        }
        drop(self.chronological.drain(..removed));
        let missing = self.missing.partition_point(|r| r.id < first_live_id);
        drop(self.missing.drain(..missing));
        self.shaped.retain(|r| r.id >= first_live_id);
    }

    /// Visit at most limit + SMAX entries: the latest SMAX anchors can be too recent.
    /// Unavailable evidence never triggers an exhaustive refill.
    pub fn nearby(&self, query: Support<'_>, before: f64, limit: usize, out: &mut Vec<u64>) {
        out.clear();
        if limit == 0 {
            return;
        }
        let Some(projection) = self.project(query) else {
            let end = self.chronological.partition_point(|r| r.time < before);
            out.extend(
                self.chronological[..end]
                    .iter()
                    .rev()
                    .take(limit)
                    .map(|r| r.id),
            );
            return;
        };
        let mut left = self.shaped.partition_point(|r| r.projection < projection);
        let mut right = left;
        let mut missing = self.missing.partition_point(|r| r.time < before);
        let mut missing_turn = false;
        for _ in 0..limit.saturating_add(SMAX) {
            if out.len() == limit {
                break;
            }
            let no_shape = left == 0 && right == self.shaped.len();
            // Alternate represented and missing evidence without adding a kernel weight.
            if missing > 0 && (missing_turn || no_shape) {
                missing -= 1;
                out.push(self.missing[missing].id);
                missing_turn = false;
                continue;
            }
            if no_shape {
                break;
            }
            let take_left = left > 0
                && (right == self.shaped.len() || {
                    let a = self.shaped[left - 1];
                    let b = self.shaped[right];
                    (a.projection - projection)
                        .abs()
                        .total_cmp(&(b.projection - projection).abs())
                        .then(a.id.cmp(&b.id))
                        .is_le()
                });
            let entry = if take_left {
                left -= 1;
                self.shaped[left]
            } else {
                let entry = self.shaped[right];
                right += 1;
                entry
            };
            if entry.time < before {
                out.push(entry.id);
            }
            missing_turn = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_cognition::harmonic_relations::{Bin, HMAX};

    #[test]
    fn harmonic_relations_index_bounds_visits_and_removes_expired_evidence() {
        let mut index = Index::new(&[0., 1., 2.]);
        let bins = [Bin { index: 1, mass: 1. }];
        let shape = Support {
            bins: &bins,
            valid: true,
            ..Default::default()
        };
        let projection = index.project(shape);
        for id in 1..=100 {
            index.insert(id, id as f64, projection);
        }
        let mut out = Vec::with_capacity(HMAX);
        index.nearby(shape, 101., HMAX, &mut out);
        assert_eq!(out, (1..=HMAX as u64).collect::<Vec<_>>());
        index.expire_before(2);
        index.nearby(shape, 101., HMAX, &mut out);
        assert_eq!(out[0], 2);
        index.nearby(shape, 0., HMAX, &mut out);
        assert!(out.is_empty());
        index.insert(101, 101., None);
        index.nearby(shape, 102., HMAX, &mut out);
        assert!(out.contains(&101));
        index.nearby(Support::default(), 102., HMAX, &mut out);
        assert_eq!(out[0], 101);
        index.expire_before(102);
        index.nearby(Support::default(), 102., HMAX, &mut out);
        assert!(out.is_empty());

        // A valid but distant endpoint does not authorize scanning past the budget.
        let mut crowded = Index::new(&[0., 1., 2.]);
        crowded.insert(1, 1., Some(0.));
        for id in 2..=100 {
            crowded.insert(id, id as f64, projection);
        }
        crowded.nearby(shape, 2., HMAX, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn harmonic_relations_index_projection_uses_normalized_square_roots() {
        let index = Index::new(&[-1., 0., 1.]);
        let bins = [Bin { index: 0, mass: 1. }, Bin { index: 2, mass: 1. }];
        let shape = Support {
            bins: &bins,
            valid: true,
            ..Default::default()
        };
        assert_eq!(index.project(shape), Some(0.));
        assert_eq!(
            index.project(Support {
                residual: 0.5,
                ..shape
            }),
            None
        );
    }
    #[test]
    fn harmonic_relations_index_expires_mixed_prefixes_in_one_batch() {
        let mut index = Index::new(&[0., 1.]);
        for id in 1..=64 {
            index.insert(id, id as f64, (id % 2 == 0).then_some(1.));
        }
        index.expire_before(33);
        assert_eq!(index.chronological.len(), 32);
        assert_eq!(index.shaped.len(), 16);
        assert_eq!(index.missing.len(), 16);
        assert!(index.chronological.iter().all(|r| r.id >= 33));
        assert!(index.shaped.iter().all(|r| r.id >= 33));
        assert!(index.missing.iter().all(|r| r.id >= 33));
        index.expire_before(65);
        assert!(index.chronological.is_empty());
        assert!(index.shaped.is_empty());
        assert!(index.missing.is_empty());
    }
}
