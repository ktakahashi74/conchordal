//! Phrase interpretation payload, independent of a marginal or joint path's mass.

use super::{Exit, Foreground, WindowDescriptor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Link {
    pub span: Foreground,
    pub right_censored: bool,
    pub support: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition) struct Interpretation {
    pub id: u64,
    pub completed_foreground: Option<Foreground>,
    pub ending: Option<WindowDescriptor>,
    pub completed_ending: Option<WindowDescriptor>,
    pub foreground: Option<Foreground>,
    pub(super) links: [Option<Link>; 16],
    pub lost_links: u64,
    pub event: Option<(Exit, u64)>,
}

impl Interpretation {
    pub fn new(id: u64, [start, end]: [u64; 2]) -> Result<Self, &'static str> {
        if id == 0 || start > end {
            return Err("invalid phrase admission identity or interval");
        }
        Ok(Self {
            id,
            completed_foreground: None,
            ending: None,
            completed_ending: None,
            foreground: Some(Foreground {
                id,
                credit: id,
                start,
                heard_end: end,
            }),
            links: [None; 16],
            lost_links: 0,
            event: None,
        })
    }

    pub fn advance(
        &mut self,
        kind: Option<Exit>,
        [start, end]: [u64; 2],
        observed: bool,
        new_id: Option<u64>,
    ) -> Result<(), &'static str> {
        if start >= end
            || kind.is_some() != new_id.is_some()
            || new_id.is_some_and(|id| id <= self.id)
            || self
                .foreground
                .is_some_and(|f| f.start > start || f.heard_end > end)
        {
            return Err("invalid phrase transition identity or clock");
        }
        if let Some(kind) = kind {
            let id = new_id.unwrap();
            self.id = id;
            let Some(old) = self.foreground else {
                self.foreground = Some(Foreground {
                    id,
                    credit: id,
                    start,
                    heard_end: if observed { end } else { start },
                });
                self.event = None;
                return Ok(());
            };
            if kind != Exit::Reinterpret {
                self.completed_foreground = Some(old);
                self.completed_ending = self.ending;
                self.ending = None;
                let link = Link {
                    span: old,
                    right_censored: kind == Exit::Overlap,
                    support: old.heard_end.saturating_sub(old.start) as f64,
                };
                let slot = self
                    .links
                    .iter()
                    .position(Option::is_none)
                    .unwrap_or_else(|| {
                        self.lost_links += 1;
                        (0..16)
                            .min_by(|&a, &b| {
                                let a = self.links[a].unwrap();
                                let b = self.links[b].unwrap();
                                a.support
                                    .total_cmp(&b.support)
                                    .then(a.span.heard_end.cmp(&b.span.heard_end))
                            })
                            .unwrap()
                    });
                self.links[slot] = Some(link);
            }
            self.foreground = match kind {
                Exit::Inactive => None,
                Exit::Reinterpret => Some(Foreground { id, ..old }),
                _ => Some(Foreground {
                    id,
                    credit: id,
                    start: end,
                    heard_end: end,
                }),
            };
            self.event = Some((kind, end));
        }
        if observed && let Some(f) = self.foreground.as_mut() {
            f.heard_end = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrase_payload_preserves_original_endings_credit_and_bounded_links() {
        let mut original = Interpretation::new(1, [10, 90]).unwrap();
        original.ending = Some(WindowDescriptor {
            group: crate::temporal_cognition::ridge::Handle {
                bus: 0,
                epoch: 1,
                generation: 2,
            },
            start: 10,
            end: 90,
            raw_values: [Some(0.25); 6],
            coverage: [1.; 6],
            source_start: 0,
            available: 90,
        });
        for kind in [Exit::New, Exit::Overlap, Exit::Reinterpret, Exit::Inactive] {
            let mut child = original;
            child
                .advance(Some(kind), [100, 200], true, Some(2))
                .unwrap();
            assert_eq!(child.id, 2);
            assert_eq!(child.event, Some((kind, 200)));
            if kind == Exit::Reinterpret {
                let f = child.foreground.unwrap();
                assert_eq!((f.id, f.credit, f.start, f.heard_end), (2, 1, 10, 200));
                assert_eq!(child.ending, original.ending);
                assert!(child.completed_foreground.is_none());
                assert!(child.links.iter().all(Option::is_none));
            } else {
                assert_eq!(child.completed_foreground, original.foreground);
                assert_eq!(child.completed_ending, original.ending);
                let link = child.links[0].unwrap();
                assert_eq!(link.span.heard_end, 90);
                assert_eq!(link.support, 80.);
                assert_eq!(link.right_censored, kind == Exit::Overlap);
                assert_eq!(child.foreground.is_none(), kind == Exit::Inactive);
            }
            let old = child;
            child.advance(None, [200, 300], false, None).unwrap();
            assert_eq!(child, old, "missing stay cannot refresh heard endpoints");
        }
        let mut state = original;
        for id in 2..=20 {
            state
                .advance(
                    Some(Exit::Overlap),
                    [(id - 1) * 100, id * 100],
                    true,
                    Some(id),
                )
                .unwrap();
        }
        assert_eq!(state.links.iter().flatten().count(), 16);
        assert_eq!(state.lost_links, 3);
        assert!(state.links.iter().flatten().any(|l| l.span.id == 1));
        for args in [
            (None, [2000, 2000], None),
            (Some(Exit::New), [2000, 2100], Some(20)),
            (None, [2000, 2100], Some(21)),
        ] {
            let before = state;
            assert!(state.advance(args.0, args.1, true, args.2).is_err());
            assert_eq!(state, before);
        }
    }
}
