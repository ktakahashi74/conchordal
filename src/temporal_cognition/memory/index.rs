//! Query-wide acoustic ranking; final correspondence still uses the original DTW.

use super::*;

pub(super) const SAMPLES: usize = 16;

pub(super) fn samples(a: &[Knot]) -> [Option<(usize, f64)>; SAMPLES] {
    let duration: f64 = a
        .iter()
        .filter(|k| k.mask != 0 && k.gap == 0)
        .map(|k| k.observed_sec)
        .sum();
    std::array::from_fn(|i| {
        let target = duration * (i as f64 + 0.5) / SAMPLES as f64;
        let mut cumulative = 0.;
        a.iter().enumerate().find_map(|(j, k)| {
            if k.mask == 0 || k.gap != 0 {
                return None;
            }
            cumulative += k.observed_sec;
            (cumulative > target).then_some((j, duration / SAMPLES as f64))
        })
    })
}

pub(super) fn select(
    query: &Query<'_>,
    b: &[Knot],
    anchors: &[Anchor; CAPACITY],
    samples: &[Option<(usize, f64)>; SAMPLES],
) -> Result<Option<(usize, Anchor)>, Error> {
    let a = &query.descriptor.knots;
    if a.is_empty() || b.is_empty() {
        return Ok(None);
    }
    let mut best: Option<(usize, Anchor)> = None;
    for (index, transform) in anchors.iter().take(b.len().div_ceil(4)).enumerate() {
        if transform.valid == 0 || transform.bound_hit != 0 || transform.out_of_range != 0 {
            continue;
        }
        let anchor = index * 4;
        let ratio = 2_f64.powf(transform.applied[1]);
        let mut j = anchor;
        let mut loss = 0.;
        let mut duration = 0.;
        let mut paired = 0.;
        for &(i, weight) in samples.iter().flatten() {
            let k = &a[i];
            duration += weight;
            let at = b[anchor].time + (k.time - a[0].time) * ratio;
            while j + 1 < b.len() && b[j + 1].time <= at {
                j += 1;
            }
            if at > b.last().unwrap().end || b[j].gap != 0 || b[j].observed_sec <= 0. {
                loss += weight;
                continue;
            }
            let next = (j + 1).min(b.len() - 1);
            let fraction = if next == j {
                0.
            } else {
                ((at - b[j].time) / (b[next].time - b[j].time)).clamp(0., 1.)
            };
            if fraction > 0. && (b[next].gap != 0 || b[next].observed_sec <= 0.) {
                loss += weight;
                continue;
            }
            let mask = k.mask & b[j].mask & if fraction == 0. { 1023 } else { b[next].mask };
            if mask == 0 {
                loss += weight;
                continue;
            }
            let mut residual = 0.;
            for d in 0..10 {
                if mask & (1 << d) == 0 {
                    continue;
                }
                let value = if fraction == 0. {
                    b[j].values[d]
                } else {
                    b[j].values[d] + fraction * (b[next].values[d] - b[j].values[d])
                };
                let shift = if d == 0 { transform.applied[0] } else { 0. };
                residual += ((k.values[d] - value - shift) / query.scales[d]).powi(2);
            }
            loss += weight * residual / f64::from(mask.count_ones());
            paired += weight;
        }
        if paired > 0. && duration > 0. {
            let cost = loss / duration;
            if !cost.is_finite() {
                return Err(Error::NumericalRange);
            }
            if best.is_none_or(|(_, old)| cost < old.cost) {
                best = Some((anchor, Anchor { cost, ..*transform }));
            }
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(start: f64, variant: usize) -> Descriptor {
        Descriptor {
            knots: (0..32)
                .map(|i| {
                    let value = if i < 8 {
                        if variant == 1 { 0.1 } else { 0. }
                    } else if variant == 2 {
                        3. + (i % 3) as f64
                    } else {
                        ((i * 11) % 17) as f64 * 0.1
                    };
                    Knot {
                        values: [8., value, 0., 0., 0., 0., 0., 0., 0., 0.],
                        mask: 3,
                        timing: 1,
                        time: start + i as f64 + 0.5,
                        start: start + i as f64,
                        end: start + i as f64 + 1.,
                        observed_sec: 1.,
                        raw_start: start + i as f64,
                        raw_end: start + i as f64 + 1.,
                        available_end: start + i as f64 + 1.,
                        ..Knot::default()
                    }
                })
                .collect(),
            local_intervals: vec![Some(1.); 32],
        }
    }

    #[test]
    fn whole_query_finds_continuation_behind_sixteen_prefix_decoys() {
        let cue = descriptor(100., 0);
        let query = Query {
            descriptor: &cue,
            epoch: 1,
            end: 132.,
            observed_end: 132.,
            scales: [1.; 10],
        };
        let mut bank: Vec<_> = (1..=17)
            .map(|id| Episode {
                identity: Identity { id, generation: 1 },
                epoch: 1,
                available_end: 32.,
                first_observed_end: 32.,
                scales: [1.; 10],
                descriptor: descriptor(0., if id == 17 { 1 } else { 2 }),
            })
            .collect();
        assert!(
            ordered_prefix(&query, &bank)
                .unwrap()
                .matches
                .iter()
                .all(|m| m.relation.identity.id != 17)
        );
        for _ in 0..2 {
            let result = ordered(&query, &bank).unwrap();
            assert!(!result.cutoff_tie);
            assert_eq!(result.pruned_candidates, 16);
            assert_eq!(result.pruned_ties, 16);
            assert!(!result.matches.is_empty());
            assert!(result.matches.iter().all(|m| m.relation.identity.id == 17));
            assert!(
                result.matches.iter().any(|m| m.relation.supported
                    && !m.relation.ambiguous
                    && m.cost.unwrap() < 0.01)
            );
            bank.reverse();
        }
        bank.pop();
        let tie = ordered(&query, &bank).unwrap();
        assert_eq!(tie.pruned_ties, 0);
    }

    #[test]
    fn masked_next_value_cannot_poison_exact_supported_sample() {
        let mut cue = descriptor(100., 0);
        for k in cue.knots.iter_mut().skip(1) {
            k.observed_sec = 0.;
        }
        let mut reference = descriptor(0., 0);
        reference.knots[1].mask = 0;
        reference.knots[1].values.fill(f64::NAN);
        let query = Query {
            descriptor: &cue,
            epoch: 1,
            end: 132.,
            observed_end: 132.,
            scales: [1.; 10],
        };
        let mut anchors = [Anchor::default(); CAPACITY];
        anchors[0] = Anchor {
            valid: 1,
            pitch_samples: 1,
            interval_samples: 1,
            ..Anchor::default()
        };
        let result = select(&query, &reference.knots, &anchors, &samples(&cue.knots))
            .unwrap()
            .unwrap();
        assert_eq!(result.0, 0);
        assert_eq!(result.1.cost, 0.);
        let mut missing = cue.clone();
        missing.knots[0].observed_sec = 0.;
        assert!(samples(&missing.knots).iter().all(Option::is_none));
        assert!(
            select(&query, &reference.knots, &anchors, &[None; SAMPLES])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn one_knot_query_keeps_acoustic_match_without_inventing_tempo() {
        let mut cue = descriptor(100., 0);
        cue.knots.truncate(1);
        cue.local_intervals.truncate(1);
        let query = Query {
            descriptor: &cue,
            epoch: 1,
            end: 101.,
            observed_end: 101.,
            scales: [1.; 10],
        };
        let bank = [Episode {
            identity: Identity {
                id: 1,
                generation: 1,
            },
            epoch: 1,
            available_end: 32.,
            first_observed_end: 32.,
            scales: [1.; 10],
            descriptor: descriptor(0., 0),
        }];
        let report = ordered(&query, &bank).unwrap();
        assert!(
            report
                .matches
                .iter()
                .any(|m| m.relation.supported && m.cost == Some(0.))
        );
        assert!(
            report
                .matches
                .iter()
                .all(|m| m.relation.transformation[1].is_none())
        );
        assert!(report.coarse[0].approximate);
    }
}
