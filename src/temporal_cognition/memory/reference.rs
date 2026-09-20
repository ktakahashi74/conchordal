//! Finite offline reference over every retained episode, knot anchor and refinement.

use super::*;
use std::io::BufRead;

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct ResultRow {
    id: u64,
    cost: Option<f64>,
    anchor: Option<usize>,
    cells: u64,
    target: bool,
}

// Search-policy experiment only: every observed query knot participates in ranking.
fn whole_query_rank(
    query: &Query<'_>,
    episode: &Episode,
    samples: Option<usize>,
    spacing: usize,
) -> Option<(f64, usize)> {
    let a = &query.descriptor.knots;
    let b = &episode.descriptor.knots;
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let supported: Vec<_> = a
        .iter()
        .filter(|k| k.observed_sec > 0. && k.mask != 0 && k.gap == 0)
        .collect();
    let total: f64 = supported.iter().map(|k| k.observed_sec).sum();
    let selected: Vec<_> = if let Some(n) = samples {
        (0..n)
            .filter_map(|i| {
                let target = total * (i as f64 + 0.5) / n as f64;
                let mut cumulative = 0.;
                supported.iter().find_map(|k| {
                    cumulative += k.observed_sec;
                    (cumulative > target).then_some((*k, total / n as f64))
                })
            })
            .collect()
    } else {
        supported.iter().map(|k| (*k, k.observed_sec)).collect()
    };
    let mut anchors = [Anchor::default(); CAPACITY];
    let mut diagnostic = AnchorDiagnostic::default();
    matcher::coarse(
        a,
        b,
        &CoarseConfig {
            spacing: spacing as u32,
            limit: 128 / spacing as u32,
            bounds: [2.; 2],
            grid: 1. / 16.,
            scales: query.scales,
        },
        &mut anchors,
        &mut diagnostic,
    )
    .unwrap();
    let mut best: Option<(f64, usize)> = None;
    for (index, t) in anchors.iter().take(b.len().div_ceil(spacing)).enumerate() {
        let anchor = index * spacing;
        if t.valid == 0
            || t.bound_hit != 0
            || t.out_of_range != 0
            || t.pitch_samples == 0
            || t.interval_samples == 0
        {
            continue;
        }
        let ratio = 2_f64.powf(t.applied[1]);
        let mut j = anchor;
        let mut loss = 0.;
        let mut duration = 0.;
        let mut paired = 0.;
        for &(k, weight) in &selected {
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
            let mask = k.mask & b[j].mask & b[next].mask;
            if mask == 0 {
                loss += weight;
                continue;
            }
            let mut residual = 0.;
            for d in 0..10 {
                if mask & (1 << d) == 0 {
                    continue;
                }
                let predicted = b[j].values[d] + fraction * (b[next].values[d] - b[j].values[d]);
                let shift = if d == 0 { t.applied[0] } else { 0. };
                residual += ((k.values[d] - predicted - shift) / query.scales[d]).powi(2);
            }
            loss += weight * residual / f64::from(mask.count_ones());
            paired += weight;
        }
        if paired > 0. && duration > 0. {
            let cost = loss / duration;
            if best.is_none_or(|old| cost < old.0) {
                best = Some((cost, anchor));
            }
        }
    }
    best
}

fn exhaustive(
    query: &Query<'_>,
    episode: &Episode,
    material: super::super::long_form::Material,
) -> Result<ResultRow, Error> {
    validate_query(query, std::slice::from_ref(episode))?;
    let mut row = ResultRow {
        id: episode.identity.id,
        cost: None,
        anchor: None,
        cells: 0,
        target: episode
            .descriptor
            .knots
            .first()
            .is_some_and(|k| material.is_target(k.start, episode.first_observed_end)),
    };
    if episode.epoch != query.epoch
        || episode.available_end > query.observed_end
        || episode.first_observed_end > query.descriptor.knots[0].start
    {
        return Ok(row);
    }
    matcher::validate(&episode.descriptor.knots, episode.available_end)?;
    if episode.scales != query.scales {
        return Err(Error::InvalidInput);
    }
    let mut anchors = [Anchor::default(); CAPACITY];
    let mut diagnostic = AnchorDiagnostic::default();
    matcher::coarse(
        &query.descriptor.knots,
        &episode.descriptor.knots,
        &CoarseConfig {
            spacing: 1,
            limit: 128,
            bounds: [2.; 2],
            grid: 1. / 16.,
            scales: query.scales,
        },
        &mut anchors,
        &mut diagnostic,
    )?;
    let mut output = Output::default();
    for (index, transform) in anchors
        .iter()
        .take(episode.descriptor.knots.len())
        .enumerate()
    {
        if transform.valid == 0
            || transform.bound_hit != 0
            || transform.out_of_range != 0
            || transform.pitch_samples == 0
            || transform.interval_samples == 0
        {
            continue;
        }
        let choices = refinements(transform);
        for &shift in &choices[0] {
            for &tempo in &choices[1] {
                if shift.abs() >= 2. || tempo.abs() >= 2. {
                    continue;
                }
                matcher::dtw(
                    &query.descriptor.knots,
                    &episode.descriptor.knots,
                    &Config {
                        anchor: index as u32,
                        band: 128,
                        shift,
                        ratio: 2_f64.powf(tempo),
                        tempo_shift: tempo,
                        scales: query.scales,
                        insertion: 1.,
                        deletion: 1.,
                    },
                    &mut output,
                )?;
                row.cells += u64::from(output.cells);
                if output.observed > 0
                    && output.valid_coordinates > 0
                    && output.total.is_finite()
                    && output.band_edge == 0
                {
                    let cost = output.total / f64::from(output.observed);
                    if row.cost.is_none_or(|old| cost < old) {
                        row.cost = Some(cost);
                        row.anchor = Some(index);
                    }
                }
            }
        }
    }
    Ok(row)
}

#[derive(serde::Deserialize)]
#[serde(tag = "kind")]
enum Capture {
    #[serde(rename = "episode")]
    Episode {
        id: u64,
        generation: u64,
        epoch: u64,
        available_end: f64,
        first_observed_end: f64,
        scales: [f64; 10],
        knots: Vec<Knot>,
    },
    #[serde(rename = "query")]
    Query {
        query_id: u64,
        epoch: u64,
        start: f64,
        end: f64,
        observed_end: f64,
        scales: [f64; 10],
        knots: Vec<Knot>,
        bank_ids: Vec<u64>,
        matches: Vec<serde_json::Value>,
    },
}

#[test]
#[ignore = "explicit exhaustive I9 reference over the captured finite performance"]
fn captured_long_return_queries_against_roomy_exhaustive_bank() {
    let selected_probe = std::env::var("CONCHORDAL_I9_REFERENCE_PROBE")
        .ok()
        .map(|s| s.parse::<usize>().expect("registered probe index required"));
    assert!(selected_probe.is_none_or(|i| i < 3));
    let path = std::path::PathBuf::from(
        std::env::var("CONCHORDAL_I9_CAPTURE")
            .unwrap_or_else(|_| "target/temporal-dcc/i9-captured-1800/bus1-epoch0.jsonl".into()),
    );
    let material_path = path.parent().unwrap().join("material.json");
    let material: super::super::long_form::Material = if material_path.exists() {
        serde_json::from_slice(&std::fs::read(material_path).unwrap()).unwrap()
    } else {
        Default::default()
    };
    let input = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut episodes = Vec::new();
    let mut queries: [Option<Capture>; 3] = [None, None, None];
    for line in input.lines() {
        let event: Capture = serde_json::from_str(&line.unwrap()).unwrap();
        match event {
            Capture::Episode {
                id,
                generation,
                epoch,
                available_end,
                first_observed_end,
                scales,
                knots,
            } => episodes.push(Episode {
                identity: Identity { id, generation },
                epoch,
                available_end,
                first_observed_end,
                scales,
                descriptor: Descriptor {
                    local_intervals: vec![None; knots.len()],
                    knots,
                },
            }),
            Capture::Query { end, .. } => {
                for (i, [begin, until]) in material.returns.into_iter().enumerate() {
                    if begin + 2. <= end && end < until && queries[i].is_none() {
                        queries[i] = Some(event);
                        break;
                    }
                }
            }
        }
    }
    let jobs = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8);
    let mut results = Vec::new();
    for (index, capture) in queries.into_iter().enumerate() {
        if selected_probe.is_some_and(|i| i != index) {
            continue;
        }
        let Some(Capture::Query {
            query_id,
            epoch,
            start,
            end,
            observed_end,
            scales,
            knots,
            bank_ids,
            matches,
        }) = capture
        else {
            panic!("missing preregistered return cue {index}");
        };
        let descriptor = Descriptor {
            local_intervals: vec![None; knots.len()],
            knots,
        };
        let query = Query {
            descriptor: &descriptor,
            epoch,
            end,
            observed_end,
            scales,
        };
        let full_bank: Vec<_> = episodes
            .iter()
            .filter(|e| {
                e.epoch == epoch && e.available_end <= observed_end && e.first_observed_end <= start
            })
            .cloned()
            .collect();
        let indexed = ordered(&query, &full_bank).unwrap();
        let mut whole_rank: Vec<_> = full_bank
            .iter()
            .filter_map(|e| {
                whole_query_rank(&query, e, None, 1).map(|(cost, anchor)| {
                    (
                        cost,
                        e.identity.id,
                        anchor,
                        material.is_target(e.descriptor.knots[0].start, e.first_observed_end),
                    )
                })
            })
            .collect();
        whole_rank.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut quantile_probes = Vec::new();
        for samples in [8, 16] {
            let began = std::time::Instant::now();
            let mut ranked: Vec<_> = full_bank
                .iter()
                .filter_map(|e| {
                    whole_query_rank(&query, e, Some(samples), 4).map(|(cost, anchor)| {
                        (
                            cost,
                            e.identity.id,
                            anchor,
                            material.is_target(e.descriptor.knots[0].start, e.first_observed_end),
                        )
                    })
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            quantile_probes.push(serde_json::json!({"samples":samples,"spacing":4,"wall_seconds":began.elapsed().as_secs_f64(),
                "first_target_rank":ranked.iter().position(|r|r.3),"cutoff_tie":ranked.len()>16 && ranked[15].0==ranked[16].0,
                "first_32":ranked.iter().take(32).collect::<Vec<_>>() }));
        }
        let mut spacing_probes = Vec::new();
        for spacing in [1, 2, 4] {
            let mut ranked = Vec::new();
            let mut anchors = [Anchor::default(); CAPACITY];
            let mut diagnostic = AnchorDiagnostic::default();
            for e in &full_bank {
                matcher::coarse(
                    &descriptor.knots,
                    &e.descriptor.knots,
                    &CoarseConfig {
                        spacing,
                        limit: 128 / spacing,
                        bounds: [2.; 2],
                        grid: 1. / 16.,
                        scales,
                    },
                    &mut anchors,
                    &mut diagnostic,
                )
                .unwrap();
                if diagnostic.index >= 0 {
                    let a = anchors[diagnostic.index as usize];
                    ranked.push((
                        a.cost,
                        e.identity.id,
                        diagnostic.index as u32 * spacing,
                        material.is_target(e.descriptor.knots[0].start, e.first_observed_end),
                    ));
                }
            }
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            spacing_probes.push(serde_json::json!({"spacing":spacing,
                "first_target_rank":ranked.iter().position(|r| r.3),
                "cutoff_tie":ranked.len()>16 && ranked[15].0 == ranked[16].0,
                "first_32":ranked.iter().take(32).collect::<Vec<_>>() }));
        }
        let began = std::time::Instant::now();
        let mut rows = std::thread::scope(|scope| {
            let handles: Vec<_> = episodes
                .chunks(episodes.len().div_ceil(jobs))
                .map(|chunk| {
                    let query = &query;
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|e| exhaustive(query, e, material).unwrap())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        rows.sort_by(|a, b| {
            a.cost
                .unwrap_or(f64::INFINITY)
                .total_cmp(&b.cost.unwrap_or(f64::INFINITY))
                .then(a.id.cmp(&b.id))
        });
        let targets: Vec<_> = rows
            .iter()
            .filter(|r| r.target && r.cost.is_some_and(|c| c <= 1.))
            .collect();
        let present: Vec<_> = targets
            .iter()
            .filter(|r| bank_ids.contains(&r.id))
            .map(|r| r.id)
            .collect();
        let reached: Vec<_> = targets
            .iter()
            .filter(|r| {
                matches.iter().any(|m| {
                    m["id"] == r.id
                        && m["supported"] == true
                        && m["ambiguous"] == false
                        && m["cost"].as_f64().is_some_and(|c| c <= 1.)
                })
            })
            .map(|r| r.id)
            .collect();
        let reached_without_eviction: Vec<_> = targets
            .iter()
            .filter(|r| {
                indexed.matches.iter().any(|m| {
                    m.relation.identity.id == r.id
                        && m.relation.supported
                        && !m.relation.ambiguous
                        && m.cost.is_some_and(|c| c <= 1.)
                })
            })
            .map(|r| r.id)
            .collect();
        let mut capacity_probes = Vec::new();
        for limit in [32, 64, 128, 256, 512, 1024] {
            let began = std::time::Instant::now();
            let report = ordered_index(&query, &full_bank, true, limit).unwrap();
            let reached: Vec<_> = targets
                .iter()
                .filter(|r| {
                    report.matches.iter().any(|m| {
                        m.relation.identity.id == r.id
                            && m.relation.supported
                            && !m.relation.ambiguous
                            && m.cost.is_some_and(|c| c <= 1.)
                    })
                })
                .map(|r| r.id)
                .collect();
            capacity_probes.push(serde_json::json!({
                "candidate_limit": limit, "targets_reached": reached,
                "dp_cells": report.dp_cells, "matches": report.matches.len(),
                "pruned_candidates": report.pruned_candidates, "pruned_ties": report.pruned_ties,
                "wall_seconds": began.elapsed().as_secs_f64()
            }));
        }
        let result = serde_json::json!({"probe":index,"query_id":query_id,"start":start,"end":end,"observed_end":observed_end,
            "archive_episodes":episodes.len(),"eligible_episodes":episodes.iter().filter(|e|e.available_end<=observed_end && e.first_observed_end<=start).count(),
            "reference_supported_targets":targets,"targets_in_bounded_bank":present,"targets_reached_in_bounded_search":reached,
            "targets_reached_without_eviction":reached_without_eviction,"full_bank_cutoff_tie":indexed.cutoff_tie,
            "candidate_capacity_probes":capacity_probes,
            "index_spacing_probes":spacing_probes,"quantile_query_index_probes":quantile_probes,
            "whole_query_index_probe":{"first_target_rank":whole_rank.iter().position(|r|r.3),
                "cutoff_tie":whole_rank.len()>16 && whole_rank[15].0 == whole_rank[16].0,
                "first_32":whole_rank.iter().take(32).collect::<Vec<_>>()},
            "full_bank_dp_cells":indexed.dp_cells,"index_pruned_candidates":indexed.pruned_candidates,"index_pruned_ties":indexed.pruned_ties,"full_bank_matches":indexed.matches.iter().map(|m| serde_json::json!({
                "id":m.relation.identity.id,"cost":m.cost,"supported":m.relation.supported,"ambiguous":m.relation.ambiguous,"anchor":m.anchor})).collect::<Vec<_>>(),
            "best_reference":rows.iter().take(5).collect::<Vec<_>>(),"dp_cells":rows.iter().map(|r|r.cells).sum::<u64>(),
            "wall_seconds":began.elapsed().as_secs_f64(),"jobs":jobs});
        println!("I9_ROOMY {result}");
        results.push(result);
        std::fs::write(
            path.with_file_name(selected_probe.map_or_else(
                || "runtime-index-reference.json".to_owned(),
                |i| format!("runtime-index-reference-probe-{i}.json"),
            )),
            serde_json::to_vec_pretty(&results).unwrap(),
        )
        .unwrap();
    }
}
