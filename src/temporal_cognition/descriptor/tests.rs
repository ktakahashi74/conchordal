use super::*;
use serde_json::Value;

fn config(capacity: usize, cadence: u8) -> Config {
    Config {
        group: Handle {
            bus: 0,
            epoch: 2,
            generation: 3,
        },
        sample_rate: 48000,
        first_hop_start: 0,
        hop: 512,
        cadence,
        span_start: 0.,
        span_end: None,
        scales: [1.; 10],
        capacity,
    }
}

fn raw(index: u64, values: [Option<f64>; 10]) -> RawDescriptor {
    RawDescriptor {
        group: config(8, 2).group,
        start: index * 512,
        end: (index + 1) * 512,
        known_samples: 512,
        values,
        source_start: index.saturating_sub(3) * 512,
        source_end: (index + 1) * 512,
        available_end: (index + 2) * 512,
    }
}

fn decode(hex: &str) -> Vec<Block> {
    let bytes: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect();
    assert_eq!(bytes.len() % 320, 0);
    bytes
        .chunks_exact(320)
        .map(|b| {
            let f = |i| f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
            Block {
                moments: std::array::from_fn(|j| [f(3 * j), f(3 * j + 1), f(3 * j + 2)]),
                time: f(30),
                observed: f(31),
                start: f(32),
                end: f(33),
                raw_start: f(34),
                raw_end: f(35),
                available: f(36),
                missing: f(37),
                epoch: u64::from_le_bytes(b[304..312].try_into().unwrap()),
                generation: u64::from_le_bytes(b[312..320].try_into().unwrap()),
            }
        })
        .collect()
}

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 128. * f64::EPSILON * a.abs().max(b.abs()).max(1.),
        "actual={a} expected={b}"
    );
}

fn compare(actual: &[Block], expected: &[Block], mismatches: &mut usize) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(
            (
                a.start,
                a.end,
                a.raw_start,
                a.raw_end,
                a.available,
                a.epoch,
                a.generation
            ),
            (
                b.start,
                b.end,
                b.raw_start,
                b.raw_end,
                b.available,
                b.epoch,
                b.generation
            )
        );
        let mut x = Vec::new();
        let mut y = Vec::new();
        a.encode(&mut x);
        b.encode(&mut y);
        for (x, y) in x[..304].chunks_exact(8).zip(y[..304].chunks_exact(8)) {
            let x = f64::from_le_bytes(x.try_into().unwrap());
            let y = f64::from_le_bytes(y.try_into().unwrap());
            *mismatches += usize::from(x.to_bits() != y.to_bits());
            close(x, y);
        }
        assert_eq!(a.matching().mask, b.matching().mask);
        assert_eq!(a.matching().timing, b.matching().timing);
        assert_eq!(a.matching().gap, b.matching().gap);
    }
}

#[test]
fn registered_python_blocks_cadence_and_compression_match_with_exact_input_bits() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/descriptors.json"
    ))
    .unwrap();
    let (mut checkpoints, mut operations, mut mismatches, mut max_error_drift) = (0, 0, 0, 0f64);
    for case in fixture["cases"].as_array().unwrap() {
        let mut cfg = config(
            case["capacity"].as_u64().unwrap() as usize,
            case["cadence"].as_u64().unwrap() as u8,
        );
        cfg.span_start = case["span_start_sample"].as_u64().unwrap() as f64 / 48000.;
        cfg.span_end = case["span_end_sample"].as_u64().map(|s| s as f64 / 48000.);
        cfg.scales =
            std::array::from_fn(|i| f64::from_bits(case["scales_bits"][i].as_u64().unwrap()));
        let mut span = Span::new(cfg).unwrap();
        let allocation = span.bank.blocks.as_ptr();
        for step in case["steps"].as_array().unwrap() {
            let cut = step["cut_sample"].as_u64().unwrap();
            operations += 1;
            match step["kind"].as_str().unwrap() {
                "push" => {
                    let r = &step["raw"];
                    let intervals: Vec<_> = r["known_sample_intervals"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap()))
                        .collect();
                    let raw = RawDescriptor {
                        group: cfg.group,
                        start: r["sample_start"].as_u64().unwrap(),
                        end: r["sample_end"].as_u64().unwrap(),
                        known_samples: intervals.iter().map(|(a, b)| b - a).sum(),
                        values: std::array::from_fn(|j| {
                            r["values_bits"][j].as_u64().map(f64::from_bits)
                        }),
                        source_start: r["source_start_sample"].as_u64().unwrap(),
                        source_end: r["source_end_sample"].as_u64().unwrap(),
                        available_end: r["available_sample"].as_u64().unwrap(),
                    };
                    assert!(
                        span.push(&raw, &intervals, cut).unwrap(),
                        "{}",
                        case["name"]
                    );
                }
                "gap" => span
                    .gap(
                        step["end_sample"].as_u64().unwrap() as f64 / 48000.,
                        step["available_sample"].as_u64().unwrap(),
                        cut,
                    )
                    .unwrap(),
                "finish" => {
                    let frozen = span.finish(cut).unwrap();
                    assert_eq!(frozen.group, cfg.group);
                    assert_eq!(frozen.reconstruction_error, span.bank.reconstruction_error);
                    assert_eq!(frozen.scales, span.bank.scales);
                    assert!(frozen.available_at <= cut);
                    let expected = decode(step["expected"]["knots_hex"].as_str().unwrap());
                    compare(&frozen.blocks, &expected, &mut mismatches);
                    assert_eq!(frozen.packed().len(), frozen.blocks.len() * 320);
                    if frozen.blocks.len() <= 128 {
                        assert!(frozen.matching().is_ok());
                    } else {
                        assert!(frozen.matching().is_err());
                    }
                }
                _ => unreachable!(),
            }
            assert_eq!(allocation, span.bank.blocks.as_ptr());
            assert!(span.bank.count <= cfg.capacity);
            if let Some(e) = step.get("expected") {
                checkpoints += 1;
                let expected = decode(e["knots_hex"].as_str().unwrap());
                compare(
                    &span.bank.blocks[..span.bank.count],
                    &expected,
                    &mut mismatches,
                );
                for (a, b) in [
                    (span.pending, &e["pending_hex"]),
                    (span.gap_pending, &e["pending_gap_hex"]),
                ] {
                    let reference = b.as_str().map(decode);
                    assert_eq!(a.is_some(), reference.is_some());
                    if let (Some(a), Some(b)) = (a, reference) {
                        compare(&[a], &b, &mut mismatches);
                    }
                }
                let error = f64::from_bits(e["reconstruction_error_bits"].as_u64().unwrap());
                close(span.bank.reconstruction_error, error);
                max_error_drift =
                    max_error_drift.max((span.bank.reconstruction_error - error).abs());
                assert_eq!(span.bank.merges, e["merges"].as_u64().unwrap());
                assert_eq!(
                    span.bank.coordinate_evaluations,
                    e["priority_coordinate_evaluations"].as_u64().unwrap()
                );
                assert_eq!(
                    span.max_insertions,
                    e["max_insertions_per_hop"].as_u64().unwrap()
                );
                assert_eq!(
                    span.last_insertions,
                    e["last_insertions_per_hop"].as_u64().unwrap()
                );
                assert_eq!(span.bank.frozen, e["frozen"].as_bool().unwrap());
            }
        }
    }
    println!(
        "DESCRIPTOR_ORACLE {}",
        serde_json::json!({"cases":fixture["cases"].as_array().unwrap().len(),"operations":operations,"checkpoints":checkpoints,"different_f64_fields":mismatches,"max_reconstruction_error_drift":max_error_drift})
    );
}

#[test]
fn exact_layout_masks_gap_timing_and_partial_acquisition_clipping() {
    assert_eq!(std::mem::size_of::<Block>(), 320);
    assert_eq!(std::mem::offset_of!(Block, time), 240);
    assert_eq!(std::mem::offset_of!(Block, start), 256);
    assert_eq!(std::mem::offset_of!(Block, epoch), 304);
    let mut cfg = config(8, 2);
    cfg.span_start = 100. / 48000.;
    cfg.span_end = Some(400. / 48000.);
    let mut span = Span::new(cfg).unwrap();
    let mut r = raw(0, [None; 10]);
    r.known_samples = 362;
    let intervals = [(0, 150), (300, 512)];
    span.push(&r, &intervals, 1024).unwrap();
    let before = span.bank.blocks[..span.bank.count].to_vec();
    assert!(!span.push(&r, &intervals, 1024).unwrap());
    assert!(span.push(&r, &[(0, 100), (250, 512)], 1024).is_err());
    assert_eq!(&span.bank.blocks[..span.bank.count], before);
    let frozen = span.finish(1024).unwrap();
    let b = frozen.blocks[0];
    close(b.observed, 150. / 48000.);
    close(b.missing, 150. / 48000.);
    assert_eq!(b.matching().mask, 0);
    assert_eq!(b.matching().timing, 0);
    assert_eq!(b.raw_start, 0.);
    assert_eq!(b.raw_end, 512. / 48000.);
    let r = raw(0, [Some(2.); 10]);
    let b = Block::from_raw(&r, 48000, &[(0, 512)], 100. / 48000., 400. / 48000.).unwrap();
    assert_eq!(b.matching().mask, 1023);
    close(b.moments[0][0], 300. / 48000.);
    assert_eq!(b.time, 512. / 48000.);
}

#[test]
fn protected_endpoints_earliest_ties_and_direct_reconstruction_error() {
    let mut bank = Bank::new([1.; 10], 3).unwrap();
    for i in 0..20 {
        let r = raw(i, [Some(2.); 10]);
        bank.append(
            Block::from_raw(
                &r,
                48000,
                &[(r.start, r.end)],
                r.start as f64 / 48000.,
                r.end as f64 / 48000.,
            )
            .unwrap(),
        )
        .unwrap();
    }
    assert_eq!(bank.count, 3);
    assert_eq!(bank.blocks[0].end, 512. / 48000.);
    assert_eq!(bank.blocks[1].start, 512. / 48000.);
    assert_eq!(bank.blocks[1].end, 19. * 512. / 48000.);
    assert_eq!(bank.blocks[2].start, 19. * 512. / 48000.);
    assert_eq!(bank.reconstruction_error, 0.);
    let mut bank = Bank::new([1.; 10], 8).unwrap();
    let mut original = Vec::new();
    for i in 0..300 {
        let r = raw(i, [Some(1e8 + i as f64 * 0.01); 10]);
        let b = Block::from_raw(
            &r,
            48000,
            &[(r.start, r.end)],
            r.start as f64 / 48000.,
            r.end as f64 / 48000.,
        )
        .unwrap();
        original.push(b);
        bank.append(b).unwrap();
    }
    let direct: f64 = bank.blocks[..bank.count]
        .iter()
        .map(|b| {
            original
                .iter()
                .filter(|r| r.start >= b.start && r.end <= b.end)
                .map(|r| {
                    r.moments
                        .iter()
                        .zip(b.moments)
                        .map(|(r, b)| r[0] * (r[1] - b[1]).powi(2))
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum();
    let difference = (bank.reconstruction_error - direct).abs();
    assert!(
        difference < 1e-5,
        "recorded={} direct={direct}",
        bank.reconstruction_error
    );
    println!(
        "DESCRIPTOR_RECONSTRUCTION {}",
        serde_json::json!({"recorded":bank.reconstruction_error,"direct_original_samples":direct,"absolute_difference":difference,"general_upper_bound":false})
    );
}

#[test]
fn gap_resume_three_insertions_and_frozen_copy_preserve_causal_evidence() {
    let mut span = Span::new(config(8, 2)).unwrap();
    let a = raw(0, [Some(1.); 10]);
    span.push(&a, &[(a.start, a.end)], 1024).unwrap();
    let b = raw(3, [Some(2.); 10]);
    span.push(&b, &[(b.start, b.end)], 2560).unwrap();
    assert_eq!(span.last_insertions, 3);
    assert_eq!(span.bank.count, 3);
    assert!(span.finish(2048).is_err());
    let frozen = span.finish(2560).unwrap();
    let original = frozen.packed();
    let mut copy = frozen.packed();
    copy[0] ^= 1;
    assert_eq!(original, frozen.packed());
    assert_ne!(original, copy);
    let descriptor = frozen.matching().unwrap();
    assert_eq!(descriptor.local_intervals, vec![None, None, None]);
    assert!(
        span.push(&raw(4, [Some(1.); 10]), &[(2048, 2560)], 3072)
            .is_err()
    );
    assert_eq!(span.finish(3072).unwrap().packed(), original);
}

#[test]
fn invalid_boundary_replay_and_future_cut_do_not_consume_weight() {
    let mut span = Span::new(config(8, 2)).unwrap();
    let a = raw(0, [Some(1.); 10]);
    span.push(&a, &[(0, 512)], 1024).unwrap();
    let before = span.pending;
    let b = raw(1, [Some(2.); 10]);
    assert!(span.push(&b, &[(512, 1024)], 1024).is_err());
    assert_eq!(span.pending, before);
    let mut conflict = a;
    conflict.values[0] = Some(4.);
    assert!(span.push(&conflict, &[(0, 512)], 1024).is_err());
    assert_eq!(span.pending, before);
    let mut foreign = b;
    foreign.group.bus = 1;
    assert!(span.push(&foreign, &[(512, 1024)], 1536).is_err());
    assert_eq!(span.pending, before);
    assert!(span.push(&b, &[(512, 1024)], 1536).unwrap());
    assert_eq!(span.bank.count, 1);
    close(span.bank.blocks[0].observed, 1024. / 48000.);
    let mut cfg = config(8, 2);
    cfg.span_end = Some(4096. / 48000.);
    let mut incomplete = Span::new(cfg).unwrap();
    incomplete.push(&a, &[(0, 512)], 1024).unwrap();
    assert!(incomplete.finish(1024).is_err());
    assert!(incomplete.pending.is_some());
    assert_eq!(incomplete.bank.count, 0);
}

#[test]
fn registered_prefix_queries_match_without_changing_live_state() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/prefixes.json"
    ))
    .unwrap();
    let (mut queries, mut operations, mut mismatches) = (0, 0, 0);
    for case in fixture["cases"].as_array().unwrap() {
        let mut span = Span::new(config(
            case["capacity"].as_u64().unwrap() as usize,
            case["cadence"].as_u64().unwrap() as u8,
        ))
        .unwrap();
        let allocation = span.bank.blocks.as_ptr();
        for step in case["steps"].as_array().unwrap() {
            let cut = step["cut"].as_u64().unwrap();
            operations += 1;
            if step["kind"] == "push" {
                let i = step["index"].as_u64().unwrap();
                let intervals: Vec<_> = step["intervals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap()))
                    .collect();
                let mut r = raw(
                    i,
                    std::array::from_fn(|j| step["values_bits"][j].as_u64().map(f64::from_bits)),
                );
                r.known_samples = intervals.iter().map(|(a, b)| b - a).sum();
                span.push(&r, &intervals, cut).unwrap();
            } else {
                span.gap(cut as f64 / 48000., cut, cut).unwrap();
            }
            if let Some(e) = step.get("query") {
                queries += 1;
                let before = span.bank.blocks.clone();
                let pending = (span.pending, span.gap_pending);
                let counters = (
                    span.insertions,
                    span.bank.merges,
                    span.bank.reconstruction_error,
                    span.last_cut,
                );
                let frozen = span.prefix(cut).unwrap();
                compare(
                    &frozen.blocks,
                    &decode(e["packed"].as_str().unwrap()),
                    &mut mismatches,
                );
                close(
                    frozen.reconstruction_error,
                    f64::from_bits(e["error_bits"].as_u64().unwrap()),
                );
                assert_eq!(
                    frozen.end,
                    e["end_sample"].as_u64().unwrap() as f64 / 48000.
                );
                assert_eq!(frozen.supporting_audio_end, e["audio_end"].as_u64());
                assert_eq!(frozen.available_at, e["available_sample"].as_u64().unwrap());
                assert_eq!(frozen.captured_at, cut);
                assert_eq!(frozen.scales, span.bank.scales);
                assert_eq!(span.bank.blocks, before);
                assert_eq!((span.pending, span.gap_pending), pending);
                assert_eq!(
                    (
                        span.insertions,
                        span.bank.merges,
                        span.bank.reconstruction_error,
                        span.last_cut
                    ),
                    counters
                );
                assert_eq!(span.bank.blocks.as_ptr(), allocation);
                assert!(!span.bank.frozen);
            }
        }
    }
    println!(
        "PREFIX_ORACLE {}",
        serde_json::json!({"cases": fixture["cases"].as_array().unwrap().len(), "operations": operations, "queries":queries,"different_f64_fields":mismatches})
    );
}

#[test]
fn one_knot_prefix_with_trailing_gap_preserves_registered_unknown_tie() {
    let mut span = Span::new(config(8, 2)).unwrap();
    for i in 0..2 {
        let r = raw(i, [Some(2.); 10]);
        span.push(&r, &[(r.start, r.end)], r.available_end).unwrap();
    }
    let reference = span.prefix(1536).unwrap().matching().unwrap();
    span.gap(2048. / 48000., 2048, 2048).unwrap();
    let query = span.prefix(2048).unwrap().matching().unwrap();
    let mut output = matcher::Output::default();
    matcher::dtw(
        &query.knots,
        &reference.knots,
        &matcher::Config {
            anchor: 0,
            band: 16,
            shift: 0.,
            ratio: 1.,
            tempo_shift: 0.,
            scales: [1.; 10],
            insertion: 1.,
            deletion: 1.,
        },
        &mut output,
    )
    .unwrap();
    assert_eq!(output.total, 1.);
    assert_eq!(output.observed, 1);
    assert_eq!(output.valid_coordinates, 0);
    assert_eq!(output.inserted, 1);
    assert_eq!(output.missing, 1);
    println!(
        "PREFIX_SHORT_GAP {}",
        serde_json::json!({"total":output.total,"observed":output.observed,"valid_coordinates":output.valid_coordinates,"inserted":output.inserted,"missing":output.missing,"supported":false})
    );
}

#[test]
fn saved_prefix_matches_and_delivers_after_its_live_owner_advances() {
    use crate::temporal_cognition::transport::{
        Controller, Identity, Packet, RESULT_LIMIT, Ticket,
    };
    let mut original = Span::new(config(8, 2)).unwrap();
    for i in 0..4 {
        let r = raw(i, [Some(1. + i as f64); 10]);
        original
            .push(&r, &[(r.start, r.end)], r.available_end)
            .unwrap();
    }
    let frozen = original.finish(2560).unwrap();
    let stored = memory::Episode {
        identity: Identity {
            id: 1,
            generation: 1,
        },
        epoch: 2,
        available_end: frozen.available_at as f64 / 48000.,
        first_observed_end: frozen.end,
        scales: frozen.scales,
        descriptor: frozen.matching().unwrap(),
    };
    let mut cfg = config(8, 2);
    cfg.first_hop_start = 5120;
    cfg.span_start = 5120. / 48000.;
    let mut live = Span::new(cfg).unwrap();
    for i in 10..14 {
        let r = raw(i, [Some((i - 9) as f64); 10]);
        live.push(&r, &[(r.start, r.end)], r.available_end).unwrap();
    }
    live.gap(8192. / 48000., 8192, 8192).unwrap();
    let query = live.prefix(8192).unwrap();
    let saved = query.packed();
    let r = raw(16, [Some(9.); 10]);
    live.push(&r, &[(r.start, r.end)], r.available_end).unwrap();
    assert_eq!(query.packed(), saved);
    assert!(live.prefix(8192).is_err());
    let descriptor = query.matching().unwrap();
    let report = memory::ordered(
        &memory::Query {
            descriptor: &descriptor,
            epoch: 2,
            end: query.end,
            observed_end: query.captured_at as f64 / 48000.,
            scales: query.scales,
        },
        &[stored],
    )
    .unwrap();
    assert!(report.matches.iter().any(|m| m.relation.supported));
    assert_eq!(query.supporting_audio_end, Some(7168));
    assert_eq!(query.end, 8192. / 48000.);
    let ticket = Ticket {
        bus: query.group.bus,
        epoch: query.group.epoch,
        model_version: 1,
        query: Identity {
            id: 1,
            generation: query.group.generation,
        },
        support_id: 3,
        support_start: 5120,
        support_end: 8192,
        supporting_audio_end: query.supporting_audio_end,
        available_at: query.available_at,
        issued_at: 9216,
        deadline: 12288,
    };
    let mut controller = Controller::new(0, 2, 1, 256);
    controller
        .register(Identity {
            id: 1,
            generation: 1,
        })
        .unwrap();
    controller.dispatch(ticket, 9216).unwrap();
    let mut packet = Packet {
        ticket,
        completed_at: 9728,
        relations: vec![None; RESULT_LIMIT].into_boxed_slice(),
    };
    for (out, matched) in packet.relations.iter_mut().zip(&report.matches) {
        *out = Some(matched.relation);
    }
    let receipt = controller.receive(&packet, 10240).unwrap();
    assert_eq!(
        receipt.supported,
        report
            .matches
            .iter()
            .filter(|m| m.relation.supported)
            .count()
    );
    assert_eq!(receipt.received_at, 10240);
    assert_eq!(receipt.ticket.supporting_audio_end, Some(7168));
    println!(
        "PREFIX_TRANSPORT {}",
        serde_json::json!({"query_end_sample":ticket.support_end,"audio_end_sample":ticket.supporting_audio_end,"captured_at":query.captured_at,"issued_at":ticket.issued_at,"received_at":receipt.received_at,"supported_matches":receipt.supported,"production_worker":false})
    );
}

#[test]
fn live_prefix_copy_closes_pending_without_consuming_or_recompressing_the_owner() {
    let mut cfg = config(4, 2);
    cfg.span_end = Some(20. * 512. / 48000.);
    let mut span = Span::new(cfg).unwrap();
    assert!(span.prefix(512).is_err());
    for i in 0..9 {
        let r = raw(i, [Some(i as f64); 10]);
        span.push(&r, &[(r.start, r.end)], r.available_end).unwrap();
    }
    let before = span.bank.blocks.clone();
    let pending = span.pending;
    let error = span.bank.reconstruction_error;
    let last = span.last;
    let replay = span.last_intervals.clone();
    let cut = span.last_cut;
    let insertions = span.insertions;
    assert!(span.finish(cut).is_err());
    let query = span.prefix(cut).unwrap();
    assert_eq!(query.blocks.len(), 4);
    assert!(query.reconstruction_error > error);
    assert_eq!(query.start, cfg.span_start);
    assert_eq!(query.end, 9. * 512. / 48000.);
    assert_eq!(query.group, cfg.group);
    assert_eq!(query.captured_at, cut);
    assert_eq!(query.supporting_audio_end, Some(9 * 512));
    assert_eq!(query.available_at, 10 * 512);
    assert_eq!(span.bank.blocks, before);
    assert_eq!(span.pending, pending);
    assert_eq!(span.bank.reconstruction_error, error);
    assert_eq!(span.last.unwrap().end, last.unwrap().end);
    assert_eq!(span.last_intervals, replay);
    assert_eq!(span.last_cut, cut);
    assert_eq!(span.insertions, insertions);
    assert_eq!(span.bank.merges, 0);
    assert!(!span.bank.frozen);
    let packed = query.packed();
    assert_eq!(span.prefix(cut).unwrap().packed(), packed);
    let r = raw(9, [Some(9.); 10]);
    span.push(&r, &[(r.start, r.end)], r.available_end).unwrap();
    assert_eq!(query.packed(), packed);
    assert!(span.prefix(cut).is_err());
    close(span.bank.blocks[3].moments[0][1], 8.5);
    assert_eq!(span.bank.merges, 1);
}

#[test]
fn prefix_audio_age_survives_trailing_gaps_and_masked_observations() {
    let mut span = Span::new(config(8, 2)).unwrap();
    let mut a = raw(0, [Some(1.); 10]);
    a.source_end = 4096;
    a.available_end = 4608;
    span.push(&a, &[(0, 512)], 4608).unwrap();
    let b = raw(1, [Some(2.); 10]);
    span.push(&b, &[(512, 1024)], 4608).unwrap();
    assert_eq!(span.prefix(4608).unwrap().supporting_audio_end, Some(4096));
    span.gap(1., 48000, 48000).unwrap();
    let query = span.prefix(48000).unwrap();
    assert_eq!(query.end, 1.);
    assert_eq!(query.supporting_audio_end, Some(4096));
    assert_eq!(query.available_at, 48000);
    assert!(span.gap_pending.is_some());
    assert_eq!(query.blocks.last().unwrap().matching().gap, 1);
    assert!(span.prefix(4608).is_err());

    let mut masked = Span::new(config(8, 2)).unwrap();
    for i in 0..3 {
        let r = raw(i, [None; 10]);
        masked
            .push(&r, &[(r.start, r.end)], r.available_end)
            .unwrap();
    }
    let query = masked.prefix(2048).unwrap();
    assert_eq!(query.supporting_audio_end, None);
    assert!(query.matching().unwrap().knots.iter().all(|k| k.mask == 0));
    let saved = masked.finish(2048).unwrap();
    assert_eq!(masked.prefix(2560).unwrap().packed(), saved.packed());
}

#[test]
fn earlier_blocks_keep_the_latest_availability_cut() {
    let mut span = Span::new(config(8, 1)).unwrap();
    let mut a = raw(0, [Some(1.); 10]);
    a.available_end = 4096;
    span.push(&a, &[(0, 512)], 4096).unwrap();
    let b = raw(1, [Some(2.); 10]);
    span.push(&b, &[(512, 1024)], 4096).unwrap();
    assert!(span.finish(1536).is_err());
    let frozen = span.finish(4096).unwrap();
    assert_eq!(frozen.available_at, 4096);
    let matched = frozen.matching().unwrap();
    assert_eq!(matched.knots[0].available_end, 4096. / 48000.);
    assert_eq!(matched.knots[1].available_end, 1536. / 48000.);
    assert!(matched.local_intervals[1].unwrap() > 0.);
}

#[test]
#[ignore = "release detached prefix/copy/export cost; excludes full O04"]
fn prefix_copy_cost_probe() {
    use crate::temporal_cognition::transport::{Controller, Packet, Receipt, Ticket};
    use std::{hint::black_box, time::Instant};
    for capacity in [64, 128] {
        let mut span = Span::new(config(capacity, 2)).unwrap();
        for i in 0..capacity as u64 * 2 + 1 {
            let r = raw(i, [Some(i as f64); 10]);
            span.push(&r, &[(r.start, r.end)], r.available_end).unwrap();
        }
        assert!(span.pending.is_some());
        let mut copying = Vec::with_capacity(6000);
        let mut exporting = Vec::with_capacity(6000);
        for i in 0..6600 {
            let start = Instant::now();
            let query = black_box(span.prefix(black_box(span.last_cut)).unwrap());
            let copied = start.elapsed().as_secs_f64() * 1e6;
            let start = Instant::now();
            black_box(query.packed());
            black_box(query.matching().unwrap());
            let exported = start.elapsed().as_secs_f64() * 1e6;
            if i >= 600 {
                copying.push(copied);
                exporting.push(exported);
            }
        }
        copying.sort_by(f64::total_cmp);
        exporting.sort_by(f64::total_cmp);
        assert_eq!(span.bank.merges, 0);
        assert!(span.pending.is_some());
        println!(
            "PREFIX_COST {}",
            serde_json::json!({
                "capacity":capacity,"queries":6000,"copy_p99_us":copying[5940],"copy_max_us":copying[5999],
                "exports_p99_us":exporting[5940],"exports_max_us":exporting[5999],
                "copy_bank_payload_bytes":capacity*320,"copy_bank_header_bytes":std::mem::size_of::<Bank>(),
                "frozen_header_bytes":std::mem::size_of::<Frozen>(),"frozen_payload_bytes":capacity*320,
                "packed_export_bytes":capacity*320,"matcher_rows_bytes":capacity*std::mem::size_of::<matcher::Knot>(),
                "matcher_interval_bytes":capacity*std::mem::size_of::<Option<f64>>(),
                "ticket_bytes":std::mem::size_of::<Ticket>(),"packet_header_bytes":std::mem::size_of::<Packet>(),"packet_baseline_64_relations_bytes":std::mem::size_of::<Packet>() + super::super::transport::RESULT_LIMIT*std::mem::size_of::<Option<super::super::transport::Relation>>(),
                "receipt_bytes":std::mem::size_of::<Receipt>(),"controller_bytes":std::mem::size_of::<Controller>(),
                "full_O04":false,"scope":"one full live bank with pending block; copied-bank compression and detached frozen payload; exports include packed bytes and matcher arrays, but no matcher execution, worker, beam, device or allocator/RSS census"
            })
        );
    }
}

#[test]
#[ignore = "release descriptor cadence and compression cost; excludes full O04"]
fn descriptor_compression_cost_probe() {
    use std::{hint::black_box, time::Instant};
    for capacity in [64, 128, 256] {
        let mut span = Span::new(config(capacity, 2)).unwrap();
        let allocation = span.bank.blocks.as_ptr();
        let mut times = Vec::with_capacity(6000);
        let mut close_times = Vec::new();
        for i in 0..6600 {
            let r = raw(
                i,
                std::array::from_fn(|j| Some((i as f64 * 0.19 + j as f64 * 0.3).sin())),
            );
            let start = Instant::now();
            span.push(
                black_box(&r),
                black_box(&[(r.start, r.end)]),
                r.available_end,
            )
            .unwrap();
            if i >= 600 {
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                times.push(elapsed);
                if span.last_insertions > 0 {
                    close_times.push(elapsed);
                }
            }
        }
        let started = Instant::now();
        let frozen = span.finish(6601 * 512).unwrap();
        let freeze_us = started.elapsed().as_secs_f64() * 1e6;
        black_box(frozen);
        times.sort_by(f64::total_cmp);
        close_times.sort_by(f64::total_cmp);
        assert_eq!(allocation, span.bank.blocks.as_ptr());
        println!(
            "descriptor_compression_cost {}",
            serde_json::json!({"capacity":capacity,"calls":6000,"median_us":times[3000],"p99_us":times[5940],"max_us":times[5999],"closing_calls":close_times.len(),"closing_p99_us":close_times[close_times.len()*99/100],"freeze_copy_us":freeze_us,"merges":span.bank.merges,"priority_coordinate_evaluations":span.bank.coordinate_evaluations,"block_bytes":std::mem::size_of::<Block>(),"bank_payload_bytes":capacity*320,"span_header_bytes":std::mem::size_of::<Span>(),"replay_bitmap_payload_bytes":(span.last_intervals.capacity()+span.interval_scratch.capacity())*8,"full_O04":false,"scope":"one retained span, supplied raw descriptors, cadence2 with saturated moment compression; raw extraction,all1024 spans,matching,beam,two workers and device excluded"})
        );
    }
}
