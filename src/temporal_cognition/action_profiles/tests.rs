use super::*;
use crate::config::{TemporalBodyConfig, TemporalBodyMedoid, TemporalBodyPrototypesConfig};

fn fixture() -> (AppConfig, serde_json::Value, Vec<u8>) {
    let mut config = AppConfig::default();
    config.audio.sample_rate = 48_000;
    config.analysis.nfft = 2048;
    config.analysis.hop_size = 512;
    config.temporal_body = Some(TemporalBodyConfig {
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
    });
    config.temporal_body_prototypes = Some(TemporalBodyPrototypesConfig {
        model_version: "ab".repeat(32),
        sample_rate: 48_000,
        nfft: 2048,
        hop_size: 512,
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
        medoids: vec![TemporalBodyMedoid {
            record_id: "fixed-body".into(),
            raw_values: [0.; 6],
            mask: 63,
        }],
    });
    let slots: [[Option<usize>; 32]; 7] = std::array::from_fn(|class| {
        std::array::from_fn(|cell| {
            let supported = match class {
                0 | 3 | 4 => cell == 0,
                1 | 2 => cell > 0,
                _ => true,
            };
            supported.then_some(0)
        })
    });
    let header = serde_json::json!({
        "schema": "i10-action-profiles-v1", "body_model_version": "ab".repeat(32),
        "sample_rate": 48000, "nfft": 2048, "hop_samples": 512,
        "horizon_samples": 192000, "frames_per_trajectory": 375, "trajectory_count": 1,
        "classes": NAMES, "action_offsets": (0..32).map(|i| (i * 192000 + 15) / 31).collect::<Vec<_>>(),
        "profiles": [{ "record_id": "fixed-body", "source_bus": 0,
            "recipe": { "routing": "both" }, "trajectories": slots,
            "source_descriptor": { "bus": 0, "mask": 63, "raw_values": ([0.; 6]) } }],
    });
    let mut payload = Vec::new();
    for _ in 0..375 {
        payload.extend(2046_u32.to_le_bytes()); // Unknown centroid, known zero decline.
        payload.extend(0_u32.to_le_bytes());
        for value in [0_f64, 0.5, 1., 0., 0., 0.125, 1., 0.1, 0.2, 0.3, 4.] {
            payload.extend(value.to_le_bytes());
        }
    }
    (config, header, payload)
}

fn bytes(header: &serde_json::Value, payload: &[u8]) -> Vec<u8> {
    let header = serde_json::to_vec(header).unwrap();
    let mut bytes = b"I10AP001".to_vec();
    bytes.extend((header.len() as u32).to_le_bytes());
    bytes.extend(1_u32.to_le_bytes());
    bytes.extend(header);
    bytes.extend(payload);
    bytes
}

fn decode(config: &AppConfig, bytes: &[u8]) -> Result<Profiles> {
    Profiles::decode(
        config,
        &TemporalActionProfilesConfig {
            file: "unused".into(),
            evaluation_delay_ms: None,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        },
        bytes,
    )
}

pub(in crate::temporal_cognition) fn model() -> Profiles {
    let (config, header, payload) = fixture();
    decode(&config, &bytes(&header, &payload)).unwrap()
}

pub(in crate::temporal_cognition) fn distinct_release_model() -> Profiles {
    let mut profiles = model();
    let mut frames = profiles.frames.to_vec();
    for mut frame in profiles.frames.iter().copied() {
        frame.mask &= !(1 << 1);
        frame.values[1] = 0.;
        frame.values[10] = 0.5;
        frames.push(frame);
    }
    profiles.frames = frames.into_boxed_slice();
    profiles.profiles[0].trajectories[5] = [Some(1); 32];
    profiles
}

pub(in crate::temporal_cognition) fn assert_table_matches_direct(
    table: &Table,
    profiles: &Profiles,
    context: &context::Context,
    gesture: &super::super::gesture::Gesture,
) {
    let snapshot = table.snapshot.unwrap();
    for (prototype, group) in snapshot.groups.iter().enumerate().take(profiles.count()) {
        let Some(group) = *group else { continue };
        for (class_index, class) in CLASSES.into_iter().enumerate() {
            for (index, offset) in profiles.offsets().iter().copied().enumerate() {
                let relative = profiles.evaluation_delay_samples.map_or(offset, |delay| {
                    let end = offset + delay;
                    end + (512 - end % 512) % 512
                });
                let direct = (relative <= 192_000)
                    .then(|| {
                        context.project_action_window(
                            profiles,
                            prototype,
                            class,
                            offset,
                            snapshot.issued_at + relative,
                            group,
                            gesture.rms_reference(),
                            Some(gesture),
                            snapshot
                                .arrival_issues
                                .iter()
                                .flatten()
                                .find(|a| a.group == group),
                            None,
                        )
                    })
                    .flatten();
                assert_eq!(
                    serde_json::to_vec(&table.cells[(prototype * 7 + class_index) * 32 + index])
                        .unwrap(),
                    serde_json::to_vec(&direct).unwrap(),
                    "prototype {prototype} class {class_index} offset {offset}",
                );
            }
        }
    }
    let resources = table.resources();
    assert!(resources.reused_cells > 0);
    assert!(resources.articulation_frames > 0);
    assert!(resources.reused_articulation_frames > 0);
    assert_eq!(
        resources.attempted_cells,
        resources.projection_calls
            + resources.reused_cells
            + resources.evaluation_unavailable_cells
    );
}

#[test]
fn evaluation_timing_config_preserves_omitted_policy_and_rejects_invalid_delays() {
    let (mut config, header, payload) = fixture();
    let bytes = bytes(&header, &payload);
    let text = format!("file = 'unused'\nsha256 = '{:x}'\n", Sha256::digest(&bytes));
    let omitted: TemporalActionProfilesConfig = toml::from_str(&text).unwrap();
    assert_eq!(omitted.evaluation_delay_ms, None);
    assert!(
        !toml::to_string(&omitted)
            .unwrap()
            .contains("evaluation_delay")
    );
    for delay in [0, 250, 1000, 4000, 4001, u32::MAX] {
        let spec: TemporalActionProfilesConfig =
            toml::from_str(&format!("{text}evaluation_delay_ms = {delay}\n")).unwrap();
        config.temporal_action_profiles = Some(spec.clone());
        assert_eq!(validate_config(&config).is_ok(), delay <= 4000);
        let decoded = Profiles::decode(&config, &spec, &bytes);
        assert_eq!(decoded.is_ok(), delay <= 4000);
        if let Ok(profiles) = decoded {
            assert_eq!(
                profiles.evaluation_delay_samples,
                Some(u64::from(delay) * 48)
            );
        }
    }
    for delay in ["-1", "0.5", "'250'", "4294967296"] {
        assert!(
            toml::from_str::<TemporalActionProfilesConfig>(&format!(
                "{text}evaluation_delay_ms = {delay}\n"
            ))
            .is_err()
        );
    }
}

#[test]
fn evaluation_timing_uses_issue_relative_complete_hops_without_horizon_clipping() {
    let mut profiles = model();
    for issue in [0, 1, 511, 5632, 1 << 52] {
        for (delay, supported) in [
            (None, 32),
            (Some(0), 32),
            (Some(12000), 30),
            (Some(48000), 24),
            (Some(192000), 1),
        ] {
            profiles.evaluation_delay_samples = delay;
            let mut count = 0;
            for offset in profiles.offsets {
                let raw = offset + delay.unwrap_or(0);
                let relative = if delay.is_none() {
                    raw
                } else {
                    raw + (512 - raw % 512) % 512
                };
                let expected = (relative <= 192000).then_some(issue + relative);
                assert_eq!(profiles.evaluation_at(issue, offset), expected);
                count += usize::from(expected.is_some());
            }
            assert_eq!(count, supported);
        }
    }
    profiles.evaluation_delay_samples = None;
    assert_eq!(profiles.evaluation_at(u64::MAX, 0), Some(u64::MAX));
    assert_eq!(profiles.evaluation_at(u64::MAX, 1), None);
    assert_eq!(profiles.evaluation_at(0, 192001), None);
    profiles.evaluation_delay_samples = Some(12000);
    assert_eq!(profiles.evaluation_at(1, 0), Some(12289));
    assert_eq!(profiles.evaluation_at(u64::MAX - 12000, 0), None);
    assert_eq!(profiles.evaluation_at(0, u64::MAX), None);
}

#[test]
fn switching_evaluation_timing_rebuilds_frozen_table_without_mutating_observations() {
    use crate::temporal_cognition::{context, gesture};
    let mut profiles = distinct_release_model();
    let mut observed = context::Context::new(1, 0, 0, 48000, 512).unwrap();
    let mut gesture = gesture::Gesture::new(
        1,
        0,
        48000,
        512,
        crate::config::TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        },
    )
    .unwrap();
    let mut input = context::tests::input(1, 0.5);
    input.assignment.end_sample = 512;
    input.correlation_window = [0, 512];
    let raw = &mut input.features[0].as_mut().unwrap().raw;
    raw.start = 0;
    raw.end = 512;
    raw.known_samples = 512;
    raw.source_start = 0;
    raw.source_end = 512;
    raw.available_end = 512;
    gesture
        .advance(&input, &context::tests::ridges(), 512)
        .unwrap();
    observed.advance(&input, None, 512).unwrap();
    let mut shared = body_model::Shared {
        model_version: profiles.body_model_version,
        bus: 1,
        epoch: 0,
        end_sample: 512,
        descriptors: [None; 7],
        assignments: [None; 8],
    };
    shared.assignments[0] = Some(body_model::Assignment {
        key: (0, input.group_handles[0].unwrap().generation),
        distance: 0.,
        common_coordinates: 6,
    });
    let before = serde_json::to_value(observed.snapshot()).unwrap();
    let mut table = Table::new();
    let mut frozen_publications = Vec::new();
    let binding = consumer::Binding {
        source_id: 7,
        source_generation: 2,
        body_generation: 0,
        bus: 1,
        end: 512,
        available: 512,
        model_version: profiles.body_model_version,
        prototype: 0,
        distance: 0.,
        common_coordinates: 6,
    };
    let input = crate::life::action_candidates::Input {
        class: Class::OnsetNow,
        issued_at: 512,
        at: 512,
        excitation_at: Some(512),
        release_at: None,
        reconsider_at: None,
        consumes_due_opportunity: false,
        withhold_until: None,
    };
    for (build, (delay, cells, unavailable)) in [
        (None, 129, 0),
        (Some(0), 129, 0),
        (Some(12000), 121, 14),
        (Some(48000), 97, 56),
        (None, 129, 0),
    ]
    .into_iter()
    .enumerate()
    {
        profiles.evaluation_delay_samples = delay;
        let old_unavailable = table.resources().evaluation_unavailable_cells;
        let snapshot = table
            .refresh(&profiles, &observed, &shared, &gesture, None)
            .unwrap();
        let published = table.publication(&profiles).unwrap();
        assert_eq!(published.key, consumer::Key::from(&snapshot));
        assert_eq!(published.key.lookup_version, 3);
        assert_eq!(published.key.routed_prototypes, 1);
        assert_eq!(published.key.profile_sha256, snapshot.profile_sha256);
        assert_eq!(
            published.key.body_model_version,
            snapshot.body_model_version
        );
        assert_eq!(published.key.bus, snapshot.bus);
        assert_eq!(published.key.epoch, snapshot.epoch);
        assert_eq!(published.key.issued_at, snapshot.issued_at);
        assert_eq!(published.key.assignment_end, snapshot.assignment_end);
        assert_eq!(
            published.key.evaluation_delay_samples,
            snapshot.evaluation_delay_samples
        );
        assert_eq!(published.key.groups, snapshot.groups);
        profiles.sha256[0] ^= 1;
        assert!(table.publication(&profiles).is_none());
        profiles.sha256[0] ^= 1;
        profiles.body_model_version[0] ^= 1;
        assert!(table.publication(&profiles).is_none());
        profiles.body_model_version[0] ^= 1;
        profiles.profiles[0].source_routed = false;
        assert!(table.publication(&profiles).is_none());
        let mut unrouted_table = Table::new();
        unrouted_table
            .refresh(&profiles, &observed, &shared, &gesture, None)
            .unwrap();
        let unrouted = unrouted_table.publication(&profiles).unwrap();
        assert_eq!(unrouted.key.routed_prototypes, 0);
        assert!(matches!(
            unrouted.pair(Some(binding), (7, 2, Some(0)), 1, 512, input, input),
            consumer::Pair::UnroutedPrototype
        ));
        profiles.profiles[0].source_routed = true;
        let shifted = crate::life::action_candidates::Input {
            class: Class::Continue,
            issued_at: 512 + profiles.offsets[1],
            at: 512 + profiles.offsets[1],
            excitation_at: None,
            ..input
        };
        let consumer::Pair::Paired { body_default, .. } = published.pair(
            Some(binding),
            (7, 2, Some(0)),
            1,
            shifted.issued_at,
            shifted,
            shifted,
        ) else {
            panic!("verified control identity enables shifted continuation")
        };
        assert_eq!(body_default.class, Class::Wait);
        assert_eq!(
            serde_json::to_value(body_default).unwrap(),
            serde_json::to_value(consumer::Entry::from(
                table.cells[2 * 32 + 1].as_ref().unwrap()
            ))
            .unwrap()
        );
        let original_wait = profiles.profiles[0].trajectories[2][1].take();
        let incompatible = table.publication(&profiles).unwrap();
        assert!(matches!(
            incompatible.pair(
                Some(binding),
                (7, 2, Some(0)),
                1,
                shifted.issued_at,
                shifted,
                shifted
            ),
            consumer::Pair::UnknownCell
        ));
        profiles.profiles[0].trajectories[2][1] = original_wait;

        assert!(published.bytes() < snapshot.table_bytes / 4);
        let result = published.pair(Some(binding), (7, 2, Some(0)), 1, 512, input, input);
        let saved = serde_json::to_value(result).unwrap();
        frozen_publications.push((published, saved));
        for (old, expected) in &frozen_publications {
            assert_eq!(
                serde_json::to_value(old.pair(
                    Some(binding),
                    (7, 2, Some(0)),
                    1,
                    512,
                    input,
                    input
                ))
                .unwrap(),
                *expected
            );
        }
        assert_eq!(snapshot.cells, cells);
        assert_eq!(snapshot.evaluation_delay_samples, delay);
        assert_eq!(table.resources().rebuilds, build as u64 + 1);
        assert_eq!(table.resources().invalidations, build as u64);
        assert_eq!(table.resources().deferred_calls, 0);
        assert_eq!(
            table.resources().evaluation_unavailable_cells - old_unavailable,
            unavailable
        );
        assert_table_matches_direct(&table, &profiles, &observed, &gesture);
        for cell in table.cells.iter().flatten() {
            assert_eq!(cell.issued_at, 512);
            assert!(cell.action_at <= cell.evaluation_at);
            assert!(cell.evaluation_at <= 512 + 192000);
            if let Some(delay) = delay {
                assert!(cell.evaluation_at - cell.action_at >= delay);
                assert!(cell.evaluation_at - cell.action_at < delay + 512);
                assert_eq!((cell.evaluation_at - cell.issued_at) % 512, 0);
            } else {
                assert_eq!(cell.action_at, cell.evaluation_at);
            }
        }
        let repeat = table
            .refresh(&profiles, &observed, &shared, &gesture, None)
            .unwrap();
        assert_eq!(
            serde_json::to_value(snapshot).unwrap(),
            serde_json::to_value(repeat).unwrap()
        );
        assert_eq!(table.resources().rebuilds, build as u64 + 1);
    }
    assert_eq!(before, serde_json::to_value(observed.snapshot()).unwrap());
}

#[test]
fn loader_requires_the_original_acquisition_route() {
    let (config, header, payload) = fixture();
    for bus in 0..2 {
        for route in ["both", "habitat", "presentation"] {
            let mut header = header.clone();
            header["profiles"][0]["source_bus"] = bus.into();
            header["profiles"][0]["source_descriptor"]["bus"] = bus.into();
            header["profiles"][0]["recipe"]["routing"] = route.into();
            let profiles = decode(&config, &bytes(&header, &payload)).unwrap();
            assert_eq!(
                profiles.profiles[0].source_routed,
                route == "both"
                    || (bus == 0 && route == "habitat")
                    || (bus == 1 && route == "presentation")
            );
        }
    }
    for fault in 0..7 {
        let mut bad = header.clone();
        match fault {
            0 => bad["profiles"][0]["source_bus"] = serde_json::Value::Null,
            1 => bad["profiles"][0]["source_bus"] = 2.into(),
            2 => bad["profiles"][0]["source_bus"] = "0".into(),
            3 => bad["profiles"][0]["source_descriptor"]["bus"] = 1.into(),
            4 => bad["profiles"][0]["source_descriptor"]["bus"] = serde_json::Value::Null,
            5 => bad["profiles"][0]["recipe"]["routing"] = serde_json::Value::Null,
            _ => bad["profiles"][0]["recipe"]["routing"] = "unregistered".into(),
        }
        assert!(
            decode(&config, &bytes(&bad, &payload)).is_err(),
            "route fault {fault}"
        );
    }
}

#[test]
fn loader_rejects_wrong_identity_grid_bindings_and_payload() {
    let (config, header, payload) = fixture();
    for fault in 0..14 {
        let mut bad = header.clone();
        match fault {
            0 => bad["schema"] = "unregistered".into(),
            1 => bad["body_model_version"] = "cd".repeat(32).into(),
            2 => bad["sample_rate"] = 44100.into(),
            3 => bad["nfft"] = 4096.into(),
            4 => bad["hop_samples"] = 480.into(),
            5 => bad["action_offsets"][1] = 1.into(),
            6 => bad["profiles"][0]["record_id"] = "other-body".into(),
            7 => bad["profiles"][0]["trajectories"][0][1] = 0.into(),
            8 => bad["profiles"][0]["trajectories"][4][0] = 1.into(),
            9 => bad["classes"][0] = "continue".into(),
            10 => bad["profiles"][0]["source_descriptor"]["mask"] = 44.into(),
            11 => bad["profiles"][0]["source_descriptor"]["raw_values"][0] = 0.1.into(),
            12 => bad["routing_projection"] = "routed_body".into(),
            _ => bad["profiles"] = serde_json::json!([]),
        }
        assert!(
            decode(&config, &bytes(&bad, &payload)).is_err(),
            "header fault {fault}"
        );
    }
    for fault in 0..6 {
        let mut bad = payload.clone();
        match fault {
            0 => {
                bad.pop();
            }
            1 => bad[4] = 1,
            2 => bad[1] |= 8,
            3 => bad[24..32].copy_from_slice(&f64::NAN.to_le_bytes()),
            4 => bad[8..16].copy_from_slice(&1_f64.to_le_bytes()),
            _ => bad[56..64].copy_from_slice(&1.1_f64.to_le_bytes()),
        }
        assert!(
            decode(&config, &bytes(&header, &bad)).is_err(),
            "payload fault {fault}"
        );
    }
    let good = bytes(&header, &payload);
    assert!(
        Profiles::decode(
            &config,
            &TemporalActionProfilesConfig {
                file: "unused".into(),
                evaluation_delay_ms: None,
                sha256: "00".repeat(32),
            },
            &good
        )
        .is_err()
    );
    let mut bad_config = config.clone();
    bad_config
        .temporal_body_prototypes
        .as_mut()
        .unwrap()
        .model_version = "a".into();
    assert!(decode(&bad_config, &good).is_err());
}

#[test]
fn projection_preserves_unknown_zero_and_separates_observation_clocks() {
    let profiles = model();
    let previous = window::Frame {
        start: 512,
        end: 1024,
        source_end: 1024,
        available: 1024,
        raw: [Feature::Observed(0.); 10],
        energy: Feature::Observed(1.),
    };
    let frames: Vec<_> = profiles
        .frames(0, Class::Continue, 0, 1024, Some(previous), Some(12.))
        .unwrap()
        .collect();
    assert_eq!(frames.len(), 375);
    assert_eq!((frames[0].start, frames[374].end), (1024, 193024));
    assert_eq!(frames[0].raw[0], Feature::Unsupported);
    assert_eq!(frames[0].raw[3], Feature::Projected(1.));
    assert_eq!(frames[0].raw[4], Feature::Projected(0.));
    assert_eq!(frames[0].raw[5], Feature::Unsupported);
    assert_eq!(frames[1].raw[5], Feature::Projected(0.125));
    assert_eq!(frames[0].raw[6], Feature::Projected(0.25));
    assert_eq!(frames[0].energy, Feature::Projected(4.));
    let summary = window::summarize(
        512,
        2048,
        1024,
        48000,
        1.,
        std::iter::once(previous).chain(frames.into_iter()),
    )
    .unwrap();
    assert_eq!(summary.values[0], Feature::Projected(1. / 3.));
    assert_eq!(summary.values[2], Feature::Unsupported);
    assert_eq!(summary.observed_fraction[0], 1. / 3.);
    assert_eq!(summary.projected_fraction[0], 2. / 3.);
    for previous in [
        None,
        Some(window::Frame {
            available: 1025,
            ..previous
        }),
        Some(window::Frame {
            end: 1023,
            ..previous
        }),
        Some(window::Frame {
            raw: [Feature::Projected(0.); 10],
            ..previous
        }),
    ] {
        let first = profiles
            .frames(0, Class::Continue, 0, 1024, previous, None)
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(first.raw[3], Feature::Unsupported);
        assert_eq!(first.raw[4], Feature::Unsupported);
        assert_eq!(first.raw[6], Feature::Unsupported);
    }
    assert!(
        profiles
            .frames(0, Class::Continue, 1, 1024, None, None)
            .is_none()
    );
    assert!(
        profiles
            .frames(0, Class::Wait, 0, 1024, None, None)
            .is_none()
    );
    assert!(
        profiles
            .frames(1, Class::Continue, 0, 1024, None, None)
            .is_none()
    );
    assert!(
        profiles
            .frames(0, Class::Continue, 0, u64::MAX, None, None)
            .is_none()
    );
    assert!(
        profiles
            .frames(0, Class::Continue, 0, 1024, None, Some(f64::NAN))
            .is_none()
    );
}

#[test]
fn missing_binding_and_silent_energy_are_not_interpolated_or_invented() {
    let (config, mut header, mut payload) = fixture();
    header["profiles"][0]["trajectories"][6][1] = serde_json::Value::Null;
    payload[88..96].copy_from_slice(&0_f64.to_le_bytes());
    let profiles = decode(&config, &bytes(&header, &payload)).unwrap();
    assert!(
        profiles
            .frames(0, Class::Gap, 6194, 1024, None, None)
            .is_none()
    );
    let first = profiles
        .frames(0, Class::Continue, 0, 1024, None, Some(0.))
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(first.energy, Feature::Projected(0.));
    assert_eq!(first.raw[6], Feature::Projected(0.));
    payload[88..96].copy_from_slice(&f64::MAX.to_le_bytes());
    let profiles = decode(&config, &bytes(&header, &payload)).unwrap();
    let first = profiles
        .frames(0, Class::Continue, 0, 1024, None, Some(f64::MAX))
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(first.raw[6], Feature::Unsupported);
}
