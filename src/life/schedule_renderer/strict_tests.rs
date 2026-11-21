use super::*;
use crate::core::stream::loo::SourceKind;
use crate::life::sound::{BodyKind, ToneAdsr};

const TIME: Timebase = Timebase {
    fs: 48000.,
    hop: 512,
};

fn voice(generation: u32, at: u64, tone_id: u64, routing: Routing) -> PhonationBatch {
    let mut batch = tests::outcome_batch(BodyKind::Sine);
    batch.source_generation = generation;
    batch.routing = routing;
    batch.tones[0].onset = at;
    batch.tones[0].hold_ticks = Some(750);
    batch.tones[0].adsr = Some(ToneAdsr {
        attack_sec: 0.,
        decay_sec: 0.,
        sustain_level: 1.,
        release_sec: 0.01,
    });
    batch.tones[0].tone_id = tone_id;
    if let ToneCmd::On { tone_id: id, .. } = &mut batch.cmds[0] {
        *id = tone_id;
    }
    batch
}

fn source(kind: SourceKind, generation: u32) -> SourcePcm {
    SourcePcm {
        site_id: 0,
        kind,
        id: 2,
        generation,
        analyze_hr: true,
        onset_history: false,
        tonal_context: None,
        habitat: vec![0.; TIME.hop],
    }
}

fn wet(owners: usize) -> ScheduleRenderer {
    let mut r = ScheduleRenderer::new(TIME);
    r.room = RoomBuses::prepare(RoomMode::Both, TIME.hop, owners, false).unwrap();
    r
}

fn reference(dry: &[f32], ir: &[f32], index: usize) -> f64 {
    dry[..=index]
        .iter()
        .rev()
        .zip(ir)
        .map(|(&x, &h)| f64::from(x) * f64::from(h))
        .sum()
}

#[test]
fn solo_history_without_packet_excludes_the_entire_tail() {
    let rhythms = NeuralRhythms::default();
    let mut r = wet(1);
    let mut observer = AcousticTemporalExpectation::new(48000).unwrap();
    r.prepare_self_sound(std::iter::once((2, 7, 0)), 0, &observer);
    let batch = voice(7, 5, 1, Routing::default());
    let mut heard_body = false;
    let mut heard_tail = false;
    for hop in 0..692 {
        let now = hop * 512;
        let frame = r
            .render(
                if hop == 0 {
                    std::slice::from_ref(&batch)
                } else {
                    &[]
                },
                now,
                &rhythms,
            )
            .unwrap();
        observer.process(now, frame.habitat, |_| {});
        let own = r.self_sound[0].as_ref().unwrap();
        assert_eq!(r.buf_habitat, own.habitat);
        heard_body |= own.body.iter().any(|&x| x != 0.);
        if hop > 3 {
            assert!(own.body.iter().all(|&x| x == 0.));
            heard_tail |= r.buf_habitat.iter().any(|&x| x != 0.);
        }
        let through = own.history.observed_through_frame();
        assert_eq!(
            own.history.observed_external_energy(through - 480, through),
            Some([0.; 3])
        );
    }
    assert!(heard_body && heard_tail);
    assert!(r.buf_habitat.iter().all(|&x| x == 0.));
    assert_eq!(r.room.as_ref().unwrap().counts().peak, 4);
}

#[test]
fn rebirth_in_release_shares_one_wet_stem_across_all_three_readers() {
    let rhythms = NeuralRhythms::default();
    let mut r = wet(2);
    let mut prior = ScheduleRenderer::new(TIME);
    let mut other = ScheduleRenderer::new(TIME);
    let mut observer = AcousticTemporalExpectation::new(48000).unwrap();
    let mut reference_history = AcousticTemporalExpectation::new(48000).unwrap();
    let ir = york_response();
    let mut prior_stem = Vec::new();
    let mut new_stem = Vec::new();
    let mut maximum = 0.0f64;
    for hop in 0..7 {
        let now = hop * 512;
        let generation = if hop == 0 { 7 } else { 8 };
        r.prepare_self_sound(std::iter::once((2, generation, 0)), now, &observer);
        let (mut old, mut new) = (Vec::new(), Vec::new());
        if hop == 0 {
            old.push(voice(7, 13, 1, Routing::default()));
        }
        if hop == 1 {
            // Reuse the old tone ID, then overlap a second Tone of this generation.
            new.push(voice(8, now + 3, 1, Routing::default()));
            new.push(voice(8, now + 9, 2, Routing::default()));
        }
        if hop == 3 {
            new.push(voice(
                8,
                now,
                3,
                Routing {
                    to_presentation: true,
                    to_habitat: false,
                },
            ));
        }
        prior_stem.extend_from_slice(prior.render(&old, now, &rhythms).unwrap().habitat);
        new_stem.extend_from_slice(other.render(&new, now, &rhythms).unwrap().habitat);
        old.extend(new);
        r.loo_pcm.clear();
        r.loo_pcm.push(source(SourceKind::Voice, generation));
        if hop == 1 {
            r.loo_pcm.push(source(SourceKind::Voice, 7));
        }
        let mut metabolism = [source(SourceKind::Voice, generation)];
        let frame = r
            .render_with_source_pcm(
                &old,
                now,
                &rhythms,
                &mut metabolism,
                &mut [],
                &mut [],
                |_, _, _, _| {},
            )
            .unwrap();
        let mix = frame.habitat.to_vec();
        let own = r.self_sound.iter().flatten().next().unwrap();
        assert_eq!(own.habitat, metabolism[0].habitat);
        assert_eq!(own.habitat, r.loo_pcm[0].habitat);
        for i in 0..512 {
            let n = now as usize + i;
            let external = mix[i] - own.habitat[i];
            let expected = if hop == 0 {
                0.
            } else {
                reference(&prior_stem, &ir, n)
            };
            maximum = maximum.max((f64::from(external) - expected).abs());
            if hop == 1 {
                maximum = maximum.max(
                    (f64::from(mix[i] - r.loo_pcm[1].habitat[i]) - reference(&new_stem, &ir, n))
                        .abs(),
                );
            }
        }
        assert!(maximum < 1e-5, "{maximum:e}");
        let stats = r.room.as_ref().unwrap().counts();
        assert_eq!(stats.reserved, 5);
        assert_eq!(stats.active, if hop == 1 { 5 } else { 4 });
        assert_eq!(stats.peak, if hop == 0 { 4 } else { 5 });
        // The newborn inherits the completed window and the pending prefix of H.
        let external: Vec<_> = mix.iter().zip(&own.habitat).map(|(h, x)| h - x).collect();
        reference_history.process(now, if hop == 0 { &mix } else { &external }, |_| {});
        if hop > 0 {
            let end = own.history.observed_through_frame();
            let reference = OwnSoundHistory::new(&reference_history, now + 512);
            assert_eq!(
                own.history.observed_external_energy(end - 480, end),
                reference.observed_external_energy(end - 480, end)
            );
        }
        observer.process(now, &mix, |_| {});
    }
    eprintln!("strict rebirth max external PCM error {maximum:e}");
}

#[test]
fn analysis_delivery_gap_keeps_room_tail_and_render_jump_resets_all_buses() {
    let rhythms = NeuralRhythms::default();
    let mut r = wet(1);
    let observer = AcousticTemporalExpectation::new(48000).unwrap();
    r.prepare_self_sound(std::iter::once((2, 7, 0)), 0, &observer);
    let mut sources = [source(SourceKind::Voice, 7)];
    let input = [voice(7, 3, 1, Routing::default())];
    for hop in 0..6 {
        // The continuous local reader advances the room even without a packet.
        if hop == 2 {
            r.loo_pcm.clear();
        }
        if hop == 1 || hop == 4 {
            r.loo_pcm.push(source(SourceKind::Voice, 7));
        }
        let frame = r
            .render_with_source_pcm(
                if hop == 0 { &input } else { &[] },
                hop * 512,
                &rhythms,
                &mut sources,
                &mut [],
                &mut [],
                |_, _, _, _| {},
            )
            .unwrap();
        assert_eq!(frame.habitat, sources[0].habitat);
        assert!(frame.habitat.iter().any(|&x| x != 0.));
        if let Some(loo) = r.loo_pcm.first() {
            assert_eq!(loo.habitat, sources[0].habitat);
        }
    }
    let frame = r
        .render_with_source_pcm(
            &[],
            12 * 512,
            &rhythms,
            &mut sources,
            &mut [],
            &mut [],
            |_, _, _, _| {},
        )
        .unwrap();
    for pcm in [frame.habitat, frame.presentation, &sources[0].habitat] {
        assert!(pcm.iter().all(|&x| x == 0.));
    }
    let own = r.self_sound.iter().flatten().next().unwrap();
    let end = own.history.observed_through_frame();
    assert_eq!(
        own.history.observed_external_energy(end - 480, end),
        Some([0.; 3])
    );
    assert_eq!(r.room.as_ref().unwrap().counts().discontinuities, 1);
}

#[test]
fn opposite_phase_inputs_cancel_on_bus_and_route_off_keeps_the_own_tail() {
    let mut room = RoomBuses::prepare(RoomMode::Both, 512, 2, false)
        .unwrap()
        .unwrap();
    let mut inputs = [source(SourceKind::Voice, 7), source(SourceKind::Flora, 0)];
    let ir = york_response();
    let mut mixture = [0.0; 512];
    let mut wet_mix = [0.0; 512];
    for hop in 0..5 {
        for s in &mut inputs {
            s.habitat.fill(0.);
        }
        if hop == 0 {
            inputs[0].habitat[511] = 0.25;
            inputs[1].habitat[511] = -0.25;
        }
        for (i, sample) in mixture.iter_mut().enumerate() {
            *sample = inputs[0].habitat[i] + inputs[1].habitat[i];
        }
        room.start_hop(hop * 512, 512);
        room.habitat
            .as_mut()
            .unwrap()
            .process(&mixture, &mut wet_mix);
        room.histories
            .bind_readers(hop * 512, &[], &inputs, &[])
            .unwrap();
        room.histories
            .process(hop * 512, &mut [], &mut inputs, &mut [])
            .unwrap();
        for (i, &mixed) in wet_mix.iter().enumerate() {
            let t = hop as usize * 512 + i;
            let tap = t.checked_sub(511).map_or(0., |j| ir[j]);
            assert_eq!(mixed, 0.);
            for (source, other) in inputs.iter().zip([-0.25, 0.25]) {
                assert!(
                    (f64::from(mixed - source.habitat[i]) - other * f64::from(tap)).abs() < 1e-6
                );
            }
        }
    }
    assert!(inputs[0].habitat.iter().any(|&x| x != 0.));
}

#[test]
fn rain_removes_drops_and_preserves_the_ringing_objects() {
    use crate::core::{landscape::Landscape, log2space::Log2Space};
    use crate::life::rain::Rain;
    use crate::scenario::{FieldTarget, RainSpec};
    let landscape = Landscape::new(Log2Space::new(110., 8000., 48));
    let make = || {
        Rain::new(
            2,
            RainSpec {
                rate: 200.,
                objects: 4,
                ring: 1.,
                hit: 1.,
                leave_drops_out: true,
                ..RainSpec::default()
            },
            (300., 6000.),
            FieldTarget::Consonance,
            21,
            &landscape,
            48000.,
            512,
        )
    };
    let mut rain = [make()];
    let mut dry = make();
    let mut renderer = wet(1);
    renderer.loo_pcm.push(source(SourceKind::Rain, 0));
    let ir = york_response();
    let mut objects = Vec::new();
    let mut drops_heard = false;
    for hop in 0..32 {
        let (mut p, mut h) = ([0.; 512], [0.; 512]);
        dry.render_add(&mut p, &mut h);
        objects.extend(
            h.iter()
                .zip(dry.drops_habitat_pcm())
                .map(|(&total, &drops)| total - drops),
        );
        drops_heard |= dry.drops_habitat_pcm().iter().any(|&x| x != 0.);
        let frame = renderer
            .render_with_source_pcm(
                &[],
                hop * 512,
                &NeuralRhythms::default(),
                &mut [],
                &mut [],
                &mut rain,
                |_, _, _, _| {},
            )
            .unwrap();
        let mix = frame.habitat.to_vec();
        let own = &renderer.loo_pcm[0].habitat;
        for i in (0..512).step_by(17) {
            let expected = reference(&objects, &ir, hop as usize * 512 + i);
            assert!((f64::from(mix[i] - own[i]) - expected).abs() < 1e-5);
        }
    }
    assert!(drops_heard && objects.iter().any(|x| x.abs() > 1e-5));
}

#[cfg(feature = "native")]
#[test]
fn onset_tonal_and_hr_consume_the_same_wet_packet_on_the_render_frame() {
    use crate::core::analysis_worker::AnalysisState;
    use crate::core::stream::{analysis::AnalysisStream, loo::PcmPacket};
    use crate::core::tonal_context::Decoder;
    let (kernel, mut params) = crate::runtime::body_metabolism_test_core();
    params.habituation.enabled = true;
    let mut stream = AnalysisStream::new(params, kernel);
    stream.enable_tonal_context(Decoder::unqueried_for_test());
    stream.enable_onset_habituation(0.5);
    let mut silent_reference = AnalysisState::new(stream.clone(), None, None);
    let mut analysis = AnalysisState::new(stream, None, None);
    let mut r = wet(2);
    let mut a = source(SourceKind::Voice, 7);
    a.onset_history = true;
    a.tonal_context = Some(true);
    let mut b = source(SourceKind::Voice, 8);
    b.tonal_context = Some(false);
    r.loo_pcm = vec![a, b];
    let (tx, rx) = crate::handoff::unbounded();
    let mut heard = [false; 3];
    let mut warmed = false;
    for hop in 0..130 {
        let mut command = voice(7, hop * 512, hop, Routing::default());
        command.tones[0].hold_ticks = Some(7000);
        let frame = r
            .render_with_source_pcm(
                if hop == 20 || hop == 60 {
                    std::slice::from_ref(&command)
                } else {
                    &[]
                },
                hop * 512,
                &NeuralRhythms::default(),
                &mut [],
                &mut [],
                &mut [],
                |_, _, _, _| {},
            )
            .unwrap();
        let mix = frame.habitat.to_vec();
        // The renderer still advances during a dropped analysis hop.
        if hop == 85 {
            continue;
        }
        assert_eq!(analysis.begin_hop(hop), hop == 86);
        let head = PcmPacket {
            sites: Vec::new(),
            sources: std::mem::take(&mut r.loo_pcm),
            recycle: tx.clone(),
        };
        assert_eq!(head.sources[0].habitat, mix);
        silent_reference.begin_hop(hop);
        let ready = analysis.observe(hop, &mix, Some(&head));
        assert_eq!(ready, silent_reference.observe(hop, &[0.; 512], None));
        drop(head);
        r.loo_pcm = rx.try_recv().unwrap().0;
        if hop == 86 {
            assert!(!ready);
        }
        if !ready {
            continue;
        }
        if hop > 86 {
            warmed = true;
        }
        let field = analysis.snapshot();
        let external = &field.ecology.loo[0];
        assert!(
            external
                .h_scan
                .iter()
                .chain(&external.r_scan)
                .all(|&x| x == 0.)
        );
        assert!(
            external
                .onset_habituation_state_scan
                .as_ref()
                .unwrap()
                .iter()
                .all(|&x| x == 0.)
        );
        assert!(field.ecology.loo[1].onset_habituation_state_scan.is_none());
        heard[0] |= field.ecology.loo[1].h_scan.iter().any(|&x| x > 0.);
        heard[1] |= field.onset_habituation_state_scan.iter().any(|&x| x > 0.);
        let own = &field.ecology.tonal_observations[0];
        let shared = &field.ecology.tonal_observations[1];
        assert!(own.excluded && !own.shared && !shared.excluded && shared.shared);
        if let Some(excluded) = &own.snapshot {
            let reference = silent_reference.snapshot().ecology.tonal_context.unwrap();
            assert_eq!(excluded.observed, reference.observed);
            assert_eq!(excluded.end, reference.end);
            heard[2] |= field
                .ecology
                .tonal_context
                .as_ref()
                .is_some_and(|mix| mix.observed != reference.observed);
        }
    }
    assert_eq!(heard, [true; 3]);
    assert!(warmed);
    assert_eq!(r.room.as_ref().unwrap().counts().discontinuities, 0);
}

#[cfg(feature = "profile-alloc")]
#[test]
fn strict_room_pool_birth_replacement_and_reader_fanout_allocate_nothing() {
    let mut room = RoomBuses::prepare(RoomMode::Both, 512, 2, false)
        .unwrap()
        .unwrap();
    let mut a = [source(SourceKind::Voice, 0), source(SourceKind::Voice, 1)];
    let mut b = [source(SourceKind::Voice, 0)];
    crate::runtime_profile::begin_allocations();
    for generation in 0..12 {
        a[0].generation = generation;
        a[1].generation = generation + 1;
        b[0].generation = generation;
        for s in a.iter_mut().chain(&mut b) {
            s.habitat.fill(0.);
            s.habitat[511] = 0.125;
        }
        room.start_hop(u64::from(generation) * 512, 512);
        let mut left = [0.; 512];
        room.process_presentation(&a[0].habitat, &mut left);
        room.habitat
            .as_mut()
            .unwrap()
            .process(&a[0].habitat, &mut left);
        room.histories
            .bind_readers(u64::from(generation) * 512, &[], &a, &b)
            .unwrap();
        room.histories
            .process(u64::from(generation) * 512, &mut [], &mut a, &mut b)
            .unwrap();
        assert_eq!(a[0].habitat, b[0].habitat);
    }
    let allocation = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!((allocation.count, allocation.bytes), (0, 0));
}

#[cfg(feature = "profile-alloc")]
#[test]
fn site_capture_and_reflections_reuse_storage() {
    let (score, _) = crate::scripting::ScriptHost::compile_source(
        "place(sine().sustain().leave_self_out(true),at(375).site(1,12)); wait(1);",
        Some(21),
    )
    .unwrap();
    for mode in [RoomMode::Off, RoomMode::Presentation, RoomMode::Both] {
        let mut r = ScheduleRenderer::new(TIME);
        r.sites = SiteBuses::prepare(&score.sites, 512);
        r.sites.as_mut().unwrap().plan(&score, TIME);
        r.room = RoomBuses::prepare(mode, 512, 1, true).unwrap();
        let mut captured = source(SourceKind::Voice, 7);
        captured.site_id = 1;
        r.loo_pcm.push(captured);
        let mut batch = voice(7, 0, 0, Routing::default());
        batch.body_policy = Some(crate::life::action_candidates::PolicySnapshot {
            site_id: 1,
            at: 0,
            is_alive: true,
            gate_allows_onset: true,
        });
        batch.tones[0].hold_ticks = Some(48000);
        r.render(&[batch], 0, &NeuralRhythms::default()).unwrap();
        for n in 1..64 {
            crate::runtime_profile::begin_allocations();
            r.render(&[], n * 512, &NeuralRhythms::default()).unwrap();
            let alloc = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!((alloc.count, alloc.bytes), (0, 0), "{mode:?}, hop {n}");
        }
    }
}

#[test]
fn strict_room_rejects_an_unaddressable_pool_during_preparation() {
    let error = RoomBuses::prepare(RoomMode::Both, TIME.hop, usize::MAX / 4, false)
        .err()
        .unwrap();
    assert!(error.contains("addressable storage"), "{error}");
}
