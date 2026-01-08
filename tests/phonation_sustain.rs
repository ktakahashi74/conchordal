use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::gate_clock::next_gate_tick;
use conchordal::life::individual::{
    AnyArticulationCore, ArticulationWrapper, DroneCore, PhonationBatch, PhonationNoteSpec,
};
use conchordal::life::intent::{BodySnapshot, IntentBoard, IntentKind};
use conchordal::life::phonation_engine::{
    CoreTickCtx, PhonationClock, PhonationCmd, PhonationKick, ThetaGateClock,
};
use conchordal::life::population::Population;
use conchordal::life::scenario::{
    Action, BirthTiming, IndividualConfig, LifeConfig, OnBirthPhonation, PhonationConfig,
};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::sound_voice::default_release_ticks;
use conchordal::life::world_model::WorldModel;

fn setup(tb: Timebase) -> (Population, WorldModel, Landscape) {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let world = WorldModel::new(tb, space.clone());
    let landscape = Landscape::new(space);
    let mut pop = Population::new(tb);
    pop.set_seed(0);
    (pop, world, landscape)
}

fn life_with_phonation(phonation: PhonationConfig) -> LifeConfig {
    let mut life = LifeConfig::default();
    life.phonation = phonation;
    life
}

fn expected_gate_or_fallback(
    tb: &Timebase,
    now: Tick,
    theta: &conchordal::core::modulation::RhythmBand,
) -> Tick {
    let base = now.saturating_add(tb.min_lead_ticks());
    let hop = (tb.hop as Tick).max(1);
    let mut gate_tick =
        next_gate_tick(base, tb.fs, *theta, 0.0).unwrap_or_else(|| tb.ceil_to_hop_tick(base));
    if gate_tick <= now {
        gate_tick = next_gate_tick(base.saturating_add(1), tb.fs, *theta, 0.0)
            .unwrap_or_else(|| tb.ceil_to_hop_tick(base.saturating_add(hop)));
    }
    gate_tick
}

fn publish_at_hop_containing(
    pop: &mut Population,
    world: &mut WorldModel,
    landscape: &Landscape,
    onset: Tick,
    tb: Timebase,
) -> Vec<PhonationBatch> {
    let hop = (tb.hop as Tick).max(1);
    let frame_start = onset - (onset % hop);
    pop.set_current_frame((frame_start / hop) as u64);
    pop.publish_intents(world, landscape, frame_start)
}

fn advance_to_onset_frame(
    pop: &mut Population,
    world: &mut WorldModel,
    landscape: &Landscape,
    onset: Tick,
    tb: Timebase,
) {
    let hop = (tb.hop as Tick).max(1);
    let target_frame_start = onset - (onset % hop);
    let mut frame_start = 0;
    while frame_start < target_frame_start {
        pop.set_current_frame((frame_start / hop) as u64);
        let _ = pop.publish_intents(world, landscape, frame_start);
        frame_start = frame_start.saturating_add(hop);
    }
}

fn expected_gate_index_for_onset(tb: &Timebase, onset: Tick, theta: RhythmBand) -> u64 {
    let hop = (tb.hop as Tick).max(1);
    let mut clock = ThetaGateClock::default();
    let rhythms = NeuralRhythms {
        theta,
        ..NeuralRhythms::default()
    };
    let mut frame_start: Tick = 0;
    loop {
        let frame_end = frame_start.saturating_add(hop);
        let ctx = CoreTickCtx {
            now_tick: frame_start,
            frame_end,
            fs: tb.fs,
            rhythms,
        };
        let mut candidates = Vec::new();
        clock.gather_candidates(&ctx, &mut candidates);
        candidates.sort_by_key(|c| c.tick);
        for candidate in candidates {
            if candidate.tick >= onset {
                return candidate.gate;
            }
        }
        if frame_start >= onset {
            break;
        }
        frame_start = frame_start.saturating_add(hop);
    }
    0
}

fn find_note_on(batches: &[PhonationBatch]) -> Option<(u64, Tick)> {
    for batch in batches {
        for cmd in &batch.cmds {
            if let PhonationCmd::NoteOn { note_id, .. } = cmd {
                let onset = batch
                    .notes
                    .iter()
                    .find(|note| note.note_id == *note_id)
                    .map(|note| note.onset)?;
                return Some((*note_id, onset));
            }
        }
    }
    None
}

fn find_onset_gate(batches: &[PhonationBatch], onset_tick: Tick) -> Option<u64> {
    for batch in batches {
        for onset in &batch.onsets {
            if onset.onset_tick == onset_tick {
                return Some(onset.gate);
            }
        }
    }
    None
}

fn find_note_off(batches: &[PhonationBatch], note_id: u64) -> Option<Tick> {
    for batch in batches {
        for cmd in &batch.cmds {
            if let PhonationCmd::NoteOff {
                note_id: id,
                off_tick,
            } = cmd
            {
                if *id == note_id {
                    return Some(*off_tick);
                }
            }
        }
    }
    None
}

#[test]
fn sustain_gate_emits_note_on_once() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Gate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    advance_to_onset_frame(&mut pop, &mut world, &landscape, expected, tb);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let (note_id, onset) = find_note_on(&batches).expect("note on");
    assert_eq!(onset, expected);
    let expected_gate = expected_gate_index_for_onset(&tb, onset, landscape.rhythm.theta);
    assert_eq!(find_onset_gate(&batches, onset), Some(expected_gate));
    assert!(find_note_off(&batches, note_id).is_none());
    assert!(
        world
            .board
            .query_range(0..u64::MAX)
            .all(|i| i.kind != IntentKind::BirthOnce)
    );

    let hop = (tb.hop as Tick).max(1);
    let next_frame = (expected - (expected % hop)).saturating_add(hop);
    pop.set_current_frame((next_frame / hop) as u64);
    let next_batches = pop.publish_intents(&mut world, &landscape, next_frame);
    assert!(find_note_on(&next_batches).is_none());
    assert!(find_note_off(&next_batches, note_id).is_none());
}

#[test]
fn sustain_immediate_emits_note_on_once() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Immediate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let expected = tb.ceil_to_hop_tick(now.saturating_add(tb.min_lead_ticks()));
    advance_to_onset_frame(&mut pop, &mut world, &landscape, expected, tb);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let (note_id, onset) = find_note_on(&batches).expect("note on");
    assert_eq!(onset, expected);
    let expected_gate = expected_gate_index_for_onset(&tb, onset, landscape.rhythm.theta);
    assert_eq!(find_onset_gate(&batches, onset), Some(expected_gate));
    assert!(find_note_off(&batches, note_id).is_none());

    let hop = (tb.hop as Tick).max(1);
    let next_frame = (expected - (expected % hop)).saturating_add(hop);
    pop.set_current_frame((next_frame / hop) as u64);
    let next_batches = pop.publish_intents(&mut world, &landscape, next_frame);
    assert!(find_note_on(&next_batches).is_none());
    assert!(find_note_off(&next_batches, note_id).is_none());
}

#[test]
fn sustain_death_emits_note_off_once() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Gate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let (note_id, onset) = find_note_on(&batches).expect("note on");

    if let Some(agent) = pop.individuals.first_mut() {
        agent.release_gain = 0.0;
    }
    let hop = (tb.hop as Tick).max(1);
    let next_frame = (expected - (expected % hop)).saturating_add(hop);
    pop.set_current_frame((next_frame / hop) as u64);
    let off_batches = pop.publish_intents(&mut world, &landscape, next_frame);
    let off_tick = find_note_off(&off_batches, note_id).expect("note off");
    assert_eq!(off_tick, next_frame.max(onset));

    let later_frame = next_frame.saturating_add(hop);
    pop.set_current_frame((later_frame / hop) as u64);
    let later_batches = pop.publish_intents(&mut world, &landscape, later_frame);
    assert!(find_note_off(&later_batches, note_id).is_none());
}

#[test]
fn sustain_retains_until_note_off() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Gate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let (note_id, _) = find_note_on(&batches).expect("note on");

    if let Some(agent) = pop.individuals.first_mut() {
        agent.release_gain = 0.0;
        assert!(agent.should_retain());
    }

    let hop = (tb.hop as Tick).max(1);
    let next_frame = (expected - (expected % hop)).saturating_add(hop);
    pop.set_current_frame((next_frame / hop) as u64);
    let off_batches = pop.publish_intents(&mut world, &landscape, next_frame);
    assert!(find_note_off(&off_batches, note_id).is_some());
    if let Some(agent) = pop.individuals.first() {
        assert!(!agent.should_retain());
    }
}

#[test]
fn sustain_shutdown_emits_note_off_and_renderer_idles() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Gate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let onset = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, onset, tb);
    let (note_id, _) = find_note_on(&batches).expect("note on");

    let mut renderer = ScheduleRenderer::new(tb);
    let board = IntentBoard::new(1, 1);
    let rhythms = NeuralRhythms::default();
    let hop = (tb.hop as Tick).max(1);
    let frame_start = onset - (onset % hop);
    renderer.render(&board, &batches, frame_start, &rhythms);
    assert!(!renderer.is_idle());

    let shutdown_tick = frame_start.saturating_add(hop);
    pop.set_current_frame((shutdown_tick / hop) as u64);
    let shutdown_batches = pop.flush_sustain_note_offs(shutdown_tick);
    assert_eq!(
        find_note_off(&shutdown_batches, note_id),
        Some(shutdown_tick)
    );
    renderer.render(&board, &shutdown_batches, shutdown_tick, &rhythms);

    let done_tick = shutdown_tick
        .saturating_add(default_release_ticks(tb))
        .saturating_add(1);
    renderer.render(&board, &[], done_tick, &rhythms);
    assert!(renderer.is_idle());
}

#[test]
fn sustain_remove_emits_note_off_same_frame() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Gate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let onset = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, onset, tb);
    let (note_id, _) = find_note_on(&batches).expect("note on");

    let hop = (tb.hop as Tick).max(1);
    let onset_frame_start = onset - (onset % hop);

    let mut renderer = ScheduleRenderer::new(tb);
    let board = IntentBoard::new(1, 1);
    let rhythms = NeuralRhythms::default();
    renderer.render(&board, &batches, onset_frame_start, &rhythms);

    let remove_frame_start = onset_frame_start.saturating_add(hop);
    pop.set_current_frame((remove_frame_start / hop) as u64);
    pop.remove_agent(1);
    let pending = pop.take_pending_phonation_batches();
    assert_eq!(find_note_off(&pending, note_id), Some(remove_frame_start));
    renderer.render(&board, &pending, remove_frame_start, &rhythms);

    let done_tick = remove_frame_start
        .saturating_add(default_release_ticks(tb))
        .saturating_add(1);
    renderer.render(&board, &[], done_tick, &rhythms);
    assert!(renderer.is_idle());
}

#[test]
fn sustain_skips_note_on_when_no_spec() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1000.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: life_with_phonation(PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            timing: BirthTiming::Gate,
            ..Default::default()
        }),
        tag: None,
    };
    pop.apply_action(
        Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );
    if let Some(agent) = pop.individuals.first_mut() {
        agent.release_gain = 0.0;
    }

    let now: Tick = 0;
    let onset = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    let batches = publish_at_hop_containing(&mut pop, &mut world, &landscape, onset, tb);
    assert!(find_note_on(&batches).is_none());
    if let Some(agent) = pop.individuals.first() {
        assert!(!agent.birth_pending());
        assert!(agent.sustain_note_id.is_none());
    }
    let hop = (tb.hop as Tick).max(1);
    let next_frame = (onset - (onset % hop)).saturating_add(hop);
    pop.set_current_frame((next_frame / hop) as u64);
    let next_batches = pop.publish_intents(&mut world, &landscape, next_frame);
    assert!(find_note_on(&next_batches).is_none());
}

#[test]
fn sustain_hold_ticks_max_does_not_expire() {
    let tb = Timebase {
        fs: 1000.0,
        hop: 16,
    };
    let mut renderer = ScheduleRenderer::new(tb);
    let board = IntentBoard::new(1, 1);
    let rhythms = NeuralRhythms::default();

    let note_id = 1;
    let onset: Tick = 0;
    let note = PhonationNoteSpec {
        note_id,
        onset,
        hold_ticks: Some(Tick::MAX),
        freq_hz: 440.0,
        amp: 0.5,
        smoothing_tau_sec: 0.0,
        body: BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        },
        articulation: ArticulationWrapper::new(
            AnyArticulationCore::Drone(DroneCore {
                phase: 0.0,
                sway_rate: 0.1,
            }),
            1.0,
        ),
    };
    let batch_on = PhonationBatch {
        source_id: 1,
        cmds: vec![PhonationCmd::NoteOn {
            note_id,
            kick: PhonationKick::Birth,
        }],
        notes: vec![note],
        onsets: Vec::new(),
    };
    renderer.render(&board, &[batch_on], onset, &rhythms);
    assert!(!renderer.is_idle());

    let far_tick = tb.sec_to_tick(120.0);
    renderer.render(&board, &[], far_tick, &rhythms);
    assert!(!renderer.is_idle());

    let batch_off = PhonationBatch {
        source_id: 1,
        cmds: vec![PhonationCmd::NoteOff {
            note_id,
            off_tick: far_tick,
        }],
        notes: Vec::new(),
        onsets: Vec::new(),
    };
    renderer.render(&board, &[batch_off], far_tick, &rhythms);
    let done_tick = far_tick
        .saturating_add(default_release_ticks(tb))
        .saturating_add(1);
    renderer.render(&board, &[], done_tick, &rhythms);
    assert!(renderer.is_idle());
}

#[test]
fn sustain_voice_releases_after_note_off() {
    let tb = Timebase {
        fs: 1000.0,
        hop: 16,
    };
    let mut renderer = ScheduleRenderer::new(tb);
    let board = IntentBoard::new(1, 1);
    let rhythms = NeuralRhythms::default();

    let note_id = 1;
    let onset: Tick = 0;
    let note = PhonationNoteSpec {
        note_id,
        onset,
        hold_ticks: Some(Tick::MAX),
        freq_hz: 440.0,
        amp: 0.5,
        smoothing_tau_sec: 0.0,
        body: BodySnapshot {
            kind: "sine".to_string(),
            amp_scale: 1.0,
            brightness: 0.0,
            noise_mix: 0.0,
        },
        articulation: ArticulationWrapper::new(
            AnyArticulationCore::Drone(DroneCore {
                phase: 0.0,
                sway_rate: 0.1,
            }),
            1.0,
        ),
    };
    let batch_on = PhonationBatch {
        source_id: 1,
        cmds: vec![PhonationCmd::NoteOn {
            note_id,
            kick: PhonationKick::Birth,
        }],
        notes: vec![note],
        onsets: Vec::new(),
    };
    renderer.render(&board, &[batch_on], onset, &rhythms);
    assert!(!renderer.is_idle());

    let off_tick = tb.hop as Tick;
    let batch_off = PhonationBatch {
        source_id: 1,
        cmds: vec![PhonationCmd::NoteOff { note_id, off_tick }],
        notes: Vec::new(),
        onsets: Vec::new(),
    };
    renderer.render(&board, &[batch_off], off_tick, &rhythms);

    let done_tick = off_tick
        .saturating_add(default_release_ticks(tb))
        .saturating_add(1);
    renderer.render(&board, &[], done_tick, &rhythms);
    assert!(renderer.is_idle());
}
