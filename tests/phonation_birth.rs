use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::gate_clock::next_gate_tick;
use conchordal::life::intent::IntentKind;
use conchordal::life::population::Population;
use conchordal::life::scenario::{
    BirthTiming, IndividualConfig, LifeConfig, OnBirthPhonation, PhonationConfig,
};
use conchordal::life::sound_voice::SoundVoice;
use conchordal::life::world_model::WorldModel;

fn setup(tb: Timebase) -> (Population, WorldModel, Landscape) {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let world = WorldModel::new(tb, space.clone());
    let landscape = Landscape::new(space);
    let pop = Population::new(tb);
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
) {
    let hop = (tb.hop as Tick).max(1);
    let frame_start = onset - (onset % hop);
    let frame_end = frame_start.saturating_add(hop);
    pop.set_current_frame((frame_start / hop) as u64);
    pop.publish_intents(world, landscape, frame_start, frame_end);
}

fn find_birth_intent(world: &WorldModel) -> Option<conchordal::life::intent::Intent> {
    world
        .board
        .query_range(0..u64::MAX)
        .find(|i| i.kind == IntentKind::BirthOnce)
        .cloned()
}

#[test]
fn default_phonation_birth_fires_on_gate() {
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
        amp: 0.0,
        life: LifeConfig::default(),
        tag: None,
    };
    pop.apply_action(
        conchordal::life::scenario::Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let intent = find_birth_intent(&world).expect("expected BirthOnce intent");
    assert_eq!(intent.kind, IntentKind::BirthOnce);
    assert!(intent.onset > now);
    assert_eq!(intent.onset, expected);
}

#[test]
fn phonation_immediate_fires_next_hop() {
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
            on_birth: OnBirthPhonation::Once,
            timing: BirthTiming::Immediate,
        }),
        tag: None,
    };
    pop.apply_action(
        conchordal::life::scenario::Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now0: Tick = 0;
    let expected = tb.ceil_to_hop_tick(now0.saturating_add(tb.min_lead_ticks()));
    publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let intent = find_birth_intent(&world).expect("expected BirthOnce intent");
    assert_eq!(intent.kind, IntentKind::BirthOnce);
    assert_eq!(intent.onset, expected);
}

#[test]
fn phonation_off_skips_birth() {
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
            on_birth: OnBirthPhonation::Off,
            timing: BirthTiming::Gate,
        }),
        tag: None,
    };
    pop.apply_action(
        conchordal::life::scenario::Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );
    let now: Tick = 0;
    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    assert!(
        world
            .board
            .query_range(0..u64::MAX)
            .all(|i| i.kind != IntentKind::BirthOnce)
    );
}

#[test]
fn birth_pending_prevents_prune_before_onset() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let (mut pop, mut world, mut landscape) = setup(tb);
    pop.set_current_frame(0);
    landscape.rhythm.theta.freq_hz = 1.0;
    landscape.rhythm.theta.phase = 0.0;

    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.0,
        life: LifeConfig::default(),
        tag: None,
    };
    pop.apply_action(
        conchordal::life::scenario::Action::AddAgent {
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
    let hop = (tb.hop as Tick).max(1);
    pop.set_current_frame(0);
    pop.publish_intents(&mut world, &landscape, 0, hop);
    assert_eq!(pop.individuals.len(), 1);
    assert!(find_birth_intent(&world).is_none());

    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    assert!(find_birth_intent(&world).is_some());
}

#[test]
fn birth_kick_fires_on_onset() {
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
        life: LifeConfig::default(),
        tag: None,
    };
    pop.apply_action(
        conchordal::life::scenario::Action::AddAgent {
            id: 1,
            agent: agent_cfg,
        },
        &landscape,
        None,
    );

    let now: Tick = 0;
    let expected = expected_gate_or_fallback(&tb, now, &landscape.rhythm.theta);
    publish_at_hop_containing(&mut pop, &mut world, &landscape, expected, tb);
    let intent = find_birth_intent(&world).expect("expected BirthOnce intent");
    assert!(
        intent.articulation.is_some(),
        "BirthOnce intent must carry articulation for kick"
    );
    let onset = intent.onset;
    let mut voice = SoundVoice::from_intent(tb, intent).expect("voice");
    let rhythms = NeuralRhythms::default();
    let dt = 1.0 / tb.fs;
    assert!(voice.kick_if_due(onset, &rhythms, dt));
}
