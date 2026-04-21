use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::Timebase;
use conchordal::life::control::VoiceControl;
use conchordal::life::population::Population;
use conchordal::life::scenario::{ArticulationCoreConfig, EnvelopeConfig, VoiceConfig};
use conchordal::life::voice::VoiceMetadata;
use conchordal::life::world_model::WorldModel;

fn test_timebase() -> Timebase {
    Timebase {
        fs: 48_000.0,
        hop: 64,
    }
}

fn custom_envelope() -> EnvelopeConfig {
    EnvelopeConfig {
        attack_sec: 0.123,
        decay_sec: 0.456,
        sustain_level: 0.789,
        release_sec: 0.555,
    }
}

fn assert_adsr_matches(adsr: conchordal::life::sound::ToneAdsr, expected: &EnvelopeConfig) {
    assert!((adsr.attack_sec - expected.attack_sec).abs() < 1e-6);
    assert!((adsr.decay_sec - expected.decay_sec).abs() < 1e-6);
    assert!((adsr.sustain_level - expected.sustain_level).abs() < 1e-6);
    assert!((adsr.release_sec - expected.release_sec).abs() < 1e-6);
}

fn spawn_and_get_adsr(articulation: ArticulationCoreConfig, envelope: EnvelopeConfig) {
    let tb = test_timebase();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(tb, space.clone());
    let mut pop = Population::new(tb);

    let mut control = VoiceControl::default();
    control.pitch.freq = 440.0;
    control.body.amp = 0.4;
    control.body.envelope = envelope.clone();

    let cfg = VoiceConfig {
        control,
        articulation,
    };
    let meta = VoiceMetadata {
        group_id: 0,
        member_idx: 0,
        generation: 0,
        parent_id: None,
    };
    pop.add_voice(cfg.spawn(1, 0, meta, tb.fs, 0));

    let landscape = Landscape::new(space);
    let batches = pop.collect_phonation_batches(&mut world, &landscape, 0);
    let tone_spec = batches
        .iter()
        .flat_map(|b| b.tones.iter())
        .next()
        .expect("expected at least one tone spec");
    let adsr = tone_spec
        .adsr
        .expect("voice should carry adsr derived from body.envelope regardless of articulation");
    assert_adsr_matches(adsr, &envelope);
}

#[test]
fn entrain_brain_uses_body_envelope_for_tone_adsr() {
    spawn_and_get_adsr(ArticulationCoreConfig::default(), custom_envelope());
}

#[test]
fn drone_brain_uses_body_envelope_for_tone_adsr() {
    let articulation = ArticulationCoreConfig::Drone {
        sway: None,
        breath_gain_init: None,
    };
    spawn_and_get_adsr(articulation, custom_envelope());
}
