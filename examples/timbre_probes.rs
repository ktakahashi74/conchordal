//! Timbre audition probes.
//!
//! Renders a deterministic battery of short WAVs through the real scheduled
//! tone path (`Tone` + `AnyBackend`) so a human can audition the synth bodies
//! in isolation. This is deliberately not a cargo test: the machine cannot
//! judge timbre, the ear is the assay (docs/design-notes/timbre.md).
//!
//! Run: `cargo run --release --example timbre_probes`
//! Output: `target/timbre_probes/*.wav` (ephemeral, untracked)

use conchordal::core::log2space::Log2Space;
use conchordal::core::mode_pattern::ModePattern;
use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::sound::{BodyKind, BodySnapshot, Tone};
use hound::{SampleFormat, WavSpec, WavWriter};
use rand::{SeedableRng, rngs::SmallRng};
use std::path::Path;
use std::sync::Arc;

const FS: f32 = 48_000.0;
const HOP: usize = 64;
const FREQ_HZ: f32 = 220.0;

struct Probe {
    name: &'static str,
    listen_for: &'static str,
    snapshot: BodySnapshot,
    hold_sec: f32,
    total_sec: f32,
    continuous_drive: f32,
    restrike_sec: Option<f32>,
}

fn snapshot(
    kind: BodyKind,
    brightness: f32,
    spread: f32,
    unison: usize,
    motion: f32,
    ratios: Option<Arc<[f32]>>,
) -> BodySnapshot {
    BodySnapshot {
        kind,
        amp_scale: 1.0,
        brightness,
        inharmonic: 0.0,
        spread,
        unison,
        motion,
        ratios,
    }
}

fn pattern_ratios(pattern: ModePattern) -> Arc<[f32]> {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut rng = SmallRng::seed_from_u64(7);
    Arc::from(pattern.eval(FREQ_HZ, &space, None, &mut rng))
}

fn modal_table_ratios(name: &str) -> Arc<[f32]> {
    pattern_ratios(ModePattern::modal_table(name).expect("known modal table"))
}

fn render_probe(probe: &Probe) -> Vec<f32> {
    let tb = Timebase { fs: FS, hop: HOP };
    let hold_ticks = (probe.hold_sec * FS) as Tick;
    let mut tone = Tone::from_parts(
        tb,
        0,
        hold_ticks,
        FREQ_HZ,
        0.5,
        Some(probe.snapshot.clone()),
        None,
        None,
    )
    .expect("probe tone");
    tone.seed_modal_phases(0xC0FF_EE00 ^ probe.name.len() as u64);
    tone.trigger_impulse(1.0);
    if probe.continuous_drive > 0.0 {
        tone.set_continuous_drive(probe.continuous_drive);
    }

    let restrike_tick = probe.restrike_sec.map(|s| (s * FS) as Tick);
    let total_samples = (probe.total_sec * FS) as usize;
    let dt = 1.0 / FS;
    let mut rhythms = NeuralRhythms::default();
    let mut out = vec![0.0f32; total_samples];
    let mut tick: Tick = 0;
    for chunk in out.chunks_mut(HOP) {
        if let Some(rt) = restrike_tick
            && tick <= rt
            && rt < tick + chunk.len() as Tick
        {
            tone.trigger_impulse(1.0);
        }
        tone.render_block(tick, FS, dt, &mut rhythms, chunk);
        tick += chunk.len() as Tick;
    }
    out
}

/// Normalize to 0.9 peak and write a mono 16-bit WAV. Returns the pre-normalize peak.
fn write_wav(dir: &Path, name: &str, samples: &mut [f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    if peak > 0.0 {
        let gain = 0.9 / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: FS as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let path = dir.join(format!("{name}.wav"));
    let mut writer = WavWriter::create(&path, spec).expect("create wav");
    for &s in samples.iter() {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    peak
}

fn main() {
    let strike = |name, listen_for, snapshot| Probe {
        name,
        listen_for,
        snapshot,
        hold_sec: 2.5,
        total_sec: 4.0,
        continuous_drive: 0.0,
        restrike_sec: None,
    };

    let probes = vec![
        strike(
            "sine_strike",
            "reference anchor: pure tone, short onset boost, no spectral motion",
            snapshot(BodyKind::Sine, 0.0, 0.0, 1, 0.0, None),
        ),
        strike(
            "harmonic_b020_strike",
            "dark body: few partials; ring-down should darken only subtly",
            snapshot(BodyKind::Harmonic, 0.2, 0.0, 1, 0.0, None),
        ),
        strike(
            "harmonic_b050_strike",
            "damping audition: bright attack, upper partials thin toward the floor as it rings",
            snapshot(BodyKind::Harmonic, 0.5, 0.0, 1, 0.0, None),
        ),
        strike(
            "harmonic_b080_strike",
            "bright body: strongest attack-vs-sustain brightness contrast",
            snapshot(BodyKind::Harmonic, 0.8, 0.0, 1, 0.0, None),
        ),
        Probe {
            name: "harmonic_b050_restrike",
            listen_for: "re-strike at 2 s should re-brighten the already-darkened tone",
            snapshot: snapshot(BodyKind::Harmonic, 0.5, 0.0, 1, 0.0, None),
            hold_sec: 3.5,
            total_sec: 4.5,
            continuous_drive: 0.0,
            restrike_sec: Some(2.0),
        },
        Probe {
            name: "harmonic_b050_breath",
            listen_for: "strong continuous drive holds brightness up (compare with plain strike)",
            snapshot: snapshot(BodyKind::Harmonic, 0.5, 0.0, 1, 0.0, None),
            hold_sec: 2.5,
            total_sec: 4.0,
            continuous_drive: 0.6,
            restrike_sec: None,
        },
        strike(
            "harmonic_spread_unison",
            "chorus thickening: 5 detuned copies, slow beating, no pitch wobble",
            snapshot(BodyKind::Harmonic, 0.5, 0.6, 5, 0.0, None),
        ),
        strike(
            "harmonic_motion",
            "vibrato + 1/f jitter: organic pitch motion, timbre otherwise steady",
            snapshot(BodyKind::Harmonic, 0.5, 0.0, 1, 0.6, None),
        ),
        strike(
            "modal_aluminum_strike",
            "inharmonic bar: metallic strike, modes decay at different rates",
            snapshot(
                BodyKind::Modal,
                0.6,
                0.0,
                1,
                0.0,
                Some(modal_table_ratios("uniform_aluminum_bar")),
            ),
        ),
        strike(
            "modal_wine_glass_strike",
            "glass: sparse high modes, long singing ring",
            snapshot(
                BodyKind::Modal,
                0.6,
                0.0,
                1,
                0.0,
                Some(modal_table_ratios("wine_glass")),
            ),
        ),
        strike(
            "modal_xylophone_strike",
            "wooden percussion: fast decay, strong widely-spaced overtones",
            snapshot(
                BodyKind::Modal,
                0.6,
                0.0,
                1,
                0.0,
                Some(modal_table_ratios("xylophone")),
            ),
        ),
        strike(
            "modal_stiff_string_strike",
            "stretched partials: piano-like inharmonicity, compare against harmonic_b050",
            snapshot(
                BodyKind::Modal,
                0.6,
                0.0,
                1,
                0.0,
                Some(pattern_ratios(
                    ModePattern::stiff_string_modes(0.004).with_count(9),
                )),
            ),
        ),
    ];

    let dir = Path::new("target/timbre_probes");
    std::fs::create_dir_all(dir).expect("create output dir");

    println!("rendering {} probes to {}", probes.len(), dir.display());
    println!();
    for probe in &probes {
        let mut samples = render_probe(probe);
        let peak = write_wav(dir, probe.name, &mut samples);
        assert!(
            peak > 1.0e-4,
            "probe {} rendered near-silence (peak {peak}); synth path broken?",
            probe.name
        );
        println!(
            "  {:<28} {:>5.1}s  peak {:.3}\n      listen: {}",
            probe.name, probe.total_sec, peak, probe.listen_for
        );
    }
    println!();
    println!("done. audition with e.g.: mpv target/timbre_probes/*.wav");
}
