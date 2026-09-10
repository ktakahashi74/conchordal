//! A closed acoustic loop for the research policy, before PhonationEngine integration.

use super::TemporalParticipation;
use crate::core::stream::dorsal::DorsalStream;
use crate::core::temporal_expectation::{AcousticTemporalExpectation, OwnSoundHistory};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const FS: u32 = 48_000;
const WINDOW: usize = 480;

struct Run {
    audio: Vec<f32>,
    actions: [Vec<u64>; 3],
    periods: [Vec<f64>; 3],
    frequencies: [f32; 3],
    energies: Vec<[[f32; 3]; 3]>,
}

fn run_condition(case: &str, condition: &str, seed: u64, seconds: usize, reverse: bool) -> Run {
    let frequencies = match case {
        "different_bands" => [180.0, 1800.0, 6800.0],
        "join_leave" => [180.0, 180.0, 1800.0],
        "same_band" | "solo" => [440.0; 3],
        _ => panic!("unknown research case"),
    };
    let (coupling, sensitivity) = match condition {
        "intrinsic" => (0.0, 0.0),
        "context" => (0.7, 0.0),
        "overlap_avoid" => (0.7, 0.8),
        "overlap_seek" => (0.7, -0.8),
        _ => panic!("unknown research condition"),
    };
    let mut population_rng = SmallRng::seed_from_u64(seed);
    let mut voices = std::array::from_fn::<_, 3, _>(|i| {
        let mut voice =
            TemporalParticipation::new(FS, 2.0, coupling, 0, population_rng.random(), FS as u64);
        let mut dorsal = DorsalStream::new(FS as f32);
        let first: [f32; WINDOW] = std::array::from_fn(|sample| {
            let age = sample as f32 / FS as f32;
            0.3 * (std::f32::consts::TAU * frequencies[i] * age).sin() * (-age / 0.018).exp()
        });
        dorsal.process(&first);
        let e = dorsal.last_metrics();
        voice.set_overlap(sensitivity, [e.e_low, e.e_mid, e.e_high]);
        voice
    });
    let mut observer = AcousticTemporalExpectation::new(FS).unwrap();
    let mut self_dorsal = std::array::from_fn::<_, 3, _>(|_| DorsalStream::new(FS as f32));
    let mut own_history = std::array::from_fn::<_, 3, _>(|_| OwnSoundHistory::new(&observer, 0));
    let mut latest: [[Option<u64>; 2]; 3] = [[None; 2]; 3];
    let mut result = Run {
        audio: Vec::with_capacity(seconds * FS as usize),
        actions: std::array::from_fn(|_| Vec::new()),
        periods: std::array::from_fn(|_| Vec::new()),
        frequencies,
        energies: Vec::with_capacity(seconds * 100),
    };
    for hop in 0..seconds * 100 {
        let now = (hop * WINDOW) as u64;
        let shared = observer.forecast();
        let order = if reverse { [2, 1, 0] } else { [0, 1, 2] };
        for i in order {
            let mut local = shared;
            if let Some(local) = local.as_mut() {
                observer.use_external_energy(local, &mut own_history[i].external);
            }
            let allowed = (case != "solo" || i == 0)
                && (case != "join_leave" || i != 1 || !(800..1600).contains(&hop));
            if let Some(onset) =
                voices[i].advance(now, now + WINDOW as u64, allowed, local.as_ref())
            {
                latest[i][0] = latest[i][1];
                latest[i][1] = Some(onset);
                result.actions[i].push(onset);
                result.periods[i].push(voices[i].period_frames / FS as f64);
            }
        }
        let mut mixed = [0.0; WINDOW];
        let mut band_energy = [[0.0; 3]; 3];
        let mut own_audio = [[0.0; WINDOW]; 3];
        for i in 0..3 {
            let own: [f32; WINDOW] = std::array::from_fn(|sample| {
                let frame = now + sample as u64;
                latest[i]
                    .into_iter()
                    .flatten()
                    .filter_map(|onset| frame.checked_sub(onset))
                    .filter(|age| *age < (FS as f32 * 0.18) as u64)
                    .map(|age| {
                        let t = age as f32 / FS as f32;
                        0.3 * (std::f32::consts::TAU * frequencies[i] * t).sin()
                            * (-t / 0.018).exp()
                    })
                    .sum()
            });
            self_dorsal[i].process(&own);
            let e = self_dorsal[i].last_metrics();
            band_energy[i] = [e.e_low, e.e_mid, e.e_high];
            own_audio[i] = own;
            for (sum, value) in mixed.iter_mut().zip(own) {
                *sum += value;
            }
        }
        for ((history, audio), voice) in own_history.iter_mut().zip(&own_audio).zip(&mut voices) {
            history.process(now, audio, audio, &mixed, |_, _, _| {});
            voice.observe_context(history, |_| {});
        }
        // All actors decide before any of this hop's sound reaches perception.
        observer.process(now, &mixed, |_| {});
        result.audio.extend(mixed);
        result.energies.push(band_energy);
    }
    result
}

#[test]
fn self_prediction_prevents_a_solitary_voice_from_avoiding_its_own_echo() {
    let control = run_condition("solo", "context", 21, 12, false);
    for condition in ["overlap_avoid", "overlap_seek"] {
        let result = run_condition("solo", condition, 21, 12, false);
        assert_eq!(result.actions, control.actions);
        assert_eq!(result.audio, control.audio);
    }
}

#[test]
fn actor_visit_order_does_not_turn_an_unrendered_plan_into_acoustic_evidence() {
    let forward = run_condition("same_band", "overlap_avoid", 42, 12, false);
    let reverse = run_condition("same_band", "overlap_avoid", 42, 12, true);
    assert_eq!(forward.actions, reverse.actions);
    assert_eq!(forward.audio, reverse.audio);
}

#[test]
fn spectral_overlap_changes_actions_without_stopping_the_bodies() {
    let control = run_condition("same_band", "context", 1, 12, false);
    let result = run_condition("same_band", "overlap_avoid", 1, 12, false);
    assert_ne!(result.actions, control.actions);
    assert!(result.actions.iter().all(|actions| actions.len() >= 18));
    assert!(
        result
            .audio
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
    );
}

#[test]
#[ignore = "offline acoustic participation campaign; set CONCHORDAL_PARTICIPATION_DIR"]
fn export_closed_loop_participation() -> anyhow::Result<()> {
    use std::fs::{self, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::path::Path;
    let root = std::env::var_os("CONCHORDAL_PARTICIPATION_DIR")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_PARTICIPATION_DIR is required"))?;
    for seed in [1, 21, 42] {
        for case in ["solo", "same_band", "different_bands", "join_leave"] {
            for condition in ["intrinsic", "context", "overlap_avoid", "overlap_seek"] {
                let folder = Path::new(&root).join(format!("{case}_{condition}_seed{seed}"));
                fs::create_dir(&folder)?;
                let result = run_condition(case, condition, seed, 24, false);
                let mut out = BufWriter::new(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(folder.join("actions.csv"))?,
                );
                writeln!(
                    out,
                    "voice,onset_frame,time_sec,frequency_hz,body_period_sec"
                )?;
                for i in 0..3 {
                    for (onset, period) in result.actions[i].iter().zip(&result.periods[i]) {
                        writeln!(
                            out,
                            "{i},{onset},{:.9},{},{period:.9}",
                            *onset as f64 / FS as f64,
                            result.frequencies[i]
                        )?;
                    }
                }
                out.flush()?;
                let mut energy_out = BufWriter::new(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(folder.join("energies.csv"))?,
                );
                writeln!(
                    energy_out,
                    "window_end_frame,time_sec,voice,e_low,e_mid,e_high"
                )?;
                for (hop, energies) in result.energies.iter().enumerate() {
                    for (voice, e) in energies.iter().enumerate() {
                        writeln!(
                            energy_out,
                            "{},{:.9},{voice},{:.12},{:.12},{:.12}",
                            (hop + 1) * WINDOW,
                            (hop + 1) as f64 / 100.0,
                            e[0],
                            e[1],
                            e[2]
                        )?;
                    }
                }
                energy_out.flush()?;
                let wav = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(folder.join("audio.wav"))?;
                let mut writer = hound::WavWriter::new(
                    BufWriter::new(wav),
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: FS,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )?;
                for sample in result.audio {
                    anyhow::ensure!(
                        sample.is_finite() && sample.abs() <= 1.0,
                        "invalid research audio"
                    );
                    writer.write_sample((sample * 32767.0).round() as i16)?;
                }
                writer.finalize()?;
                println!("rendered {}", folder.display());
            }
        }
    }
    Ok(())
}
