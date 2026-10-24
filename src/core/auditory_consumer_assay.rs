//! Reusable V2 consumer acquisition over the registered PCM and truth manifest.
use super::*;
const NAMES: [&str; 1] = ["current"];
use sha2::Digest;

#[test]
#[ignore = "explicit V2 consumer acquisition"]
fn acquire_current_v2_consumers() {
    let input = PathBuf::from(std::env::var("AUD_V2_INPUT").unwrap());
    let out = PathBuf::from(std::env::var("AUD_V2_OUTPUT").unwrap());
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(input.join("manifest.json")).unwrap()).unwrap();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (_, du) = erb_grid(&space);
    let peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let pilot = std::env::var_os("AUD_V2_PILOT").is_some();
    let existing: std::collections::BTreeSet<String> =
        std::fs::read_to_string(out.join("consumer-fixtures.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| {
                serde_json::from_str::<Value>(l).unwrap()["fixture"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
    let mut rows = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out.join("consumer-rows.jsonl"))
            .unwrap(),
    );
    let mut summaries = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out.join("consumer-fixtures.jsonl"))
            .unwrap(),
    );
    for fixture in manifest["fixtures"].as_array().unwrap() {
        let name = fixture["name"].as_str().unwrap();
        if (pilot && name != "onset_110.0") || existing.contains(name) {
            continue;
        }
        let bytes = std::fs::read(input.join(fixture["pcm"].as_str().unwrap())).unwrap();
        let hash = sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(hash, fixture["pcm_sha256"].as_str().unwrap());
        let pcm: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let frequencies: Vec<_> = fixture["frequencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let amplitudes: Vec<_> = fixture["amplitudes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let observe: Vec<_> = fixture["judgment_hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let flora_hops: Vec<_> = fixture["flora_hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(observe.len(), 8);
        assert_eq!(flora_hops.len(), 47);
        let onset = fixture["onset_hop"].as_u64().unwrap() as usize;
        let transient = fixture["category"] == "transient";
        let mut stationary = vec![0.0; space.n_bins()];
        for (&f, &a) in frequencies.iter().zip(&amplitudes) {
            deposit(
                &space,
                &du,
                &mut stationary,
                f,
                (a_weighting_gain_pow(f) * a * a * 0.5 / params.ref_power)
                    .powf(params.loudness_exp),
            );
        }
        let mut truth_density = if transient {
            vec![0.0; space.n_bins()]
        } else {
            stationary.clone()
        };
        let mut anchors: Vec<_> = frequencies
            .iter()
            .flat_map(|f| [0.5, 2.0 / 3.0, 1.0, 1.5].map(|r| f * r))
            .map(|f| f.clamp(space.fmin, space.fmax).log2())
            .collect();
        anchors.sort_unstable_by(f32::total_cmp);
        anchors.dedup();
        let mut streams: Vec<_> = (0..NAMES.len())
            .map(|_| SynchronousAnalysis::new(kernel.clone(), &params))
            .collect();
        let leak = (-(hop as f32) / params.fs / (params.tau_ms * 1e-3)).exp();
        let mut populations: Option<Vec<Vec<Flora>>> = None;
        let mut pitch = [0_u64; 1];
        let mut respawn = [0_u64; 1];
        let mut birth = [[0.0_f64; 3]; 1];
        let mut frames = 0;
        let mut pitch_targets = 0;
        let mut respawn_targets = 0;
        for (frame, audio) in pcm.chunks_exact(hop).enumerate() {
            for stream in &mut streams {
                stream.process(audio, &params);
                stream.landscape.recompute_consonance(&params);
            }
            if transient {
                for (old, &target) in truth_density.iter_mut().zip(&stationary) {
                    *old = leak * *old
                        + if frame >= onset {
                            (1.0 - leak) * target
                        } else {
                            0.0
                        };
                }
            }
            if !observe.contains(&frame) && !flora_hops.contains(&frame) {
                continue;
            }
            let truth = evaluate(&truth_density, &space, &params, &mut scratch, peak);
            if flora_hops.contains(&frame) {
                let populations = populations.get_or_insert_with(|| {
                    let initial = Flora::aud_m1_start_window(&truth);
                    (0..2).map(|_| initial.clone()).collect()
                });
                for (population, landscape) in populations
                    .iter_mut()
                    .zip(std::iter::once(&truth).chain(streams.iter().map(|s| &s.landscape)))
                {
                    for flora in population {
                        flora.advance(landscape);
                    }
                }
            }
            if !observe.contains(&frame) {
                continue;
            }
            frames += 1;
            pitch_targets += anchors.len();
            respawn_targets += Community::aud_m1_respawn(&truth).len();
            let assessments: Vec<_> = streams
                .iter()
                .map(|s| assess(&s.landscape, &truth, &anchors))
                .collect();
            for (i, assessment) in assessments.iter().enumerate() {
                pitch[i] += assessment["pitch_errors"].as_u64().unwrap();
                respawn[i] += assessment["respawn_errors"].as_u64().unwrap();
                for j in 0..3 {
                    birth[i][j] += assessment["birth_l1"][j].as_f64().unwrap();
                }
            }
            writeln!(
                rows,
                "{}",
                json!({"fixture":name,"frame":frame,"variants":NAMES,"assessments":assessments})
            )
            .unwrap();
        }
        let signs: Vec<_> = populations
            .unwrap()
            .iter()
            .map(|s| Flora::aud_m1_window_signs(s))
            .collect();
        let flora: Vec<usize> = signs[1..]
            .iter()
            .map(|s| {
                s.iter()
                    .zip(&signs[0])
                    .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
                    .sum()
            })
            .collect();
        let summary = json!({"fixture":name,"category":fixture["category"],"resolution":fixture["pair_input_resolution"],"frames":frames,"pitch_targets":pitch_targets,"respawn_targets":respawn_targets,"flora_targets":signs[0].iter().map(Vec::len).sum::<usize>(),"pitch":pitch,"respawn":respawn,"flora47":flora,"birth_sum_l1":birth,"pcm_sha256":hash,"processed_samples":pcm.len()/hop*hop,"trailing_samples":pcm.len()%hop});
        writeln!(summaries, "{summary}").unwrap();
        summaries.flush().unwrap();
        rows.flush().unwrap();
        println!("AUD_V2 fixture={name} pitch={pitch:?} respawn={respawn:?} flora={flora:?}");
    }
}
