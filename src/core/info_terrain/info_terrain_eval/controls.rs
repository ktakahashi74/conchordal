use super::*;
use std::f64::consts::{PI, TAU};

pub(in crate::core::info_terrain) fn gaussian(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = SplitMix64(seed);
    let mut output = Vec::with_capacity(n);
    while output.len() < n {
        let radius = (-2.0 * rng.uniform().ln()).sqrt();
        let angle = TAU * rng.uniform();
        output.push(radius * angle.cos());
        if output.len() < n {
            output.push(radius * angle.sin());
        }
    }
    output
}

pub(in crate::core::info_terrain) fn normalize(values: &mut [f64]) {
    let rms = (values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64).sqrt();
    let target = 10_f64.powf(-12.0 / 20.0) / 2_f64.sqrt();
    for x in values {
        *x *= target / rms;
    }
}

fn segments(deviant: bool) -> Vec<f64> {
    let segment_samples = FS as usize / 2;
    let half_boundary = (0.005 * f64::from(FS)).round() as isize;
    let frequency = |segment: isize| {
        if deviant && [13, 21, 29, 37].contains(&segment) {
            550.0
        } else if segment % 2 == 0 {
            440.0
        } else {
            660.0
        }
    };
    // Both tones are extended across each boundary. Their phases remain zero
    // at their own nominal segment starts, including negative next-tone time.
    let tone = |sample: isize, segment: isize| {
        (TAU * frequency(segment) * (sample - segment * segment_samples as isize) as f64
            / f64::from(FS))
        .sin()
    };
    let mut values: Vec<_> = (0..20 * FS as usize)
        .map(|sample| {
            let sample = sample as isize;
            let segment = sample / segment_samples as isize;
            let offset = sample % segment_samples as isize;
            let boundary = if offset < half_boundary && segment > 0 {
                Some(segment)
            } else if offset >= segment_samples as isize - half_boundary && segment < 39 {
                Some(segment + 1)
            } else {
                None
            };
            if let Some(next) = boundary {
                let position = (sample - next * segment_samples as isize + half_boundary) as f64
                    / (2 * half_boundary) as f64;
                let incoming = 0.5 - 0.5 * (PI * position).cos();
                (1.0 - incoming) * tone(sample, next - 1) + incoming * tone(sample, next)
            } else {
                tone(sample, segment)
            }
        })
        .collect();
    normalize(&mut values);
    values
}

pub(in crate::core::info_terrain) fn stimuli() -> Vec<(&'static str, Vec<f32>)> {
    let n = 20 * FS as usize;
    let amplitude = 10_f64.powf(-12.0 / 20.0);
    let sine = (0..n)
        .map(|i| (amplitude * (TAU * 440.0 * i as f64 / f64::from(FS)).sin()) as f32)
        .collect();
    let mut white = gaussian(1, n);
    normalize(&mut white);
    let white: Vec<f32> = white.into_iter().map(|x| x as f32).collect();
    let quiet = white
        .iter()
        .map(|x| (f64::from(*x) * 10_f64.powf(-30.0 / 20.0)) as f32)
        .collect();
    let mut pink = gaussian(2, n);
    let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
    for x in &mut pink {
        let white = *x;
        b0 = 0.99765 * b0 + 0.0990460 * white;
        b1 = 0.96300 * b1 + 0.2965164 * white;
        b2 = 0.57000 * b2 + 1.0526913 * white;
        *x = b0 + b1 + b2 + 0.1848 * white;
    }
    normalize(&mut pink);
    vec![
        ("1_sine", sine),
        ("2a_white", white),
        ("2b_white_quiet", quiet),
        ("3_pink", pink.into_iter().map(|x| x as f32).collect()),
        (
            "4_periodic",
            segments(false).into_iter().map(|x| x as f32).collect(),
        ),
        (
            "5_deviant",
            segments(true).into_iter().map(|x| x as f32).collect(),
        ),
    ]
}

fn interval_max(evaluation: &Evaluation, version: usize, start: f64, end: f64) -> Option<f64> {
    evaluation
        .trace
        .iter()
        .filter(|hop| {
            hop["time_sec"]
                .as_f64()
                .is_some_and(|t| (start..end).contains(&t))
        })
        .filter_map(|hop| hop["surprise"][version][0].as_f64())
        .max_by(f64::total_cmp)
}

pub(super) fn info_terrain_eval_controls() -> Result<()> {
    let directory = results()?.join("controls");
    fs::create_dir_all(&directory)?;
    let stream = crate::runtime::info_terrain_eval_stream(FS);
    let mut evaluations = BTreeMap::new();
    let mut summaries = Vec::new();
    let mut gain_ok = false;
    for (name, audio) in stimuli() {
        let path = directory.join(format!("{name}.wav"));
        // Float WAV avoids introducing PCM quantization into the gain check.
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: FS,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for &sample in &audio {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        let (fs, samples) = read_wav(&path)?;
        ensure!(samples == audio, "float WAV roundtrip changed samples");
        let evaluation = evaluate(
            &samples,
            fs,
            stream.clone(),
            (name == "2a_white").then(|| 10_f64.powf(-30.0 / 20.0)),
        )?;
        let rms = (samples.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
            / samples.len() as f64)
            .sqrt();
        let original = &evaluation.window_summary.as_ref().unwrap().original;
        let extended = &evaluation.window_summary.as_ref().unwrap().extended;
        println!(
            "{name}: original S={:?} precision={:?}; extended S={:?} precision={:?}",
            original.soundgen.mean,
            original.precision_mean,
            extended.soundgen.mean,
            extended.precision_mean
        );
        summaries.push(json!({"stimulus":name,"rms":rms,"wav_sha256":format!("{:x}",Sha256::digest(fs::read(&path)?)),
            "original":{"mean_surprise":original.soundgen.mean,"mean_precision":original.precision_mean},
            "extended":{"mean_surprise":extended.soundgen.mean,"mean_precision":extended.precision_mean},
            "gain_check":evaluation.gain_check}));
        if let Some(check) = &evaluation.gain_check {
            gain_ok = check["evaluation_window"]["mask_mismatches"] == 0
                && check["evaluation_window"]["tolerance_failures"] == 0;
            println!("gain check: {check}");
        }
        write_json(&directory.join(format!("{name}.json")), &evaluation)?;
        evaluations.insert(name, evaluation);
    }
    let original_and_extended = (0..2).map(|v| {
        let summary = |name| {
            let block = evaluations[name].window_summary.as_ref().unwrap();
            if v==0 { &block.original } else { &block.extended }
        };
        let one = summary("1_sine");
        let white = summary("2a_white");
        let pink = summary("3_pink");
        let order = |a: Option<f64>,b: Option<f64>,less:bool| a.zip(b).map(|(a,b)| if less {a<b} else {a>b});
        let between = |x:Option<f64>,a:Option<f64>,b:Option<f64>| x.zip(a.zip(b)).map(|(x,(a,b))| x>=a.min(b) && x<=a.max(b));
        let deviations:Vec<_> = [6.5,10.5,14.5,18.5].into_iter().map(|t| {
            let periodic = interval_max(&evaluations["4_periodic"],v,t,t+0.5);
            let deviant = interval_max(&evaluations["5_deviant"],v,t,t+0.5);
            json!({"start_sec":t,"end_sec":t+0.5,"periodic_max":periodic,"deviant_max":deviant,"deviant_greater":order(deviant,periodic,false)})
        }).collect();
        let periodic_half_seconds: Vec<_> = (4..40).map(|segment| {
            let start = segment as f64*0.5;
            let values: Vec<_> = evaluations["4_periodic"].trace.iter()
                .filter(|hop| hop["time_sec"].as_f64().is_some_and(|t| (start..start+0.5).contains(&t)))
                .filter_map(|hop| hop["surprise"][v][0].as_f64()).collect();
            json!({"start_sec":start,"mean":(!values.is_empty()).then(||values.iter().sum::<f64>()/values.len() as f64),
                "maximum":interval_max(&evaluations["4_periodic"],v,start,start+0.5)})
        }).collect();
        json!({"version":if v==0 {"original"} else {"extended"},
            "hypothesis_1":{"surprise_less_than_white":order(one.soundgen.mean,white.soundgen.mean,true),"precision_greater_than_white":order(one.precision_mean,white.precision_mean,false),
                "surprise_comparable":one.soundgen.mean.is_some()},
            "hypothesis_3":{"surprise_between":between(pink.soundgen.mean,one.soundgen.mean,white.soundgen.mean),"precision_between":between(pink.precision_mean,one.precision_mean,white.precision_mean),"direction_registered":false},
            "hypothesis_4":{"periodic_half_second_intervals":periodic_half_seconds},"hypothesis_5":deviations})
    }).collect::<Vec<_>>();
    // Diagnostic only: binary scaling is exactly representable in f32.
    let (_, white) = read_wav(&directory.join("2a_white.wav"))?;
    let binary_gain = evaluate(&white, FS, stream, Some(1.0 / 32.0))?;
    println!(
        "binary gain rounding probe: {}",
        binary_gain.gain_check.as_ref().unwrap()
    );
    let output = json!({"provenance":provenance()?,"cases":summaries,"unregistered_binary_gain_rounding_probe":binary_gain.gain_check,"hypotheses_recorded_not_asserted":original_and_extended,
        "generation_conventions":{"segments":"whole-signal RMS normalized; no start/end fade; complementary raised-cosine crossfade across +/-5 ms with each segment phase zero at its own start",
            "wav":"float32 mono, no clipping, no resampling"}});
    write_json(&results()?.join("controls.json"), &output)?;
    let mut markdown = String::from(
        "# Control measurements\n\nRevision A-2 draft 3; native habitat analysis; [2,20) s; soundgen 3.0.0 aggregation. Hypotheses recorded without acceptance assertions.\n\n| Stimulus | Original S / precision | Extended S / precision |\n|---|---:|---:|\n",
    );
    for row in &summaries {
        markdown.push_str(&format!(
            "| {} | {} / {} | {} / {} |\n",
            row["stimulus"].as_str().unwrap(),
            row["original"]["mean_surprise"],
            row["original"]["mean_precision"],
            row["extended"]["mean_surprise"],
            row["extended"]["mean_precision"]
        ));
    }
    markdown.push_str(&format!("\nGain pair implementation check in the registered [2,20) s window: {}. Full masks and indicator values compared at every bin and hop; startup diagnostics retained separately in JSON.\n",if gain_ok {"passed"} else {"FAILED"}));
    markdown.push_str(&format!("\nAdditional startup diagnostic: {} tolerance violations over the full [0,20) s input; see exact times and bins in controls.json. Unregistered exact binary gain (1/32) probe: {} tolerance violations.\n",
        evaluations["2a_white"].gain_check.as_ref().unwrap()["tolerance_failures"],
        binary_gain.gain_check.as_ref().unwrap()["tolerance_failures"]));
    fs::write(results()?.join("controls.md"), markdown)?;
    ensure!(
        gain_ok,
        "gain pair violated masks/tolerance in the registered [2,20) s evaluation window; diagnostics saved in controls.json"
    );
    Ok(())
}
