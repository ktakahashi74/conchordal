#![cfg(feature = "research")]
#![cfg(all(feature = "web", not(feature = "native")))]
use conchordal::{StepEngine, StepStatus};
use std::io::Write;

// Source figure values stay in seconds; omission still disables the observer.
const CONFIG: &str = "[analysis.felt_beat]\n";

#[test]
fn felt_beat_step_reserves_only_habitat_and_preserves_pcm() {
    let script = "let v=place(harmonic().amp(0.03).metric().cycles(3),at(220)); wait(3); release(v); wait(1);";
    let mut a = StepEngine::new(script, "", Some(1), 48_000).unwrap();
    let mut b = StepEngine::new(script, CONFIG, Some(1), 48_000).unwrap();
    let mut x = vec![0.; a.hop_len()];
    let mut y = vec![0.; b.hop_len()];
    for _ in 0..300 {
        let sa = a.step(&mut x);
        let sb = b.step(&mut y);
        assert!(!matches!(sa, StepStatus::Error(_)), "{sa:?}");
        assert_eq!(sa, sb);
        assert!(x.iter().zip(&y).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

#[test]
#[ignore = "run the prebuilt release test binary inside the timing lock"]
fn felt_beat_step_timing_abba() {
    let script = std::fs::read_to_string(std::env::var("FB1_TIMING_SCRIPT").unwrap()).unwrap();
    let dir = std::path::PathBuf::from(std::env::var("FB1_TIMING_DIR").unwrap());
    for (tag, config) in [("a1", ""), ("b1", CONFIG), ("b2", CONFIG), ("a2", "")] {
        let mut engine = StepEngine::new(&script, config, Some(1), 48_000).unwrap();
        let mut pcm = vec![0.; engine.hop_len()];
        let budget = engine.hop_len() as f64 / 48_000. * 1e6;
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("{tag}.jsonl"))).unwrap(),
        );
        let mut frame = 0;
        loop {
            let began = std::time::Instant::now();
            let status = engine.step(&mut pcm);
            let elapsed = began.elapsed().as_secs_f64() * 1e6;
            assert!(!matches!(status, StepStatus::Error(_)), "{status:?}");
            writeln!(file,"{{\"type\":\"hop_timing\",\"frame_idx\":{frame},\"time_sec\":{},\"elapsed_us\":{elapsed},\"hop_budget_us\":{budget}}}",frame as f64*budget/1e6).unwrap();
            frame += 1;
            if status == StepStatus::Finished {
                break;
            }
        }
        println!("felt StepEngine timing {tag} hops={frame}");
    }
}

#[test]
fn felt_generation_auto_reserves_without_flora_and_zero_is_neutral() {
    let base = "let v=place(harmonic().amp(0.03).metric().cycles(1),at(220).count(3)); wait(4); release(v); wait(1);";
    let zero = base.replace("metric()", "metric().syncopate(0)");
    let on = base.replace("metric()", "metric().syncopate(1)");
    let mut a = StepEngine::new(base, "", Some(1), 48000).unwrap();
    let mut b = StepEngine::new(&zero, "", Some(1), 48000).unwrap();
    let mut c = StepEngine::new(&on, "", Some(1), 48000).unwrap();
    let mut x = vec![0.; a.hop_len()];
    let mut y = x.clone();
    let mut z = x.clone();
    for _ in 0..470 {
        assert_eq!(a.step(&mut x), b.step(&mut y));
        assert!(x.iter().zip(&y).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(!matches!(c.step(&mut z), StepStatus::Error(_)));
    }
}

#[test]
#[ignore = "run only the prebuilt release binary inside the timing lock"]
fn felt_generation_step_timing_abba() {
    let script = std::fs::read_to_string(std::env::var("FB3_TIMING_SCRIPT").unwrap()).unwrap();
    let off = script.replace("syncopate(1.0)", "syncopate(0.0)");
    let dir = std::path::PathBuf::from(std::env::var("FB3_TIMING_DIR").unwrap());
    for (tag, script) in [("a1", &off), ("b1", &script), ("b2", &script), ("a2", &off)] {
        let mut engine = StepEngine::new(script, "", Some(1), 48000).unwrap();
        let mut pcm = vec![0.; engine.hop_len()];
        let budget = engine.hop_len() as f64 / 48000. * 1e6;
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("{tag}.jsonl"))).unwrap(),
        );
        let mut frame = 0;
        loop {
            let begin = std::time::Instant::now();
            let status = engine.step(&mut pcm);
            let elapsed = begin.elapsed().as_secs_f64() * 1e6;
            assert!(!matches!(status, StepStatus::Error(_)), "{status:?}");
            writeln!(file,"{{\"type\":\"hop_timing\",\"frame_idx\":{frame},\"time_sec\":{},\"elapsed_us\":{elapsed},\"hop_budget_us\":{budget}}}",frame as f64*budget/1e6).unwrap();
            frame += 1;
            if status == StepStatus::Finished {
                break;
            }
        }
        println!("felt generation timing {tag} hops={frame}");
    }
}
