use super::*;
use crate::core::onset::Onset;
fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../tests/research/felt_beat_reference.json"
    ))
    .unwrap()
}
#[test]
fn f64_linear_path_matches_numpy_source_and_mlx_scenes() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let numbers = |key: &str| {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let means = numbers("means");
        let widths = numbers("variances");
        let strengths = numbers("lambdas");
        let events = numbers("events");
        let background = case["lambda_0"].as_f64().unwrap();
        let sigma2 = case["sigma2"].as_f64().unwrap();
        let dt = case["dt"].as_f64().unwrap();
        let trace = case["trace"].as_array().unwrap();
        let (mut mu, mut variance, mut event) = (0.0_f64, 0.0002_f64, 0);
        let mut maximum = [0.0_f64; 2];
        for i in 1..trace.len() {
            let compute = |mu, variance| {
                moments(
                    mu,
                    variance,
                    background,
                    means
                        .iter()
                        .zip(&widths)
                        .zip(&strengths)
                        .map(|((&m, &v), &l)| (m, v, l)),
                )
            };
            let a = compute(mu, variance);
            let displacement = dt * (1.0 - a.rate * a.delta);
            variance += dt * (sigma2 - a.rate * (a.variance_about(displacement) - variance));
            mu += displacement;
            let t = trace[i][0].as_f64().unwrap();
            let previous = trace[i - 1][0].as_f64().unwrap();
            if event < events.len() && t >= events[event] && previous <= events[event] {
                let a = compute(mu, variance);
                variance = a.variance_about(a.delta);
                mu += a.delta;
                event += 1;
            }
            maximum[0] = maximum[0].max((mu - trace[i][1].as_f64().unwrap()).abs());
            maximum[1] = maximum[1].max((variance - trace[i][2].as_f64().unwrap()).abs());
        }
        // Accumulated rounding allowance, not a perceptual acceptance threshold.
        let roundoff = 8.0 * trace.len() as f64 * f64::EPSILON;
        assert!(
            maximum.iter().all(|e| *e < roundoff),
            "{} {maximum:?}",
            case["name"]
        );
        println!(
            "felt source {} max_abs_mu_V={maximum:?} bound={roundoff:e}",
            case["name"]
        );
    }
}

#[test]
fn source_seconds_defaults_and_research_config() {
    let p = FeltBeatParams::default();
    assert_eq!(p.sigma2, 0.05 * 0.05);
    assert_eq!(
        (p.v, p.lambda, p.lambda_0, p.weak_v, p.weak_lambda),
        (0.0001, 0.05, 0.01, 0.0005, 0.01)
    );
    use crate::config::AppConfig;
    assert!(
        AppConfig::from_toml("")
            .unwrap()
            .analysis
            .felt_beat
            .is_none()
    );
    assert!(
        AppConfig::from_toml("[analysis.felt_beat]")
            .unwrap()
            .analysis
            .felt_beat
            .is_some()
    );
    for bad in [
        "h=0",
        "v=0",
        "weak_v=-1",
        "sigma2=nan",
        "weak_lambda=-1",
        "typo=1",
    ] {
        assert!(
            AppConfig::from_toml(&format!("[analysis.felt_beat]\n{bad}")).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn seed_is_uniform_and_unknown_is_only_an_input_condition() {
    let mut f = FeltBeat::new(FeltBeatParams::default());
    assert!(f.state().unknown);
    f.process(
        0.01,
        2.0,
        Onset {
            fired: true,
            frac: 1.0,
        },
        true,
        0.0,
        0.0,
    );
    assert!(!f.state().unknown);
    assert_eq!(f.state().p_beat, 0.5);
    assert!(f.state().c < (GRID as f32) * f32::EPSILON);
    f.process(20.0, 2.0, Onset::default(), false, 0.0, 0.0);
    assert!(!f.state().unknown);
    f.invalidate();
    assert!(f.state().unknown);
}

// 256 cells with eight time cells left max m_plus error 0.00435; one time
// cell at 1024 points left 0.01029. Both exceed the same-witness five-seed
// spread (minimum 0.00243). 1024/eight gives max 0.00134 without changing
// model parameters. Both consumer draws share this end-to-end error budget.
#[test]
fn grid_matches_fine_reference_at_hop_and_event_boundaries() {
    use rand::{RngExt, SeedableRng};
    let mut accepted = true;
    for case in reference()["grid"].as_array().unwrap() {
        let period = case["period"].as_f64().unwrap() as f32;
        let dt = case["hop"].as_f64().unwrap() as f32;
        let mut f = FeltBeat::new(FeltBeatParams::default());
        f.seed();
        let mut maximum = [0.0_f64; 3];
        let mut probabilities = [Vec::new(), Vec::new()];
        let mut worst = None;
        let mut compare = |state: FeltBeatState, expected: &serde_json::Value| {
            let actual = [state.c, state.m_plus, state.p_beat];
            if (state.m_plus as f64 - expected[1].as_f64().unwrap()).abs() > maximum[1] {
                worst = Some((
                    state.c,
                    expected[0].as_f64().unwrap(),
                    state.m_plus,
                    expected[1].as_f64().unwrap(),
                    state.phase_offset,
                ));
            }
            for i in 0..3 {
                maximum[i] =
                    maximum[i].max((actual[i] as f64 - expected[i].as_f64().unwrap()).abs());
            }
            probabilities[0].push(1.0 - expected[0].as_f64().unwrap());
            probabilities[1].push(expected[1].as_f64().unwrap());
        };
        for row in case["rows"].as_array().unwrap() {
            let fired = row["fired"].as_bool().unwrap();
            let frac = row["frac"].as_f64().unwrap() as f32;
            if fired {
                let mut probe = f.clone_for_test();
                probe.interval(dt / period * frac, period, 1.0);
                probe.readout(period, 1.0, 0.0);
                compare(probe.state(), &row["pre"]);
                probe.event(period, 1.0, dt / period);
                probe.readout(period, 1.0, 0.0);
                compare(probe.state(), &row["post"]);
            }
            f.process(dt, 1.0 / period, Onset { fired, frac }, false, 0.0, 0.0);
            compare(f.state(), &row["end"]);
        }
        // The half-cycle template needs 2048 points: at 1024, T=2 m_plus
        // differed by 0.00472, exceeding this witness SD of 0.00269.
        // Match the consumer's actual five-seed sampling variation on this same
        // immutable witness. Pre/post values are sampled at identical event times.
        let spread = probabilities.map(|p| {
            let draws: Vec<f64> = (1..=5)
                .map(|seed| {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                    p.iter().filter(|&&p| rng.random::<f64>() < p).count() as f64 / p.len() as f64
                })
                .collect();
            let mean = draws.iter().sum::<f64>() / 5.0;
            (draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 4.0).sqrt()
        });
        println!("grid N={GRID} T={period} max_c_mplus_P={maximum:?} five_seed_sd={spread:?}");
        println!("grid worst mplus state/reference c,mplus,offset={worst:?}");
        accepted &= maximum[0] < spread[0] && maximum[1] < spread[1];
    }
    assert!(accepted);
}

impl FeltBeat {
    fn clone_for_test(&self) -> Self {
        let mut f = Self::new(self.params);
        f.q = self.q;
        f.p0 = self.p0;
        f.tau = self.tau;
        f.seeded = self.seeded;
        f.half = self.half;
        f.state = self.state;
        f
    }
}

fn irregular(n: usize, duration: f32, period: f32) -> Vec<f32> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let gap = crate::core::onset::REFRACTORY_SEC / period;
    // Conditional Poisson order statistics with the detector's dead time.
    let mut out: Vec<f32> = (0..n)
        .map(|_| rng.random::<f32>() * (duration - (n - 1) as f32 * gap))
        .collect();
    out.sort_by(f32::total_cmp);
    for (i, x) in out.iter_mut().enumerate() {
        *x += i as f32 * gap;
    }
    out
}

fn sequence(
    events: &[f32],
    end: f32,
    period: f32,
) -> (FeltBeat, Vec<FeltBeatState>, crate::core::meter::MeterState) {
    let mut f = FeltBeat::new(FeltBeatParams::default());
    f.seed();
    let dt = 512.0 / 48000.0;
    let d = dt / period;
    let mut index = 0;
    let mut states = Vec::new();
    let mut old = crate::core::meter::MeterNetwork::new();
    let mut old_state = crate::core::meter::MeterState::default();
    for i in 0..(end / d).ceil() as usize {
        let start = i as f32 * d;
        let fired = index < events.len() && events[index] < start + d;
        let frac = if fired {
            (events[index] - start) / d
        } else {
            0.0
        };
        if fired {
            index += 1;
        }
        f.process(
            dt,
            1.0 / period,
            Onset {
                fired,
                frac: frac.clamp(0.0, 1.0),
            },
            false,
            0.0,
            0.0,
        );
        old_state = old.process(dt, if fired { 1.0 } else { 0.0 });
        states.push(f.state());
    }
    assert_eq!(index, events.len());
    (f, states, old_state)
}

#[test]
fn causal_silence_missing_pulse_and_rain() {
    for period in [0.25, 0.5, 1.0, 2.0] {
        let regular: Vec<f32> = (1..=48).map(|i| i as f32).collect();
        let rain = irregular(48, 48.0, period);
        let predicted = |f: &FeltBeat, shift: f32| {
            let mut rate =
                f.p0 * (f.params.lambda_0 * period + f.params.lambda + f.params.weak_lambda);
            for i in 0..GRID {
                rate += f.q[i] * f.rate(i as f32 / GRID as f32 + shift, period, 1.0);
            }
            rate
        };
        for gap in [1.0, 2.0] {
            let (a, _, oa) = sequence(&regular, 48.0 + gap, period);
            let (b, _, ob) = sequence(&rain, 48.0 + gap, period);
            let ratio = |f: &FeltBeat| {
                let x = f.state().phase_offset / std::f32::consts::TAU;
                predicted(f, -x) / predicted(f, 0.5 - x)
            };
            println!(
                "grid silence T={period} gap={gap} ratio={}/{} old_conf={}/{}",
                ratio(&a),
                ratio(&b),
                oa.beat.confidence,
                ob.beat.confidence
            );
            assert!(ratio(&a) > ratio(&b));
        }
        let shape = [0.0, 1.0, 2.5, 3.0, 4.5, 5.5, 6.0, 7.5];
        let missing: Vec<f32> = (0..6)
            .flat_map(|i| shape.map(|x| x + 8.0 * i as f32))
            .collect();
        let (_, ms, om) = sequence(&missing, 48.0, period);
        let (_, rs, or) = sequence(&rain, 48.0, period);
        let (_, iso, _) = sequence(&regular, 48.01, period);
        let stats = |states: &[FeltBeatState]| {
            let last = &states[states.len() / 2..];
            let mut p: Vec<f32> = last.iter().map(|s| s.p_beat).collect();
            p.sort_by(f32::total_cmp);
            let mut c: Vec<f32> = last.iter().map(|s| s.c).collect();
            c.sort_by(f32::total_cmp);
            (
                p[p.len() / 2],
                c[c.len() / 2],
                last.iter().filter(|s| s.p_beat > 0.5).count() as f32 / last.len() as f32,
            )
        };
        let (m, r, i) = (stats(&ms), stats(&rs), stats(&iso));
        println!(
            "grid causal T={period} missing={m:?} rain={r:?} iso={i:?} old_missing_rain_conf={}/{}",
            om.beat.confidence, or.beat.confidence
        );
        assert!(m.0 > r.0);
        assert!(r.1 < i.1 && r.2 < i.2);
        for every in [2, 4] {
            let sparse: Vec<f32> = (0..48).step_by(every).map(|x| x as f32).collect();
            let (_, s, _) = sequence(&sparse, 48.0, period);
            let s = stats(&s);
            println!("grid sparse T={period} every={every} stats={s:?}");
            assert!(r.1 < s.1 && r.2 < s.2);
        }
    }
}

#[test]
fn causal_offbeats_move_the_reading_no_later_than_mixed_onsets() {
    for period in [0.25, 0.5, 1.0, 2.0] {
        let trial = |off: usize| {
            let mut events: Vec<f32> = (0..32).map(|x| x as f32).collect();
            events.extend((0..48).map(|x| 32.0 + x as f32 + if x % 4 < off { 0.5 } else { 0.0 }));
            let (_, states, _) = sequence(&events, 80.0, period);
            let start = (32.0 * period / (512.0 / 48000.0)) as usize;
            states[start..]
                .iter()
                .position(|s| s.phase_offset.cos() < 0.0)
                .map(|i| i as f32 * (512.0 / 48000.0) / period)
        };
        let (pure, mixed) = (trial(4), trial(3));
        println!("grid offbeats T={period} pure={pure:?} mixed={mixed:?}");
        assert!(pure.is_some());
        assert!(mixed.is_none() || pure.unwrap() <= mixed.unwrap());
    }
}

#[cfg(feature = "profile-alloc")]
#[test]
fn enabled_habitat_meter_has_zero_steady_state_allocations() {
    use crate::runtime_profile::{begin_allocations, finish_allocations};
    for (stability, guard, input) in [
        (0.0, false, false),
        (1.0, false, false),
        (0.0, true, false),
        (1.0, true, false),
        (0.0, true, true),
        (1.0, true, true),
    ] {
        let mut m = crate::core::meter::MeterNetwork::new();
        m.set_shaping(crate::core::meter::MeterShaping {
            stability,
            basin_hz: None,
        });
        m.enable_felt_beat(FeltBeatParams::default());
        if guard {
            m.enable_felt_period_guard();
        }
        if input {
            m.enable_felt_input(48000.0, 480);
            m.set_felt_bands(480, Some([0.0; 3]));
            m.process_at(0, 480, 48000.0, Some(0.0));
        } else {
            m.process(0.01, 0.0);
        }
        let mut standing_hops = 0;
        begin_allocations();
        for i in 0..8000 {
            let drive = if i % 50 == 0 { 1.0 } else { 0.0 };
            if input {
                m.set_felt_bands((i + 2) * 480, Some([drive, 0.0, 0.0]));
                std::hint::black_box(m.process_at((i + 1) * 480, 480, 48000.0, Some(drive)));
            } else {
                std::hint::black_box(m.process(0.01, drive));
            }
            let felt = std::hint::black_box(m.felt_beat()).unwrap();
            standing_hops += usize::from(!felt.unknown && felt.p_beat >= 0.5);
        }
        let c = finish_allocations().unwrap();
        assert_eq!((c.count, c.bytes), (0, 0));
        assert!(standing_hops > 0, "exercise the standing-beat guard");
        println!(
            "grid allocations stability={stability} guard={guard} input={input} standing_hops={standing_hops}: 0 / 0"
        );
    }
}
