use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::life::individual::AgentMetadata;
use crate::life::individual::{IndividualWrapper, SoundBody};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::scenario::{BrainConfig, IndividualConfig};

fn make_landscape() -> Landscape {
    let fs = 48_000.0;
    let space = Log2Space::new(55.0, 4000.0, 48);
    let params = LandscapeParams {
        fs,
        max_hist_cols: 64,
        alpha: 0.0,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.01),
        harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
        habituation_tau: 8.0,
        habituation_weight: 0.5,
        habituation_max_depth: 1.0,
        loudness_exp: 0.23,
        tau_ms: 60.0,
        ref_power: 1e-6,
        roughness_k: 0.1,
    };
    let nsgt_kernel = NsgtKernelLog2::new(
        NsgtLog2Config {
            fs,
            overlap: 0.5,
            nfft_override: Some(1024),
        },
        space,
    );
    let nsgt = RtNsgtKernelLog2::new(nsgt_kernel);
    Landscape::new(params, nsgt)
}

fn spawn_agent(freq: f32, id: u64) -> crate::life::individual::PureTone {
    let cfg = IndividualConfig::PureTone {
        freq,
        amp: 0.5,
        phase: None,
        rhythm_freq: None,
        rhythm_sensitivity: None,
        commitment: None,
        habituation_sensitivity: None,
        brain: BrainConfig::Entrain {
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
                attack_sec: crate::life::lifecycle::default_decay_attack(),
            },
        },
        tag: None,
    };
    let meta = AgentMetadata {
        id,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    match cfg.spawn(id, 0, meta) {
        IndividualWrapper::PureTone(ind) => ind,
        _ => unreachable!(),
    }
}

#[test]
fn test_inertia_calculation() {
    let landscape = make_landscape();
    let mut low = spawn_agent(60.0, 1);
    let mut high = spawn_agent(1000.0, 2);
    let rhythms = NeuralRhythms::default();

    low.update_organic_movement(&rhythms, 0.01, &landscape);
    high.update_organic_movement(&rhythms, 0.01, &landscape);

    assert!(
        low.integration_window > high.integration_window,
        "expected heavier (low) pitch to integrate longer than high pitch"
    );
}

#[test]
fn test_scan_logic() {
    let mut landscape = make_landscape();
    let mut agent = spawn_agent(220.0, 3);
    let mut frame = landscape.snapshot();
    let n = frame.c_last.len();
    frame.amps_last = vec![1.0; n];
    frame.c_last.fill(-5.0);
    let idx_cur = frame
        .space
        .index_of_freq(agent.body.base_freq_hz())
        .unwrap_or(0);
    frame.c_last[idx_cur] = -5.0;
    let target_alt = agent.body.base_freq_hz() * 1.5;
    if let Some(idx_alt) = frame.space.index_of_freq(target_alt) {
        if let Some(c) = frame.c_last.get_mut(idx_alt) {
            *c = 5.0;
        }
    }
    landscape.apply_frame(&frame);

    agent.accumulated_time = agent.integration_window;
    agent.last_theta_sample = -0.1;
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.mag = 1.0;
    rhythms.theta.phase = 0.25;

    let before = agent.target_freq;
    agent.update_organic_movement(&rhythms, 0.01, &landscape);
    assert!(
        agent.target_freq > before,
        "agent should move toward higher-scoring neighbor"
    );
}

#[test]
fn test_breath_gating() {
    let landscape = make_landscape();
    let mut agent = spawn_agent(330.0, 4);
    let original = agent.body.base_freq_hz();
    agent.target_freq = original * 1.5;
    agent.breath_gain = 1.0;

    let rhythms = NeuralRhythms::default();
    agent.update_organic_movement(&rhythms, 0.05, &landscape);

    assert!(
        agent.breath_gain < 1.0,
        "breath gain should fall before snapping to target"
    );
    let after_freq = agent.body.base_freq_hz();
    assert!(
        (after_freq - original).abs() < 1e-3,
        "frequency should remain until breath collapses ({} vs {})",
        after_freq,
        original
    );
}
