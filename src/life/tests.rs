use super::conductor::Conductor;
use super::population::{Population, PopulationParams};
use super::scenario::{Action, AgentConfig, Episode, Event, Scenario};
use crate::core::landscape::LandscapeFrame;
use crate::life::lifecycle::LifecycleConfig;
use crate::core::harmonicity_kernel::HarmonicityParams;
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};

#[test]
fn test_population_add_remove_agent() {
    // 1. Setup
    let params = PopulationParams {
        initial_tones_hz: vec![],
        amplitude: 0.1,
    };
    let mut pop = Population::new(params);
    let landscape = LandscapeFrame::default();

    assert_eq!(pop.agents.len(), 0, "Population should start empty");

    // 2. Add Agent
    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 1.0,
        attack_sec: 0.01,
    };
    let agent_cfg = AgentConfig::PureTone {
        freq: 440.0,
        amp: 0.5,
        phase: None,
        lifecycle: life,
        tag: Some("test_agent".to_string()),
    };
    let action_add = Action::AddAgent { agent: agent_cfg };

    pop.apply_action(action_add, &landscape);
    assert_eq!(pop.agents.len(), 1, "Agent should be added");

    // 3. Remove Agent
    let action_remove = Action::RemoveAgent {
        target: "test_agent".to_string(),
    };
    pop.apply_action(action_remove, &landscape);
    assert_eq!(pop.agents.len(), 0, "Agent should be removed");
}

#[test]
fn test_conductor_timing() {
    // 1. Create a Scenario with an event at T=1.0s
    let action = Action::Finish; // Simple marker action
    let event = Event {
        time: 1.0,
        repeat: None,
        actions: vec![action],
    };
    let episode = Episode {
        name: Some("test".into()),
        start_time: 0.0,
        events: vec![event],
    };
    let scenario = Scenario {
        episodes: vec![episode],
    };

    let mut conductor = Conductor::from_scenario(scenario);
    let mut pop = Population::new(PopulationParams {
        initial_tones_hz: vec![],
        amplitude: 0.0,
    });
    let landscape = LandscapeFrame::default();

    // 2. Dispatch at T=0.5 (Should NOT fire)
    conductor.dispatch_until(0.5, 0, &landscape, &mut pop);
    assert!(
        !pop.abort_requested,
        "Finish action should not fire yet at T=0.5"
    );

    // 3. Dispatch at T=1.1 (Should fire)
    conductor.dispatch_until(1.1, 100, &landscape, &mut pop);
    assert!(
        pop.abort_requested,
        "Finish action should fire at T=1.1"
    );
}

fn make_test_landscape(fs: f32) -> Landscape {
    let space = Log2Space::new(55.0, 4000.0, 64);
    let lparams = LandscapeParams {
        fs,
        max_hist_cols: 64,
        alpha: 0.0,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.01),
        harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
        loudness_exp: 0.23,
        tau_ms: 80.0,
        ref_power: 1e-6,
        roughness_k: 0.1,
    };
    let overlap = 0.5;
    let nfft = 2048usize;
    let nsgt_kernel = NsgtKernelLog2::new(
        NsgtLog2Config {
            fs,
            overlap,
            nfft_override: Some(nfft),
        },
        space,
    );
    let nsgt = RtNsgtKernelLog2::new(nsgt_kernel);
    Landscape::new(lparams, nsgt)
}

#[test]
fn test_agent_lifecycle_decay_death() {
    // 1. Setup Population with 1 agent that has a very short half-life
    let params = PopulationParams {
        initial_tones_hz: vec![],
        amplitude: 0.1,
    };
    let mut pop = Population::new(params);
    let landscape = LandscapeFrame::default(); // Dummy landscape

    // Half-life = 0.05s (very fast decay)
    let life = LifecycleConfig::Decay {
        initial_energy: 1.0,
        half_life_sec: 0.05,
        attack_sec: 0.001,
    };
    let agent_cfg = AgentConfig::PureTone {
        freq: 440.0,
        amp: 0.5,
        phase: None,
        lifecycle: life,
        tag: None,
    };
    pop.apply_action(Action::AddAgent { agent: agent_cfg }, &landscape);

    assert_eq!(pop.agents.len(), 1, "Agent added");

    // 2. Simulate time passing via process_frame
    // We need to run enough frames for energy to drop below threshold (1e-4)
    // Energy starts at 1.0.
    // After 0.05s -> 0.5
    // After 0.10s -> 0.25
    // ...
    // After 1.0s -> ~0.0
    let fs = 48_000.0;
    let nfft = 1024;
    let dt = 0.01; // 10ms steps
    let samples_per_hop = (fs * dt) as usize;
    let landscape_rt = make_test_landscape(fs);
    let mut time = 0.0;

    // Run for 1.0 second (should be plenty for 0.05s half-life to die)
    for i in 0..100 {
        pop.process_audio(samples_per_hop, fs, i, dt, &landscape_rt);
        pop.process_frame(i, 513, fs, nfft, dt);
        time += dt;
    }

    // 3. Verify agent is cleaned up
    assert_eq!(
        pop.agents.len(),
        0,
        "Agent should have died due to energy decay after {:.2}s",
        time
    );
}
