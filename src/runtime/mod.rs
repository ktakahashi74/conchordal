use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::*;

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use ringbuf::traits::Observer;

use crate::audio::limiter::{LimiterMeter, LimiterMode};
use crate::core::analysis_worker;
use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::log2space::Log2Space;
use crate::core::meter::MeterNetwork;
use crate::core::modulation::NeuralRhythms;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
use crate::core::nsgt_rt::{RtConfig, RtNsgtKernelLog2};
use crate::core::phase::wrap_pm_pi;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::stream::{analysis::AnalysisStream, dorsal::DorsalStream};
use crate::core::temporal_expectation::TemporalObservation;
use crate::core::timebase::Tick;
use crate::dcc_coupler::{DccCoupler, ListenerPressure};
use crate::life::community::Community;
use crate::life::conductor::Conductor;
use crate::life::phonation_engine::ParticipationPrediction;
use crate::life::report::{
    HopTimingSample, JsonlReporter, ListenerContourSample, ListenerStateSample,
    ParticipationOutcomeSample, RhythmObservation, onset_samples_from_batches, scaffold_phase_0_1,
    summarize_populations,
};
use crate::life::schedule_renderer::ScheduleRenderer;
use crate::life::voice::{PhonationBatch, SoundBody};
use crate::listener_twin::{ListenerFastState, ListenerState, ListenerTwin};
use crate::runtime_profile::{HopProfile, RunProfile, begin_allocations, finish_allocations};
use crate::scenario::{Action, ScaffoldConfig, Scenario};
use crate::scripting::ScriptHost;
use crate::viewdata::{
    DorsalFrame, ListenerFrame, PlaybackState, PredictionFrame, SimulationMeta, SpecFrame, UiFrame,
    VoiceStateInfo, WaveFrame,
};
use crate::{
    audio::output::AudioOutput, config::AppConfig, core::harmonicity_kernel::HarmonicityParams,
};

const MAX_LANDSCAPE_LAG_FRAMES: u64 = 1;
#[cfg(test)]
thread_local! {
    static UI_FRAME_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const UI_CHANNEL_CAPACITY: usize = 16;
const UI_MIN_INTERVAL: Duration = Duration::from_millis(33);

mod analysis_sync;
mod monitor;
mod phases;
mod scenario_load;
mod ui_frame;
mod wiring;
mod worker;

use analysis_sync::*;
use monitor::*;
use phases::*;
use scenario_load::*;
use ui_frame::*;
#[cfg(test)]
use wiring::*;
use worker::*;

pub use scenario_load::{compile_scenario_from_script, run_compile_only};
pub(crate) use wiring::{init_runtime, join_runtime, join_thread};
pub use wiring::{run_headless, run_render};

fn report_try<F>(reporter: &mut Option<JsonlReporter>, label: &str, f: F)
where
    F: FnOnce(&mut JsonlReporter) -> Result<(), String>,
{
    let Some(writer) = reporter.as_mut() else {
        return;
    };
    if writer.failure.is_some() {
        return;
    }
    if let Err(err) = f(writer) {
        warn!("report {label} failed: {err}");
        writer.failure = Some(err);
    }
}

#[cfg(test)]
pub(crate) fn birth_surrogate_assay_core(
    config: &AppConfig,
    runtime_sample_rate: u32,
) -> (LandscapeParams, RtNsgtKernelLog2, usize) {
    let core = build_analysis_runtime_core(config, runtime_sample_rate);
    (core.lparams, core.nsgt, core.hop)
}

#[cfg(test)]
mod perceptual_action_assay;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lowrange_tests;

#[cfg(test)]
type OfflineBodyProbe = Box<dyn FnMut(&mut WorkerState, Tick, usize, [&[f32]; 2]) + Send>;

#[cfg(test)]
mod body_metabolism_assay;

#[cfg(test)]
pub(crate) fn body_metabolism_test_core() -> (RtNsgtKernelLog2, LandscapeParams) {
    let mut config = AppConfig::default();
    config.psychoacoustics.habituation.enabled = false;
    let core = build_analysis_runtime_core(&config, 48_000);
    (core.nsgt, core.lparams)
}

#[cfg(test)]
mod meter_recurrence_tests;

#[cfg(test)]
mod articulation_persistence_tests;

#[cfg(test)]
#[path = "temporal_meter_tests.rs"]
mod temporal_meter_tests;

#[cfg(test)]
mod shutdown_tests;

#[cfg(test)]
pub(crate) mod l1_probe;

#[cfg(test)]
#[path = "../../tests/support/test_output_directory.rs"]
mod test_output_directory;
