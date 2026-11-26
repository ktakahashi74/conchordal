//! Display-only wire contract. Version 2 carries no engine state or audio output API.
//! A packet is a little-endian u32 header length, a JSON header without float arrays,
//! then the arrays as little-endian binary in the order of `arrays`. Frequency axes
//! travel only when they change; `Decoder` restores them from the last packet that had them.
use std::sync::Arc;
pub const RHYTHM_HISTORY_SEC: f64 = 30.;
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WaveFrame {
    pub fs: f32,
    pub samples: Arc<[f32]>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SpecFrame {
    pub spec_hz: Vec<f32>,
    pub amps: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DorsalFrame {
    pub e_low: f32,
    pub e_mid: f32,
    pub e_high: f32,
    pub flux: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ListenerRhythmHop {
    pub time_sec: f64,
    pub attention: DorsalFrame,
    pub meter: MeterState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerForecast {
    pub version: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub reference_sample: u64,
    pub reference_bin: Option<usize>,
    pub horizon_sec: f32,
    pub p_known: Option<[f32; 5]>,
    pub q: Option<[f32; 5]>,
    pub marginal: Option<[f32; 5]>,
    pub entropy_nats: Option<f32>,
    pub coverage_score: f32,
    pub arrival_score: Option<f32>,
    pub examples: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerOutcome {
    pub forecast: ListenerForecast,
    pub outcome: usize,
    pub observed_sample: u64,
    pub surprise_nats: Option<f32>,
    pub zero_likelihood: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerCensored {
    pub forecast: ListenerForecast,
    pub observed_sample: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerPrediction {
    pub forecast: Option<ListenerForecast>,
    pub completed: Option<ListenerOutcome>,
    pub censored: Option<ListenerCensored>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ListenerFrame {
    pub prediction: ListenerPrediction,
    pub time_sec: f64,
    pub generated_frame_id: u64,
    pub analysis_frame_id: u64,
    pub analysis_lag_frames: u64,
    pub attention_level: f32,
    pub attention: DorsalFrame,
    pub meter: MeterState,
    pub stability_level: f32,
    pub nearby_improvement_level: f32,
    pub static_pull_level: f32,
    pub has_fast_state: bool,
    pub has_state: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlaybackState {
    #[default]
    NotStarted,
    Playing,
    Ending,
    Finished,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PredictionFrame {
    pub next_gate_tick_est: Option<u64>,
    pub theta_hz: Option<f32>,
    pub delta_hz: Option<f32>,
    pub pred_n_theta_per_delta: Option<u32>,
    pub pred_tau_tick: Option<u64>,
    pub pred_horizon_tick: Option<u64>,
    pub pred_c_field_level_next_gate: Option<Arc<[f32]>>,
    pub pred_gain_raw_mean: Option<f32>,
    pub pred_gain_raw_min: Option<f32>,
    pub pred_gain_raw_max: Option<f32>,
    pub pred_gain_mixed_mean: Option<f32>,
    pub pred_gain_mixed_min: Option<f32>,
    pub pred_gain_mixed_max: Option<f32>,
    pub pred_sync_mean: Option<f32>,
    pub gate_boundary_in_hop: Option<bool>,
    pub pred_available_in_hop: Option<bool>,
    pub phonation_onsets_in_hop: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MeterBand {
    /// Wrapped oscillator phase in [-pi, pi].
    pub phase: f32,
    /// Current frequency in Hz.
    pub freq_hz: f32,
    /// Limit-cycle amplitude (presence). Not salience.
    pub amplitude: f32,
    /// Entrainment confidence in [0, 1] from phase and pattern recurrence. This is the
    /// salience: a clean quiet beat reads high, a loud wandering beat reads low.
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MeterState {
    /// Tactus (delta-rate beat).
    pub beat: MeterBand,
    /// Tatum (theta-rate), mode-locked to the beat at the emergent ratio.
    pub subdivision: MeterBand,
    /// Measure: a slow subharmonic of the beat, induced by periodic accent.
    pub measure: MeterBand,
    /// Onset / flux drive salience (the force term), kept distinct from
    /// confidence so reports and UI never blur drive with entrainment.
    pub attention_level: f32,
    /// Emergent integer subdivision ratio (2, 3, or 4); 0 when none is detected.
    pub subdivision_ratio: u8,
    /// Emergent beats-per-measure (2, 3, or 4); 0 when no accent grouping holds.
    pub measure_ratio: u8,
}
impl Default for WaveFrame {
    fn default() -> Self {
        Self {
            fs: 0.0,
            samples: Arc::from(Vec::<f32>::new()),
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SimulationMeta {
    pub script: Option<ScriptStatus>,
    pub time_sec: f64,
    pub end: Option<f64>,
    pub population_count: usize,
    pub slow_hops: u64,
    pub voice_count: usize,
    pub event_queue_len: usize,
    pub estimator_freezes: u64,
    pub estimator_last_frozen_issue: u64,
    pub audio_xruns: u64,
    pub audio_rebuilds: u64,
    pub audio_recovering: bool,
    pub peak_level: f32,
    pub scenario_name: String,
    pub scene_name: Option<String>,
    pub playback_state: PlaybackState,
    pub channel_peak: [f32; 2],
    pub window_peak: [f32; 2],
    pub kuramoto_order_r: Option<f32>,
    pub kuramoto_active_count: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ScriptStatus {
    pub evaluated_through: f64,
    pub done: bool,
    pub stopped: bool,
    pub error: Option<String>,
    pub error_position: PositionLabel,
    pub warnings: Vec<ScriptWarning>,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScriptWarning {
    pub position: PositionLabel,
    pub message: String,
    pub count: u64,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Diagnostic {
    Heading(String),
    Label(String),
    Separator,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Space {
    pub centers_hz: Vec<f32>,
    pub centers_log2: Vec<f32>,
    pub step_log2: f32,
}
impl Space {
    pub fn step(&self) -> f32 {
        self.step_log2
    }
    pub fn n_bins(&self) -> usize {
        self.centers_hz.len()
    }
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Geometry {
    pub root_start_log2: f32,
    pub space: Space,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    pub geometry: Geometry,
    pub p: Vec<f64>,
    pub mass: f64,
    pub focus: f64,
    pub conflict: f64,
    pub end_sample: u64,
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ConsonanceKernel {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}
impl ConsonanceKernel {
    pub fn score(&self, h: f32, r: f32) -> f32 {
        let finite = |x: f32| if x.is_finite() { x } else { 0. };
        let h = finite(h).clamp(0., 1.);
        let r = finite(r).clamp(0., 1.);
        finite(finite(self.a) * h + finite(self.b) * r + finite(self.c) * h * r + finite(self.d))
    }
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Landscape {
    /// Keep the acoustic fields at landscape level in the packet header.
    #[serde(flatten)]
    pub acoustic: AcousticField,
    /// Keep support at landscape level in the packet header.
    #[serde(flatten)]
    pub harmonic: HarmonicExtensions,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AcousticField {
    pub space: Space,
    pub subjective_intensity: Vec<f32>,
    pub nsgt_power: Vec<f32>,
    pub consonance_field_score: Vec<f32>,
    pub consonance_field_level: Vec<f32>,
    pub harmonicity01: Vec<f32>,
    pub roughness01: Vec<f32>,
    pub consonance_kernel: ConsonanceKernel,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HarmonicExtensions {
    pub support: Option<Arc<Snapshot>>,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct VoiceStateInfo {
    pub id: u64,
    pub freq_hz: f32,
    pub target_freq: f32,
    pub integration_window: f32,
    pub consonance: f32,
    pub support_template: Option<Arc<[(isize, f32)]>>,
    pub support_scans: Option<[Vec<f64>; 3]>,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct UiFrame {
    pub wave: WaveFrame,
    pub spec: SpecFrame,
    pub listener: ListenerFrame,
    pub listener_rhythm: Vec<ListenerRhythmHop>,
    pub landscape: Landscape,
    pub meta: SimulationMeta,
    pub prediction: PredictionFrame,
    pub voices: Vec<VoiceStateInfo>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Envelope {
    pub version: u32,
    /// The frequency axes are in this packet's arrays.
    pub axes: bool,
    pub frame: UiFrame,
}

enum Array<'a> {
    F32(&'a mut Vec<f32>),
    Shared(&'a mut Arc<[f32]>),
    F64(&'a mut Vec<f64>),
}

/// Every float array in a fixed order; axes only when the packet carries them.
fn arrays(
    frame: &mut UiFrame,
    axes: bool,
    visit: &mut dyn FnMut(Array) -> Result<(), String>,
) -> Result<(), String> {
    visit(Array::Shared(&mut frame.wave.samples))?;
    visit(Array::F32(&mut frame.spec.amps))?;
    let l = &mut frame.landscape;
    if axes {
        visit(Array::F32(&mut frame.spec.spec_hz))?;
        visit(Array::F32(&mut l.acoustic.space.centers_hz))?;
        visit(Array::F32(&mut l.acoustic.space.centers_log2))?;
    }
    for scan in [
        &mut l.acoustic.subjective_intensity,
        &mut l.acoustic.nsgt_power,
        &mut l.acoustic.consonance_field_score,
        &mut l.acoustic.consonance_field_level,
        &mut l.acoustic.harmonicity01,
        &mut l.acoustic.roughness01,
    ] {
        visit(Array::F32(scan))?;
    }
    if let Some(support) = &mut l.harmonic.support {
        let support = Arc::make_mut(support);
        visit(Array::F64(&mut support.p))?;
        if axes {
            visit(Array::F32(&mut support.geometry.space.centers_hz))?;
            visit(Array::F32(&mut support.geometry.space.centers_log2))?;
        }
    }
    if let Some(scan) = &mut frame.prediction.pred_c_field_level_next_gate {
        visit(Array::Shared(scan))?;
    }
    for voice in &mut frame.voices {
        for scan in voice.support_scans.iter_mut().flatten() {
            visit(Array::F64(scan))?;
        }
    }
    Ok(())
}

/// The frequency axes of the last packet that carried them.
struct Axes {
    spec_hz: Vec<f32>,
    space: (Vec<f32>, Vec<f32>),
    support: Option<(Vec<f32>, Vec<f32>)>,
}

fn space_axes(space: &Space) -> (Vec<f32>, Vec<f32>) {
    (space.centers_hz.clone(), space.centers_log2.clone())
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[derive(Default)]
pub struct Encoder {
    axes: Option<Axes>,
}

impl Encoder {
    pub fn encode(&mut self, mut frame: UiFrame) -> Result<Vec<u8>, serde_json::Error> {
        let support = frame
            .landscape
            .harmonic
            .support
            .as_ref()
            .map(|s| &s.geometry.space);
        let unchanged = self.axes.as_ref().is_some_and(|axes| {
            same_bits(&axes.spec_hz, &frame.spec.spec_hz)
                && same_bits(&axes.space.0, &frame.landscape.acoustic.space.centers_hz)
                && same_bits(&axes.space.1, &frame.landscape.acoustic.space.centers_log2)
                && support.is_none_or(|space| {
                    axes.support.as_ref().is_some_and(|(hz, log2)| {
                        same_bits(hz, &space.centers_hz) && same_bits(log2, &space.centers_log2)
                    })
                })
        });
        if !unchanged {
            let previous = self.axes.take().and_then(|axes| axes.support);
            self.axes = Some(Axes {
                spec_hz: frame.spec.spec_hz.clone(),
                space: space_axes(&frame.landscape.acoustic.space),
                support: support.map(space_axes).or(previous),
            });
        }
        let mut binary = Vec::new();
        arrays(&mut frame, !unchanged, &mut |array| {
            match array {
                Array::F32(v) => {
                    binary.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    binary.extend(std::mem::take(v).iter().flat_map(|x| x.to_le_bytes()));
                }
                Array::Shared(v) => {
                    binary.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    binary.extend(v.iter().flat_map(|x| x.to_le_bytes()));
                    *v = Arc::from([]);
                }
                Array::F64(v) => {
                    binary.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    binary.extend(std::mem::take(v).iter().flat_map(|x| x.to_le_bytes()));
                }
            }
            Ok(())
        })
        .expect("encoding visits cannot fail");
        if unchanged {
            frame.spec.spec_hz.clear();
            frame.landscape.acoustic.space.centers_hz.clear();
            frame.landscape.acoustic.space.centers_log2.clear();
            if let Some(support) = &mut frame.landscape.harmonic.support {
                let space = &mut Arc::make_mut(support).geometry.space;
                space.centers_hz.clear();
                space.centers_log2.clear();
            }
        }
        let header = serde_json::to_vec(&Envelope {
            version: 2,
            axes: !unchanged,
            frame,
        })?;
        let mut out = Vec::with_capacity(4 + header.len() + binary.len());
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&binary);
        Ok(out)
    }
}

fn take_array<'a>(
    rest: &mut &'a [u8],
    width: usize,
) -> Result<std::slice::ChunksExact<'a, u8>, String> {
    let truncated = || "Truncated UI snapshot".to_string();
    let count = rest.get(..4).ok_or_else(truncated)?;
    let size = (u32::from_le_bytes(count.try_into().unwrap()) as usize)
        .checked_mul(width)
        .ok_or_else(truncated)?;
    let data = rest.get(4..4 + size).ok_or_else(truncated)?;
    *rest = &rest[4 + size..];
    Ok(data.chunks_exact(width))
}

fn f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

#[derive(Default)]
pub struct Decoder {
    axes: Option<Axes>,
}

impl Decoder {
    pub fn decode(&mut self, bytes: &[u8]) -> Result<UiFrame, String> {
        let truncated = || "Truncated UI snapshot".to_string();
        let length = bytes.get(..4).ok_or_else(truncated)?;
        let length = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        let header = bytes.get(4..4 + length).ok_or_else(truncated)?;
        let packet: Envelope = serde_json::from_slice(header).map_err(|e| e.to_string())?;
        if packet.version != 2 {
            return Err(format!(
                "Unsupported UI snapshot version {}",
                packet.version
            ));
        }
        let mut frame = packet.frame;
        let mut rest = &bytes[4 + length..];
        arrays(&mut frame, packet.axes, &mut |array| {
            match array {
                Array::F32(v) => *v = take_array(&mut rest, 4)?.map(f32_le).collect(),
                Array::Shared(v) => *v = take_array(&mut rest, 4)?.map(f32_le).collect(),
                Array::F64(v) => {
                    *v = take_array(&mut rest, 8)?
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                }
            }
            Ok(())
        })?;
        if !rest.is_empty() {
            return Err("Trailing bytes in UI snapshot".into());
        }
        let support = frame.landscape.harmonic.support.as_mut().map(Arc::make_mut);
        if packet.axes {
            let previous = self.axes.take().and_then(|axes| axes.support);
            self.axes = Some(Axes {
                spec_hz: frame.spec.spec_hz.clone(),
                space: space_axes(&frame.landscape.acoustic.space),
                support: support.map(|s| space_axes(&s.geometry.space)).or(previous),
            });
        } else {
            let axes = self
                .axes
                .as_ref()
                .ok_or("UI snapshot refers to frequency axes it never received")?;
            frame.spec.spec_hz.clone_from(&axes.spec_hz);
            (
                frame.landscape.acoustic.space.centers_hz,
                frame.landscape.acoustic.space.centers_log2,
            ) = axes.space.clone();
            if let Some(support) = support {
                let (hz, log2) = axes
                    .support
                    .clone()
                    .ok_or("UI snapshot refers to support axes it never received")?;
                (
                    support.geometry.space.centers_hz,
                    support.geometry.space.centers_log2,
                ) = (hz, log2);
            }
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unknown_version_missing_axes_and_truncated_packet() {
        let header = serde_json::to_vec(&Envelope {
            version: 3,
            axes: true,
            frame: UiFrame::default(),
        })
        .unwrap();
        let mut bytes = (header.len() as u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        assert!(
            Decoder::default()
                .decode(&bytes)
                .unwrap_err()
                .contains("version 3")
        );
        let mut encoder = Encoder::default();
        let first = encoder.encode(UiFrame::default()).unwrap();
        let second = encoder.encode(UiFrame::default()).unwrap();
        assert!(Decoder::default().decode(&second).is_err());
        assert!(
            Decoder::default()
                .decode(&first[..first.len() / 2])
                .is_err()
        );
        let mut decoder = Decoder::default();
        decoder.decode(&first).unwrap();
        decoder.decode(&second).unwrap();
    }
}

/// Already formatted by the producer; preserve Rhai's diagnostic spelling.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PositionLabel(pub String);
impl std::fmt::Debug for PositionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
