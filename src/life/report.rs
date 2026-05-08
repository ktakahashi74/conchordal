use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};

use serde::Serialize;

use crate::core::landscape::LandscapeFrame;
use crate::life::population::{RuntimeEvent, SpawnReason};
use crate::life::scenario::{ScaffoldConfig, SceneMarker};
use crate::life::telemetry::LifeRecord;
use crate::life::voice::sound_body::SoundBody;
use crate::life::voice::{AnyArticulationCore, Voice};

#[derive(Debug)]
pub struct JsonlReporter {
    writer: BufWriter<File>,
    rhythm_onsets: Vec<OnsetSample>,
    rhythm_observations: Vec<RhythmObservation>,
    rhythm_summary_written: bool,
}

#[derive(Debug, Clone)]
pub struct OnsetSample {
    pub time_sec: f32,
    pub group_id: u64,
    pub voice_id: u64,
    pub generation: u32,
    pub freq_hz: f32,
    pub strength: f32,
    pub plv: Option<f32>,
    pub scaffold_mode: &'static str,
    pub scaffold_phase_0_1: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupStepSummary {
    pub time_sec: f32,
    pub group_id: u64,
    pub alive_count: usize,
    pub mean_freq_hz: f32,
    pub mean_c_field_score: f32,
    pub mean_c_field_level: f32,
    pub freq_entropy_log2: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RhythmObservation {
    pub time_sec: f32,
    pub kuramoto_order_r: Option<f32>,
    pub kuramoto_active_count: usize,
    pub onsets_in_hop: usize,
    pub theta_hz: Option<f32>,
    pub delta_hz: Option<f32>,
    pub env_open: f32,
    pub env_level: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RhythmSummary {
    pub time_sec: f32,
    pub window_start_sec: f32,
    pub window_end_sec: f32,
    pub group_id: Option<u64>,
    pub onset_count: usize,
    pub onset_density_hz: f32,
    pub ioi_mean_sec: Option<f32>,
    pub ioi_cv: Option<f32>,
    pub beat_stability: Option<f32>,
    pub mean_plv: Option<f32>,
    pub kuramoto_order_mean: Option<f32>,
    pub kuramoto_order_max: Option<f32>,
    pub sync_emergence_sec: Option<f32>,
    pub burstiness: Option<f32>,
    pub one_over_f_slope: Option<f32>,
    pub ioi_one_over_f_slope: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReportRecord<'a> {
    SceneMarker {
        time_sec: f32,
        order: u64,
        name: &'a str,
    },
    Spawn {
        time_sec: f32,
        group_id: u64,
        voice_id: u64,
        generation: u32,
        member_idx: usize,
        freq_hz: f32,
    },
    Respawn {
        time_sec: f32,
        group_id: u64,
        voice_id: u64,
        generation: u32,
        member_idx: usize,
        freq_hz: f32,
        parent_id: Option<u64>,
    },
    Death {
        time_sec: f32,
        group_id: u64,
        voice_id: u64,
        generation: u32,
        lifetime_sec: f32,
        first_k_mean: f32,
        plv_at_death: Option<f32>,
    },
    Onset {
        time_sec: f32,
        group_id: u64,
        voice_id: u64,
        generation: u32,
        freq_hz: f32,
        strength: f32,
        plv: Option<f32>,
        scaffold_mode: &'a str,
        scaffold_phase_0_1: Option<f32>,
    },
    GroupStep {
        time_sec: f32,
        group_id: u64,
        alive_count: usize,
        mean_freq_hz: f32,
        mean_c_field_score: f32,
        mean_c_field_level: f32,
        freq_entropy_log2: f32,
    },
    RhythmObservation {
        time_sec: f32,
        kuramoto_order_r: Option<f32>,
        kuramoto_active_count: usize,
        onsets_in_hop: usize,
        theta_hz: Option<f32>,
        delta_hz: Option<f32>,
        env_open: f32,
        env_level: f32,
    },
    RhythmSummary {
        time_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        group_id: Option<u64>,
        onset_count: usize,
        onset_density_hz: f32,
        ioi_mean_sec: Option<f32>,
        ioi_cv: Option<f32>,
        beat_stability: Option<f32>,
        mean_plv: Option<f32>,
        kuramoto_order_mean: Option<f32>,
        kuramoto_order_max: Option<f32>,
        sync_emergence_sec: Option<f32>,
        burstiness: Option<f32>,
        one_over_f_slope: Option<f32>,
        ioi_one_over_f_slope: Option<f32>,
    },
}

impl JsonlReporter {
    pub fn create(path: &str) -> Result<Self, String> {
        let file = File::create(path).map_err(|err| format!("create report {path}: {err}"))?;
        Ok(Self {
            writer: BufWriter::new(file),
            rhythm_onsets: Vec::new(),
            rhythm_observations: Vec::new(),
            rhythm_summary_written: false,
        })
    }

    pub fn write_scene_markers(&mut self, markers: &[SceneMarker]) -> Result<(), String> {
        for marker in markers {
            self.write_record(&ReportRecord::SceneMarker {
                time_sec: marker.time,
                order: marker.order,
                name: &marker.name,
            })?;
        }
        Ok(())
    }

    pub fn write_runtime_events(&mut self, events: &[RuntimeEvent]) -> Result<(), String> {
        for event in events {
            match event.reason {
                SpawnReason::Initial => self.write_record(&ReportRecord::Spawn {
                    time_sec: event.time_sec,
                    group_id: event.group_id,
                    voice_id: event.voice_id,
                    generation: event.generation,
                    member_idx: event.member_idx,
                    freq_hz: event.freq_hz,
                })?,
                SpawnReason::Respawn => self.write_record(&ReportRecord::Respawn {
                    time_sec: event.time_sec,
                    group_id: event.group_id,
                    voice_id: event.voice_id,
                    generation: event.generation,
                    member_idx: event.member_idx,
                    freq_hz: event.freq_hz,
                    parent_id: event.parent_id,
                })?,
            }
        }
        Ok(())
    }

    pub fn write_deaths(
        &mut self,
        records: &[LifeRecord],
        hop: usize,
        fs: f32,
    ) -> Result<(), String> {
        let frame_sec = hop as f32 / fs.max(1.0);
        for record in records {
            self.write_record(&ReportRecord::Death {
                time_sec: record.death_frame as f32 * frame_sec,
                group_id: record.group_id,
                voice_id: record.voice_id,
                generation: record.generation,
                lifetime_sec: record.lifetime_ticks as f32 * frame_sec,
                first_k_mean: record.c_level_firstk_mean,
                plv_at_death: record.plv_at_death,
            })?;
        }
        Ok(())
    }

    pub fn write_onsets(&mut self, onsets: &[OnsetSample]) -> Result<(), String> {
        for onset in onsets {
            self.write_record(&ReportRecord::Onset {
                time_sec: onset.time_sec,
                group_id: onset.group_id,
                voice_id: onset.voice_id,
                generation: onset.generation,
                freq_hz: onset.freq_hz,
                strength: onset.strength,
                plv: onset.plv,
                scaffold_mode: onset.scaffold_mode,
                scaffold_phase_0_1: onset.scaffold_phase_0_1,
            })?;
        }
        if !onsets.is_empty() {
            self.rhythm_onsets.extend_from_slice(onsets);
            self.rhythm_summary_written = false;
        }
        Ok(())
    }

    pub fn write_group_steps(&mut self, steps: &[GroupStepSummary]) -> Result<(), String> {
        for step in steps {
            self.write_record(&ReportRecord::GroupStep {
                time_sec: step.time_sec,
                group_id: step.group_id,
                alive_count: step.alive_count,
                mean_freq_hz: step.mean_freq_hz,
                mean_c_field_score: step.mean_c_field_score,
                mean_c_field_level: step.mean_c_field_level,
                freq_entropy_log2: step.freq_entropy_log2,
            })?;
        }
        Ok(())
    }

    pub fn write_rhythm_observation(
        &mut self,
        observation: RhythmObservation,
    ) -> Result<(), String> {
        self.write_record(&ReportRecord::RhythmObservation {
            time_sec: observation.time_sec,
            kuramoto_order_r: observation.kuramoto_order_r,
            kuramoto_active_count: observation.kuramoto_active_count,
            onsets_in_hop: observation.onsets_in_hop,
            theta_hz: observation.theta_hz,
            delta_hz: observation.delta_hz,
            env_open: observation.env_open,
            env_level: observation.env_level,
        })?;
        self.rhythm_observations.push(observation);
        self.rhythm_summary_written = false;
        Ok(())
    }

    pub fn write_rhythm_summary(&mut self) -> Result<(), String> {
        if self.rhythm_summary_written {
            return Ok(());
        }
        let summary = summarize_rhythm(&self.rhythm_onsets, &self.rhythm_observations, None);
        self.write_summary_record(&summary)?;

        let mut groups = BTreeSet::new();
        for onset in &self.rhythm_onsets {
            groups.insert(onset.group_id);
        }
        for group_id in groups {
            let summary = summarize_rhythm(
                &self.rhythm_onsets,
                &self.rhythm_observations,
                Some(group_id),
            );
            self.write_summary_record(&summary)?;
        }
        self.rhythm_summary_written = true;
        Ok(())
    }

    fn write_summary_record(&mut self, summary: &RhythmSummary) -> Result<(), String> {
        self.write_record(&ReportRecord::RhythmSummary {
            time_sec: summary.time_sec,
            window_start_sec: summary.window_start_sec,
            window_end_sec: summary.window_end_sec,
            group_id: summary.group_id,
            onset_count: summary.onset_count,
            onset_density_hz: summary.onset_density_hz,
            ioi_mean_sec: summary.ioi_mean_sec,
            ioi_cv: summary.ioi_cv,
            beat_stability: summary.beat_stability,
            mean_plv: summary.mean_plv,
            kuramoto_order_mean: summary.kuramoto_order_mean,
            kuramoto_order_max: summary.kuramoto_order_max,
            sync_emergence_sec: summary.sync_emergence_sec,
            burstiness: summary.burstiness,
            one_over_f_slope: summary.one_over_f_slope,
            ioi_one_over_f_slope: summary.ioi_one_over_f_slope,
        })
    }

    pub fn flush(&mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|err| format!("flush report: {err}"))
    }

    fn write_record<T: Serialize>(&mut self, record: &T) -> Result<(), String> {
        serde_json::to_writer(&mut self.writer, record)
            .map_err(|err| format!("encode report record: {err}"))?;
        self.writer
            .write_all(b"\n")
            .map_err(|err| format!("write report newline: {err}"))?;
        Ok(())
    }
}

pub fn summarize_rhythm_onsets(onsets: &[OnsetSample], group_id: Option<u64>) -> RhythmSummary {
    summarize_rhythm(onsets, &[], group_id)
}

pub fn summarize_rhythm(
    onsets: &[OnsetSample],
    observations: &[RhythmObservation],
    group_id: Option<u64>,
) -> RhythmSummary {
    let filtered: Vec<&OnsetSample> = onsets
        .iter()
        .filter(|onset| match group_id {
            Some(id) => onset.group_id == id,
            None => true,
        })
        .collect();

    let mut times: Vec<f32> = filtered
        .iter()
        .filter_map(|onset| onset.time_sec.is_finite().then_some(onset.time_sec))
        .collect();
    times.sort_by(|a, b| a.total_cmp(b));

    let window_start_sec = times.first().copied().unwrap_or(0.0);
    let window_end_sec = times.last().copied().unwrap_or(window_start_sec);
    let window_duration = (window_end_sec - window_start_sec).max(0.0);
    let onset_count = filtered.len();
    let onset_density_hz = if window_duration > f32::EPSILON {
        onset_count as f32 / window_duration
    } else {
        0.0
    };

    let iois: Vec<f32> = times
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(0.0))
        .collect();
    let (ioi_mean_sec, ioi_std_sec) = mean_and_std(&iois).unwrap_or((0.0, 0.0));
    let has_ioi = !iois.is_empty() && ioi_mean_sec.is_finite() && ioi_mean_sec > f32::EPSILON;
    let ioi_cv = if has_ioi {
        Some(ioi_std_sec / ioi_mean_sec)
    } else {
        None
    };
    let beat_stability = ioi_cv.map(|cv| (1.0 - cv).clamp(0.0, 1.0));
    let burstiness = if has_ioi {
        let denom = ioi_std_sec + ioi_mean_sec;
        if denom > f32::EPSILON {
            Some((ioi_std_sec - ioi_mean_sec) / denom)
        } else {
            None
        }
    } else {
        None
    };

    let plvs: Vec<f32> = filtered
        .iter()
        .filter_map(|onset| onset.plv.filter(|value| value.is_finite()))
        .collect();
    let mean_plv = mean_and_std(&plvs).map(|(mean, _)| mean);
    let observation_summary = (group_id.is_none())
        .then(|| summarize_rhythm_observations(observations))
        .flatten();
    let one_over_f_slope = estimate_one_over_f_slope(&times);
    let ioi_one_over_f_slope = estimate_voice_ioi_one_over_f_slope(&filtered);

    RhythmSummary {
        time_sec: window_end_sec,
        window_start_sec,
        window_end_sec,
        group_id,
        onset_count,
        onset_density_hz,
        ioi_mean_sec: has_ioi.then_some(ioi_mean_sec),
        ioi_cv,
        beat_stability,
        mean_plv,
        kuramoto_order_mean: observation_summary
            .as_ref()
            .and_then(|summary| summary.mean),
        kuramoto_order_max: observation_summary.as_ref().and_then(|summary| summary.max),
        sync_emergence_sec: observation_summary
            .as_ref()
            .and_then(|summary| summary.sync_emergence_sec),
        burstiness,
        one_over_f_slope,
        ioi_one_over_f_slope,
    }
}

#[derive(Debug, Clone, Copy)]
struct RhythmObservationSummary {
    mean: Option<f32>,
    max: Option<f32>,
    sync_emergence_sec: Option<f32>,
}

fn summarize_rhythm_observations(
    observations: &[RhythmObservation],
) -> Option<RhythmObservationSummary> {
    let mut values: Vec<(f32, f32)> = observations
        .iter()
        .filter_map(|observation| {
            let r = observation.kuramoto_order_r?;
            (observation.time_sec.is_finite()
                && r.is_finite()
                && observation.kuramoto_active_count > 0)
                .then_some((observation.time_sec, r.clamp(0.0, 1.0)))
        })
        .collect();
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    if values.is_empty() {
        return None;
    }

    let mean = values.iter().map(|(_, r)| *r).sum::<f32>() / values.len() as f32;
    let max = values
        .iter()
        .map(|(_, r)| *r)
        .fold(0.0f32, |acc, value| acc.max(value));
    let sync_emergence_sec = sync_emergence_sec(&values, 0.70, 2.0);

    Some(RhythmObservationSummary {
        mean: Some(mean),
        max: Some(max),
        sync_emergence_sec,
    })
}

fn sync_emergence_sec(values: &[(f32, f32)], threshold: f32, min_duration_sec: f32) -> Option<f32> {
    let mut run_start = None;
    for &(time_sec, order) in values {
        if order >= threshold {
            let start = *run_start.get_or_insert(time_sec);
            if time_sec - start >= min_duration_sec {
                return Some(start);
            }
        } else {
            run_start = None;
        }
    }
    None
}

fn estimate_one_over_f_slope(times: &[f32]) -> Option<f32> {
    if times.len() < 8 {
        return None;
    }
    let start = *times.first()?;
    let end = *times.last()?;
    let duration = end - start;
    if !duration.is_finite() || duration <= f32::EPSILON {
        return None;
    }

    let bin_count = ((duration / 0.5).round() as usize).clamp(16, 128);
    let mut counts = vec![0.0f32; bin_count];
    for &time in times {
        if !time.is_finite() {
            continue;
        }
        let pos = ((time - start) / duration).clamp(0.0, 1.0);
        let idx = ((pos * bin_count as f32).floor() as usize).min(bin_count - 1);
        counts[idx] += 1.0;
    }
    estimate_series_power_slope(&counts)
}

fn estimate_ioi_one_over_f_slope(iois: &[f32]) -> Option<f32> {
    estimate_series_power_slope(iois)
}

fn estimate_voice_ioi_one_over_f_slope(onsets: &[&OnsetSample]) -> Option<f32> {
    let mut by_voice: BTreeMap<(u64, u64, u32), Vec<f32>> = BTreeMap::new();
    for onset in onsets {
        if onset.time_sec.is_finite() {
            by_voice
                .entry((onset.group_id, onset.voice_id, onset.generation))
                .or_default()
                .push(onset.time_sec);
        }
    }

    let mut slopes = Vec::new();
    for times in by_voice.values_mut() {
        times.sort_by(|a, b| a.total_cmp(b));
        let iois: Vec<f32> = times
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).max(0.0))
            .collect();
        if let Some(slope) = estimate_ioi_one_over_f_slope(&iois) {
            slopes.push(slope);
        }
    }
    mean_and_std(&slopes).map(|(mean, _)| mean)
}

fn estimate_series_power_slope(values: &[f32]) -> Option<f32> {
    let mut values: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if values.len() < 8 {
        return None;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    for value in &mut values {
        *value -= mean;
    }
    let n_values = values.len();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 1..(n_values / 2) {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (n, value) in values.iter().copied().enumerate() {
            let phase = std::f32::consts::TAU * k as f32 * n as f32 / n_values as f32;
            re += value * phase.cos();
            im -= value * phase.sin();
        }
        let power = re * re + im * im;
        if power > 1e-8 && power.is_finite() {
            xs.push((k as f32 / n_values as f32).ln());
            ys.push(power.ln());
        }
    }
    linear_regression_slope(&xs, &ys)
}

fn linear_regression_slope(xs: &[f32], ys: &[f32]) -> Option<f32> {
    if xs.len() != ys.len() || xs.len() < 3 {
        return None;
    }
    let n = xs.len() as f32;
    let mean_x = xs.iter().sum::<f32>() / n;
    let mean_y = ys.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var = 0.0f32;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mean_x;
        cov += dx * (y - mean_y);
        var += dx * dx;
    }
    (var > f32::EPSILON).then_some(cov / var)
}

fn mean_and_std(values: &[f32]) -> Option<(f32, f32)> {
    if values.is_empty() {
        return None;
    }
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        sum += value;
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let mean = sum / n as f32;
    let variance = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| {
            let d = value - mean;
            d * d
        })
        .sum::<f32>()
        / n as f32;
    Some((mean, variance.max(0.0).sqrt()))
}

pub fn summarize_groups(
    voices: &[Voice],
    landscape: &LandscapeFrame,
    time_sec: f32,
) -> Vec<GroupStepSummary> {
    #[derive(Default)]
    struct GroupAccum {
        count: usize,
        sum_freq: f32,
        sum_score: f32,
        sum_level: f32,
        bins: BTreeMap<usize, usize>,
    }

    let mut by_group: BTreeMap<u64, GroupAccum> = BTreeMap::new();
    for agent in voices {
        if !agent.is_alive() {
            continue;
        }
        let freq_hz = agent.body.base_freq_hz();
        let group = by_group.entry(agent.metadata.group_id).or_default();
        group.count += 1;
        group.sum_freq += freq_hz;
        group.sum_score += landscape.evaluate_pitch_score(freq_hz);
        group.sum_level += landscape.evaluate_pitch_level(freq_hz);
        if let Some(bin) = landscape.space.index_of_freq(freq_hz) {
            *group.bins.entry(bin).or_default() += 1;
        }
    }

    by_group
        .into_iter()
        .map(|(group_id, group)| {
            let inv = if group.count > 0 {
                1.0 / group.count as f32
            } else {
                0.0
            };
            let mut entropy = 0.0f32;
            for count in group.bins.values().copied() {
                if count == 0 || group.count == 0 {
                    continue;
                }
                let p = count as f32 / group.count as f32;
                entropy -= p * p.log2();
            }
            GroupStepSummary {
                time_sec,
                group_id,
                alive_count: group.count,
                mean_freq_hz: group.sum_freq * inv,
                mean_c_field_score: group.sum_score * inv,
                mean_c_field_level: group.sum_level * inv,
                freq_entropy_log2: entropy,
            }
        })
        .collect()
}

pub fn scaffold_mode_name(config: ScaffoldConfig) -> &'static str {
    match config {
        ScaffoldConfig::Off => "off",
        ScaffoldConfig::Shared { .. } => "shared",
        ScaffoldConfig::Scrambled { .. } => "scrambled",
    }
}

pub fn scaffold_phase_0_1(config: ScaffoldConfig, time_sec: f32, frame_idx: u64) -> Option<f32> {
    match config {
        ScaffoldConfig::Off => None,
        ScaffoldConfig::Shared { freq_hz } => Some((time_sec * freq_hz.max(0.0)).rem_euclid(1.0)),
        ScaffoldConfig::Scrambled { seed, .. } => {
            let phase = hash01(seed ^ frame_idx.rotate_left(17));
            Some(phase)
        }
    }
}

pub fn onset_samples_from_batches(
    voices: &[Voice],
    batches: &[crate::life::voice::PhonationBatch],
    time_sec: f32,
    scaffold: ScaffoldConfig,
    frame_idx: u64,
) -> Vec<OnsetSample> {
    let mut by_agent = BTreeMap::new();
    for agent in voices {
        by_agent.insert(agent.id(), agent);
    }

    let scaffold_mode = scaffold_mode_name(scaffold);
    let scaffold_phase = scaffold_phase_0_1(scaffold, time_sec, frame_idx);
    let mut out = Vec::new();
    for batch in batches {
        let Some(agent) = by_agent.get(&batch.source_id).copied() else {
            continue;
        };
        let plv = match &agent.articulation.core {
            AnyArticulationCore::Entrain(core) => core.plv(),
            _ => None,
        };
        let freq_hz = agent.body.base_freq_hz();
        for onset in &batch.onsets {
            out.push(OnsetSample {
                time_sec,
                group_id: agent.metadata.group_id,
                voice_id: agent.id(),
                generation: agent.metadata.generation,
                freq_hz,
                strength: onset.strength,
                plv,
                scaffold_mode,
                scaffold_phase_0_1: scaffold_phase,
            });
        }
    }
    out
}

fn hash01(mut x: u64) -> f32 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as f64 / u64::MAX as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onset(time_sec: f32, plv: Option<f32>) -> OnsetSample {
        OnsetSample {
            time_sec,
            group_id: 1,
            voice_id: 1,
            generation: 0,
            freq_hz: 440.0,
            strength: 1.0,
            plv,
            scaffold_mode: "off",
            scaffold_phase_0_1: None,
        }
    }

    fn rhythm_observation(time_sec: f32, order: Option<f32>) -> RhythmObservation {
        RhythmObservation {
            time_sec,
            kuramoto_order_r: order,
            kuramoto_active_count: order.map(|_| 4).unwrap_or(0),
            onsets_in_hop: 0,
            theta_hz: Some(6.0),
            delta_hz: Some(1.0),
            env_open: 1.0,
            env_level: 1.0,
        }
    }

    #[test]
    fn shared_scaffold_phase_wraps_consistently() {
        let phase = scaffold_phase_0_1(ScaffoldConfig::Shared { freq_hz: 2.0 }, -0.25, 0)
            .expect("shared scaffold phase");
        assert!((phase - 0.5).abs() <= 1e-6);
    }

    #[test]
    fn scrambled_scaffold_phase_is_frame_stable() {
        let config = ScaffoldConfig::Scrambled {
            freq_hz: 3.0,
            seed: 17,
        };
        let a = scaffold_phase_0_1(config, 0.0, 42).expect("phase");
        let b = scaffold_phase_0_1(config, 99.0, 42).expect("phase");
        let c = scaffold_phase_0_1(config, 99.0, 43).expect("phase");
        assert!((a - b).abs() <= f32::EPSILON);
        assert!((a - c).abs() > f32::EPSILON);
    }

    #[test]
    fn regular_iois_have_high_beat_stability() {
        let onsets = vec![
            onset(0.0, None),
            onset(0.5, None),
            onset(1.0, None),
            onset(1.5, None),
        ];

        let summary = summarize_rhythm_onsets(&onsets, None);

        assert_eq!(summary.onset_count, 4);
        assert!((summary.ioi_mean_sec.unwrap() - 0.5).abs() <= 1e-6);
        assert!(summary.ioi_cv.unwrap().abs() <= 1e-6);
        assert!((summary.beat_stability.unwrap() - 1.0).abs() <= 1e-6);
        assert!(summary.burstiness.unwrap() < -0.99);
    }

    #[test]
    fn irregular_iois_reduce_stability_and_set_burstiness() {
        let onsets = vec![
            onset(0.0, None),
            onset(0.25, None),
            onset(1.25, None),
            onset(2.25, None),
        ];

        let summary = summarize_rhythm_onsets(&onsets, None);

        let stability = summary.beat_stability.unwrap();
        assert!(stability > 0.0);
        assert!(stability < 0.8);
        assert!(summary.ioi_cv.unwrap() > 0.2);
        assert!(summary.burstiness.unwrap().abs() > 0.1);
    }

    #[test]
    fn empty_and_single_onset_windows_are_defined() {
        let empty = summarize_rhythm_onsets(&[], None);
        assert_eq!(empty.onset_count, 0);
        assert_eq!(empty.onset_density_hz, 0.0);
        assert!(empty.ioi_mean_sec.is_none());
        assert!(empty.beat_stability.is_none());

        let single = summarize_rhythm_onsets(&[onset(2.0, None)], None);
        assert_eq!(single.onset_count, 1);
        assert_eq!(single.window_start_sec, 2.0);
        assert_eq!(single.window_end_sec, 2.0);
        assert!(single.ioi_mean_sec.is_none());
        assert!(single.beat_stability.is_none());
    }

    #[test]
    fn mean_plv_ignores_missing_values() {
        let onsets = vec![
            onset(0.0, Some(0.5)),
            onset(0.5, None),
            onset(1.0, Some(0.9)),
        ];

        let summary = summarize_rhythm_onsets(&onsets, None);

        assert!((summary.mean_plv.unwrap() - 0.7).abs() <= 1e-6);
    }

    #[test]
    fn group_filter_limits_rhythm_summary() {
        let mut onsets = vec![onset(0.0, None), onset(1.0, None)];
        let mut other = onset(10.0, None);
        other.group_id = 2;
        onsets.push(other);

        let summary = summarize_rhythm_onsets(&onsets, Some(1));

        assert_eq!(summary.group_id, Some(1));
        assert_eq!(summary.onset_count, 2);
        assert_eq!(summary.window_end_sec, 1.0);
    }

    #[test]
    fn rhythm_observations_populate_kuramoto_summary() {
        let onsets = vec![onset(0.0, None), onset(1.0, None), onset(2.0, None)];
        let observations = vec![
            rhythm_observation(0.0, Some(0.2)),
            rhythm_observation(1.0, Some(0.8)),
            rhythm_observation(2.0, Some(0.9)),
            rhythm_observation(3.0, Some(0.85)),
        ];

        let summary = summarize_rhythm(&onsets, &observations, None);

        assert!((summary.kuramoto_order_mean.unwrap() - 0.6875).abs() <= 1e-6);
        assert!((summary.kuramoto_order_max.unwrap() - 0.9).abs() <= 1e-6);
        assert_eq!(summary.sync_emergence_sec, Some(1.0));
    }

    #[test]
    fn group_filtered_summary_omits_global_kuramoto_fields() {
        let onsets = vec![onset(0.0, None), onset(1.0, None), onset(2.0, None)];
        let observations = vec![rhythm_observation(0.0, Some(0.9))];

        let summary = summarize_rhythm(&onsets, &observations, Some(1));

        assert!(summary.kuramoto_order_mean.is_none());
        assert!(summary.sync_emergence_sec.is_none());
    }

    #[test]
    fn one_over_f_slope_estimate_is_finite_for_multiscale_onsets() {
        let mut onsets = Vec::new();
        for block in 0..32 {
            let base = block as f32;
            onsets.push(onset(base, None));
            if block % 2 == 0 {
                onsets.push(onset(base + 0.08, None));
            }
            if block % 4 == 0 {
                onsets.push(onset(base + 0.16, None));
            }
            if block % 8 == 0 {
                onsets.push(onset(base + 0.24, None));
            }
        }

        let summary = summarize_rhythm_onsets(&onsets, None);

        assert!(
            summary
                .one_over_f_slope
                .is_some_and(|slope| slope.is_finite())
        );
    }

    #[test]
    fn ioi_one_over_f_slope_estimate_is_finite_for_multiscale_iois() {
        let mut time = 0.0;
        let mut onsets = vec![onset(time, None)];
        for step in 0..48 {
            let ioi = 0.18
                + if step % 2 == 0 { 0.03 } else { 0.0 }
                + if step % 4 == 0 { 0.05 } else { 0.0 }
                + if step % 8 == 0 { 0.08 } else { 0.0 };
            time += ioi;
            onsets.push(onset(time, None));
        }

        let summary = summarize_rhythm_onsets(&onsets, None);

        assert!(
            summary
                .ioi_one_over_f_slope
                .is_some_and(|slope| slope.is_finite())
        );
    }
}
