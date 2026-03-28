use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use serde::Serialize;

use crate::core::landscape::LandscapeFrame;
use crate::life::individual::sound_body::SoundBody;
use crate::life::individual::{AnyArticulationCore, Individual};
use crate::life::population::{RuntimeEvent, SpawnReason};
use crate::life::scenario::{ScaffoldConfig, SceneMarker};
use crate::life::telemetry::LifeRecord;

#[derive(Debug)]
pub struct JsonlReporter {
    writer: BufWriter<File>,
}

#[derive(Debug, Clone)]
pub struct OnsetSample {
    pub time_sec: f32,
    pub group_id: u64,
    pub agent_id: u64,
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
        agent_id: u64,
        member_idx: usize,
        freq_hz: f32,
    },
    Respawn {
        time_sec: f32,
        group_id: u64,
        agent_id: u64,
        member_idx: usize,
        freq_hz: f32,
        parent_id: Option<u64>,
    },
    Death {
        time_sec: f32,
        group_id: u64,
        agent_id: u64,
        lifetime_sec: f32,
        first_k_mean: f32,
        plv_at_death: Option<f32>,
    },
    Onset {
        time_sec: f32,
        group_id: u64,
        agent_id: u64,
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
}

impl JsonlReporter {
    pub fn create(path: &str) -> Result<Self, String> {
        let file = File::create(path).map_err(|err| format!("create report {path}: {err}"))?;
        Ok(Self {
            writer: BufWriter::new(file),
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
                    agent_id: event.agent_id,
                    member_idx: event.member_idx,
                    freq_hz: event.freq_hz,
                })?,
                SpawnReason::Respawn => self.write_record(&ReportRecord::Respawn {
                    time_sec: event.time_sec,
                    group_id: event.group_id,
                    agent_id: event.agent_id,
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
                agent_id: record.agent_id,
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
                agent_id: onset.agent_id,
                freq_hz: onset.freq_hz,
                strength: onset.strength,
                plv: onset.plv,
                scaffold_mode: onset.scaffold_mode,
                scaffold_phase_0_1: onset.scaffold_phase_0_1,
            })?;
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

pub fn summarize_groups(
    individuals: &[Individual],
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
    for agent in individuals {
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
    individuals: &[Individual],
    batches: &[crate::life::individual::PhonationBatch],
    time_sec: f32,
    scaffold: ScaffoldConfig,
    frame_idx: u64,
) -> Vec<OnsetSample> {
    let mut by_agent = BTreeMap::new();
    for agent in individuals {
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
                agent_id: agent.id(),
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
