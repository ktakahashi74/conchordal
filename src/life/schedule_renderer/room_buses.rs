//! Renderer-owned room histories. One source can feed several independent readers.
use super::{RoomBuses, SelfSound, SourcePcm};
use crate::config::RoomMode;
use crate::core::room::{RoomKernel, RoomStream, york_response, york_right_response};
use crate::core::stream::loo::SourceKind;
use crate::core::timebase::Tick;

type Owner = (SourceKind, u64, u32);

const REFLECTION_GAIN: f32 = 1. / 23.5;
const SHARED_SENSITIVITY: f32 = 12.;

pub(super) struct SiteDry {
    pub presentation: Vec<f32>,
    pub habitat: Vec<f32>,
    direct: f32,
}

/// Each occupied placement contributes one dry stem per destination.
/// The reflection input is independent of source distance.
pub(crate) struct SiteBuses {
    pub(super) dry: Vec<SiteDry>,
    receivers: Vec<ReceiverPlan>,
    pub(crate) full: Vec<crate::core::stream::loo::SitePcm>,
    wet_p: Vec<f32>,
    wet_h: Vec<f32>,
}

struct ReceiverPlan {
    gains: Vec<f32>,
    last_birth: Option<Tick>,
    populations: Vec<PlannedPopulation>,
    last_onset: Option<Tick>,
    last_tonal: Option<Tick>,
    last_voice: Option<Tick>,
    last_flora: Option<Tick>,
    energy: Option<Box<crate::core::temporal_expectation::SiteEnergyHistory>>,
}

struct PlannedPopulation {
    id: u64,
    onset: bool,
    tonal: bool,
}

impl SiteBuses {
    pub(crate) fn prepare(sites: &[crate::scenario::Site], hop: usize) -> Option<Self> {
        if sites.is_empty() {
            return None;
        }
        Some(Self {
            dry: sites
                .iter()
                .map(|s| SiteDry {
                    presentation: vec![0.; hop],
                    habitat: vec![0.; hop],
                    direct: (1. / s.x.hypot(s.y).max(1.)) as f32,
                })
                .collect(),
            receivers: sites
                .iter()
                .map(|p| ReceiverPlan {
                    gains: sites
                        .iter()
                        .map(|s| (1. / (s.x - p.x).hypot(s.y - p.y).max(1.)) as f32)
                        .collect(),
                    last_birth: None,
                    populations: Vec::new(),
                    last_onset: None,
                    last_tonal: None,
                    last_voice: None,
                    last_flora: None,
                    energy: None,
                })
                .collect(),
            full: Vec::new(),
            wet_p: vec![0.; hop],
            wet_h: vec![0.; hop],
        })
    }

    pub(crate) fn plan(
        &mut self,
        scenario: &crate::scenario::Scenario,
        time: crate::core::timebase::Timebase,
    ) {
        use crate::scenario::Action;
        for event in &scenario.events {
            let tick = time.sec_to_tick(event.time);
            for action in &event.actions {
                let plan = match action {
                    Action::Spawn {
                        population_id,
                        spec,
                        ..
                    } => {
                        let plan = &mut self.receivers[spec.control.body.site_id as usize];
                        let onset = spec.control.pitch.respond_to_onsets > 0.
                            || spec.control.phonation.rest_on_habituation > 0.;
                        let tonal = spec.control.pitch.follow_tonal_context > 0.;
                        plan.populations.push(PlannedPopulation {
                            id: *population_id,
                            onset,
                            tonal,
                        });
                        plan.last_voice = Some(plan.last_voice.unwrap_or(0).max(tick));
                        if tonal {
                            plan.last_tonal = Some(plan.last_tonal.unwrap_or(0).max(tick));
                        }
                        if onset {
                            plan.last_onset = Some(plan.last_onset.unwrap_or(0).max(tick));
                        }
                        plan
                    }
                    Action::PlaceFlora { spec, .. } => {
                        let plan = &mut self.receivers[spec.site_id as usize];
                        plan.last_flora = Some(plan.last_flora.unwrap_or(0).max(tick));
                        if spec.respond_to_onsets > 0. || spec.rest_on_habituation > 0. {
                            plan.last_onset = Some(plan.last_onset.unwrap_or(0).max(tick));
                        }
                        plan
                    }
                    Action::PlaceRain { spec, .. } => &mut self.receivers[spec.site_id as usize],
                    _ => continue,
                };
                plan.last_birth = Some(plan.last_birth.unwrap_or(0).max(tick));
            }
        }
        for plan in &mut self.receivers {
            if plan.last_voice.is_some() && time.fs >= 50. {
                plan.energy = Some(Box::new(
                    crate::core::temporal_expectation::SiteEnergyHistory::new(time.fs as u32, 0),
                ));
            }
        }
        self.full = self
            .receivers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.last_birth.is_some())
            .map(|(id, p)| crate::core::stream::loo::SitePcm {
                id: id as u32,
                flora: p.last_flora.is_some(),
                voice: p.last_voice.is_some(),
                onset: p.last_onset.is_some(),
                tonal: p.last_tonal.is_some(),
                samples: vec![0.; time.hop],
            })
            .collect();
    }

    pub(crate) fn retain_readers(
        &mut self,
        now: Tick,
        pop: &crate::life::community::Community,
        tracked: impl Fn(u32) -> bool,
    ) {
        self.full.retain_mut(|f| {
            let plan = &mut self.receivers[f.id as usize];
            f.voice = plan.last_voice.is_some_and(|t| now <= t)
                || plan.populations.iter().any(|p| pop.can_respawn(p.id))
                || pop
                    .voices
                    .iter()
                    .any(|v| v.effective_control.body.site_id == f.id)
                || tracked(f.id);
            f.flora = plan.last_flora.is_some_and(|t| now <= t)
                || pop.flora.iter().any(|v| v.site_id() == f.id);
            if !f.voice {
                plan.energy = None;
            }
            let voices = pop
                .voices
                .iter()
                .filter(|v| v.effective_control.body.site_id == f.id);
            f.tonal = plan.last_tonal.is_some_and(|t| now <= t)
                || voices
                    .clone()
                    .any(|v| v.effective_control.pitch.follow_tonal_context > 0.)
                || plan
                    .populations
                    .iter()
                    .any(|p| p.tonal && pop.can_respawn(p.id));
            f.onset = plan.last_onset.is_some_and(|t| now <= t)
                || voices.clone().any(|v| {
                    v.effective_control.pitch.respond_to_onsets > 0.
                        || v.effective_control.phonation.rest_on_habituation > 0.
                })
                || pop
                    .flora
                    .iter()
                    .any(|f2| f2.site_id() == f.id && f2.needs_onset_history())
                || plan
                    .populations
                    .iter()
                    .any(|p| p.onset && pop.can_respawn(p.id));
            plan.last_birth.is_some_and(|t| now <= t)
                || f.voice
                || f.flora
                || pop.rain.iter().any(|v| v.site_id() == f.id)
        });
    }

    pub(super) fn birth_energy(
        &self,
        site: u32,
    ) -> &crate::core::temporal_expectation::SiteEnergyHistory {
        self.receivers[site as usize]
            .energy
            .as_deref()
            .expect("planned Voice history")
    }

    pub(crate) fn align_energy(&mut self, now: Tick) {
        for receiver in &mut self.receivers {
            if let Some(energy) = &mut receiver.energy {
                energy.align(now);
            }
        }
    }

    pub(super) fn advance_energy(&mut self, now: Tick) {
        for full in &self.full {
            if let Some(history) = &mut self.receivers[full.id as usize].energy {
                history.process(now, &full.samples);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn dry_habitat(&self, id: u32) -> &[f32] {
        &self.dry[id as usize].habitat
    }

    pub(crate) fn local(&self, id: u32) -> &[f32] {
        &self
            .full
            .iter()
            .find(|f| f.id == id)
            .expect("active receiver")
            .samples
    }

    pub(super) fn clear(&mut self) {
        for site in &mut self.dry {
            site.presentation.fill(0.);
            site.habitat.fill(0.);
        }
    }

    pub(super) fn mix(
        &mut self,
        room: Option<&mut RoomBuses>,
        p: &mut [f32],
        h: &mut [f32],
        scratch: &mut [f32],
    ) {
        p.fill(0.);
        h.fill(0.);
        self.wet_p.fill(0.);
        self.wet_h.fill(0.);
        for site in &self.dry {
            for i in 0..p.len() {
                p[i] += site.direct * site.presentation[i];
                h[i] += site.direct * site.habitat[i];
                self.wet_p[i] += site.presentation[i];
                self.wet_h[i] += site.habitat[i];
            }
        }
        for full in &mut self.full {
            full.samples.fill(0.);
            for (site, &gain) in self.dry.iter().zip(&self.receivers[full.id as usize].gains) {
                for (out, dry) in full.samples.iter_mut().zip(&site.habitat) {
                    *out += gain * dry;
                }
            }
        }
        if let Some(room) = room {
            for x in &mut self.wet_p {
                *x *= REFLECTION_GAIN;
            }
            room.presentation.process(&self.wet_p, scratch);
            room.right.process(&self.wet_p, &mut room.right_pcm);
            for (i, pair) in room
                .interleaved
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                pair[1] = SHARED_SENSITIVITY * (p[i] + room.right_pcm[i]);
                p[i] += scratch[i];
                pair[0] = SHARED_SENSITIVITY * p[i];
            }
            if let Some(stream) = &mut room.habitat {
                for x in &mut self.wet_h {
                    *x *= REFLECTION_GAIN;
                }
                stream.process(&self.wet_h, scratch);
                for full in &mut self.full {
                    for (out, &reflection) in full.samples.iter_mut().zip(scratch.iter()) {
                        *out += reflection;
                    }
                }
                for (x, &r) in h.iter_mut().zip(scratch.iter()) {
                    *x += r;
                }
            }
        }
        for x in p.iter_mut().chain(h) {
            *x *= SHARED_SENSITIVITY;
        }
    }
}

struct OwnStream {
    owner: Option<Owner>,
    stream: RoomStream,
    output: Vec<f32>,
    processed: bool,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(crate) struct RoomStreamCounts {
    pub reserved: usize,
    pub active: usize,
    pub peak: usize,
    pub discontinuities: u64,
}

pub(super) struct Histories {
    spatial: bool,
    scratch: Vec<f32>,
    own: Vec<OwnStream>,
    active: usize,
    next: Option<Tick>,
    counts: RoomStreamCounts,
}

impl RoomBuses {
    pub(crate) fn prepare(
        mode: RoomMode,
        hop: usize,
        owners: usize,
        spatial: bool,
    ) -> Result<Option<Self>, String> {
        if mode == RoomMode::Off {
            return Ok(None);
        }
        let plan = |mut response: Vec<f32>| {
            // Without sites the direct tap already has unity gain at 12 m.
            if mode == RoomMode::Presentation && !spatial {
                for tap in &mut response[1..] {
                    *tap *= SHARED_SENSITIVITY * REFLECTION_GAIN;
                }
            }
            RoomKernel::with_direct(&response, hop, if spatial { 0. } else { response[0] })
        };
        let kernel = plan(york_response());
        let both = mode == RoomMode::Both;
        let own_count = if both { owners } else { 0 };
        let buses = 2 + usize::from(both);
        let count = own_count
            .checked_add(buses)
            .ok_or("room stream count overflow")?;
        let size_error = "room pool exceeds addressable storage";
        let slot_metadata = size_of::<OwnStream>() - size_of::<RoomStream>();
        let own_buffers = hop
            .checked_mul(size_of::<f32>())
            .and_then(|pcm| pcm.checked_add(slot_metadata))
            .and_then(|size| size.checked_mul(own_count))
            .ok_or(size_error)?;
        let stereo_buffers = hop.checked_mul(3 * size_of::<f32>()).ok_or(size_error)?;
        let bytes = count
            .checked_mul(kernel.stream_bytes())
            .and_then(|streams| streams.checked_add(own_buffers))
            .and_then(|size| size.checked_add(stereo_buffers))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(size_error)?;
        tracing::debug!(
            count,
            bytes,
            stream_bytes = kernel.stream_bytes(),
            "room reservation"
        );
        let right_kernel = plan(york_right_response());
        // Linux may promise virtual memory and kill the process when pages are
        // touched. Diagnose a known physical shortfall before constructing slots.
        #[cfg(target_os = "linux")]
        if let Some(available) = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find_map(|line| line.strip_prefix("MemAvailable:"))?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
            && bytes as u64 > available
        {
            return Err(format!(
                "room needs {count} streams ({bytes} bytes), only {available} bytes available"
            ));
        }
        let mut pool = kernel
            .stream_pool(count - 1, count - 1)
            .map_err(|e| format!("{e} (room requires {count} streams, {bytes} bytes)"))?;
        let presentation = pool.pop().unwrap();
        let habitat = both.then(|| pool.pop().unwrap());
        let mut own = Vec::new();
        own.try_reserve_exact(own_count)
            .map_err(|_| "room source slot allocation failed")?;
        for stream in pool {
            let mut output = Vec::new();
            output
                .try_reserve_exact(hop)
                .map_err(|_| "room source output allocation failed")?;
            output.resize(hop, 0.);
            own.push(OwnStream {
                owner: None,
                stream,
                output,
                processed: false,
            });
        }
        let right = right_kernel.stream_pool(1, 1)?.pop().unwrap();
        Ok(Some(Self {
            right,
            right_pcm: vec![0.; hop],
            interleaved: vec![0.; 2 * hop],
            presentation,
            habitat,
            histories: Histories {
                spatial,
                scratch: if spatial && both {
                    vec![0.; hop]
                } else {
                    Vec::new()
                },
                own,
                active: 0,
                next: None,
                counts: RoomStreamCounts {
                    reserved: count,
                    active: buses,
                    peak: buses,
                    discontinuities: 0,
                },
            },
        }))
    }

    pub(super) fn process_presentation(&mut self, dry: &[f32], left: &mut [f32]) {
        self.right.process(dry, &mut self.right_pcm);
        self.presentation.process(dry, left);
        for ((pair, &l), &r) in self
            .interleaved
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(left.iter())
            .zip(&self.right_pcm)
        {
            *pair = [l, r];
        }
    }

    pub(crate) fn drained(&self) -> bool {
        self.right.drained()
            && self.presentation.drained()
            && self.habitat.as_ref().is_none_or(RoomStream::drained)
    }

    pub(crate) fn strict(&self) -> bool {
        self.habitat.is_some()
    }

    pub(crate) fn counts(&self) -> RoomStreamCounts {
        self.histories.counts
    }

    pub(crate) fn discontinuous(&self, now: Tick) -> bool {
        self.histories.next.is_some_and(|next| next != now)
    }

    pub(super) fn start_hop(&mut self, now: Tick, hop: usize) {
        if self.discontinuous(now) {
            self.right.reset();
            self.presentation.reset();
            if let Some(bus) = &mut self.habitat {
                bus.reset();
            }
            self.histories.active = 0;
            self.histories.counts.discontinuities += 1;
            tracing::warn!(
                expected = self.histories.next,
                now,
                "room render clock discontinuity"
            );
        }
        self.histories.next = Some(now.saturating_add(hop as Tick));
    }
}

impl Histories {
    pub(super) fn bind_readers(
        &mut self,
        now: Tick,
        sounds: &[Option<Box<SelfSound>>],
        loo: &[SourcePcm],
        metabolism: &[SourcePcm],
    ) -> Result<(), String> {
        let readers = sounds
            .iter()
            .flatten()
            .map(|s| (SourceKind::Voice, s.source_id, s.generation))
            .chain(
                loo.iter()
                    .chain(metabolism.iter())
                    .map(|s| (s.kind, s.id, s.generation)),
            );
        // Release unseen generations first, even when their old Tone still feeds the bus.
        for i in (0..self.active).rev() {
            self.own[i].processed = false;
            if !readers.clone().any(|key| self.own[i].owner == Some(key)) {
                self.active -= 1;
                self.own.swap(i, self.active);
            }
        }
        for key in readers.clone() {
            if self.own[..self.active].iter().any(|s| s.owner == Some(key)) {
                continue;
            }
            let Some(slot) = self.own.get_mut(self.active) else {
                let required = readers.collect::<std::collections::BTreeSet<_>>().len();
                return Err(format!(
                    "strict room exhausted at tick {now}: need {required} owner slots, reserved {}, first missing {key:?}",
                    self.own.len()
                ));
            };
            slot.owner = Some(key);
            slot.processed = false;
            slot.stream.reset();
            self.active += 1;
        }
        self.counts.active = 3 + self.active;
        self.counts.peak = self.counts.peak.max(self.counts.active);
        Ok(())
    }

    pub(super) fn process(
        &mut self,
        now: Tick,
        sounds: &mut [Option<Box<SelfSound>>],
        loo: &mut [SourcePcm],
        metabolism: &mut [SourcePcm],
    ) -> Result<(), String> {
        for (key, pcm) in sounds
            .iter_mut()
            .flatten()
            .map(|s| {
                (
                    (SourceKind::Voice, s.source_id, s.generation),
                    &mut s.habitat,
                )
            })
            .chain(
                loo.iter_mut()
                    .chain(metabolism.iter_mut())
                    .map(|s| ((s.kind, s.id, s.generation), &mut s.habitat)),
            )
        {
            let slot = self.own[..self.active]
                .iter_mut()
                .find(|s| s.owner == Some(key))
                .ok_or_else(|| {
                    format!("strict room reader {key:?} changed after reservation at tick {now}")
                })?;
            if !slot.processed {
                if self.spatial {
                    // Match the shared reflection input scaling before the FIR, including its tail.
                    for (send, &dry) in self.scratch.iter_mut().zip(pcm.iter()) {
                        *send = REFLECTION_GAIN * dry;
                    }
                    slot.stream.process(&self.scratch, &mut slot.output);
                    for (wet, &dry) in slot.output.iter_mut().zip(pcm.iter()) {
                        *wet += dry;
                    }
                } else {
                    slot.stream.process(pcm, &mut slot.output);
                }
                slot.processed = true;
            }
            pcm.copy_from_slice(&slot.output);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_ends_on_the_first_hop_covering_all_wet_buses() {
        const HOP: usize = 512;
        let n = york_response().len();
        let aligned = n.next_multiple_of(HOP) - n;
        for mode in [RoomMode::Presentation, RoomMode::Both] {
            let mut room = RoomBuses::prepare(mode, HOP, 0, false).unwrap().unwrap();
            for last_bus_is_p in [true, false] {
                for s in [aligned - 1, aligned, aligned + 1] {
                    let mut h = [0.; HOP];
                    let mut p = h;
                    // One bus has an earlier impulse, the other determines the deadline.
                    p[if last_bus_is_p { s } else { 0 }] = 1.;
                    h[if last_bus_is_p { 0 } else { s }] = 1.;
                    let end = n + if last_bus_is_p || mode == RoomMode::Both {
                        s
                    } else {
                        0
                    };
                    assert!(room.drained());
                    for now in (0..end).step_by(HOP) {
                        let mut out = [0.; HOP];
                        room.start_hop(now as Tick, HOP);
                        room.right.process(&p, &mut out);
                        room.presentation.process(&p, &mut out);
                        if let Some(bus) = &mut room.habitat {
                            bus.process(&h, &mut out);
                        }
                        assert_eq!(room.drained(), now + HOP >= end);
                        h.fill(0.);
                        p.fill(0.);
                    }
                }
            }
        }
    }
}
