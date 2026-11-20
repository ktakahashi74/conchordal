//! Renderer-owned room histories. One source can feed several independent readers.
use super::{RoomBuses, SelfSound, SourcePcm};
use crate::config::RoomMode;
use crate::core::room::{RoomKernel, RoomStream, york_response, york_right_response};
use crate::core::stream::loo::SourceKind;
use crate::core::timebase::Tick;

type Owner = (SourceKind, u64, u32);

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
    ) -> Result<Option<Self>, String> {
        if mode == RoomMode::Off {
            return Ok(None);
        }
        let kernel = RoomKernel::new(&york_response(), hop);
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
        let right_kernel = RoomKernel::new(&york_right_response(), hop);
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
    pub(super) fn process(
        &mut self,
        sounds: &mut [Option<Box<SelfSound>>],
        loo: &mut [SourcePcm],
        metabolism: &mut [SourcePcm],
    ) {
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
            let index = self.own[..self.active]
                .iter()
                .position(|slot| slot.owner == Some(key))
                .unwrap_or_else(|| {
                    let index = self.active;
                    let slot = self
                        .own
                        .get_mut(index)
                        .expect("pre-scanned room source capacity");
                    slot.owner = Some(key);
                    slot.processed = false;
                    slot.stream.reset();
                    self.active += 1;
                    index
                });
            let slot = &mut self.own[index];
            if !slot.processed {
                slot.stream.process(pcm, &mut slot.output);
                slot.processed = true;
            }
            pcm.copy_from_slice(&slot.output);
        }
        self.counts.active = 3 + self.active;
        self.counts.peak = self.counts.peak.max(self.counts.active);
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
            let mut room = RoomBuses::prepare(mode, HOP, 0).unwrap().unwrap();
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
