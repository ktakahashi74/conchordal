use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, atomic::Ordering};

use serde::Serialize;

use crate::audio::output::{AudioCallbackCounters, AudioDeviceInfo};

const PROFILE_HOP_CAPACITY: usize = 100_000;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct AllocationCounts {
    pub(crate) count: u64,
    pub(crate) bytes: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct HopProfile {
    pub(crate) frame_idx: u64,
    pub(crate) time_sec: f32,
    pub(crate) alive_voice_count: usize,
    pub(crate) elapsed_us: f64,
    pub(crate) analysis_wait_us: f64,
    pub(crate) listener_wait_us: f64,
    pub(crate) landscape_update_us: f64,
    pub(crate) population_us: f64,
    pub(crate) reports_us: f64,
    pub(crate) render_route_us: f64,
    pub(crate) synthesis_us: f64,
    pub(crate) rendered_tone_count: usize,
    pub(crate) post_render_us: f64,
    pub(crate) worker_allocations: Option<AllocationCounts>,
    pub(crate) underrun_frames_total: Option<u64>,
}

#[derive(Default, Serialize)]
struct ProfileSummary {
    hop_count: usize,
    elapsed_p99_us: Option<f64>,
    elapsed_max_us: Option<f64>,
    over_budget_hops: usize,
}

#[derive(Serialize)]
struct AudioProfile {
    #[serde(flatten)]
    device: AudioDeviceInfo,
    callback_count: u64,
    callback_frames_total: u64,
    underrun_frames_total: u64,
    callback_errors_total: u64,
}

#[derive(Serialize)]
pub(crate) struct RunProfile {
    #[serde(skip)]
    file: File,
    #[serde(skip)]
    audio_counters: Option<Arc<AudioCallbackCounters>>,
    schema_version: u32,
    scope: &'static str,
    allocation_scope: &'static str,
    seed: u64,
    report_enabled: bool,
    dcc_coupling_strength: f32,
    listener_enabled: bool,
    sample_rate: u32,
    hop_size: usize,
    hop_budget_us: f64,
    allocation_instrumented: bool,
    truncated: bool,
    dropped_hops: u64,
    hop_capacity: usize,
    audio_output: &'static str,
    audio: Option<AudioProfile>,
    summary: ProfileSummary,
    hops: Vec<HopProfile>,
}

impl RunProfile {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create(
        path: &str,
        seed: u64,
        report_enabled: bool,
        dcc_coupling_strength: f32,
        listener_enabled: bool,
        sample_rate: u32,
        hop_size: usize,
        audio: Option<(AudioDeviceInfo, Arc<AudioCallbackCounters>)>,
    ) -> Result<Self, String> {
        let file = File::create(path).map_err(|err| format!("create profile {path}: {err}"))?;
        let mut hops = Vec::new();
        hops.try_reserve_exact(PROFILE_HOP_CAPACITY)
            .map_err(|err| format!("reserve profile hops: {err}"))?;
        let (audio, audio_counters) = match audio {
            Some((device, counters)) => (
                Some(AudioProfile {
                    device,
                    callback_count: 0,
                    callback_frames_total: 0,
                    underrun_frames_total: 0,
                    callback_errors_total: 0,
                }),
                Some(counters),
            ),
            None => (None, None),
        };
        Ok(Self {
            file,
            audio_counters,
            schema_version: 2,
            scope: "worker process_hop entry through return; includes report serialization/write; excludes profile row storage, final profile write, worker pacing sleep, and final report summaries",
            allocation_scope: "worker thread Rust alloc/alloc_zeroed/realloc successful calls and requested bytes during each hop; excludes profile storage, analysis/callback threads, native malloc, and final summaries",
            seed,
            report_enabled,
            dcc_coupling_strength,
            listener_enabled,
            sample_rate,
            hop_size,
            hop_budget_us: hop_size as f64 / sample_rate as f64 * 1_000_000.0,
            allocation_instrumented: cfg!(feature = "profile-alloc"),
            truncated: false,
            dropped_hops: 0,
            hop_capacity: PROFILE_HOP_CAPACITY,
            audio_output: if audio.is_some() {
                "device"
            } else {
                "no_device"
            },
            audio,
            summary: ProfileSummary::default(),
            hops,
        })
    }

    pub(crate) fn record(&mut self, hop: HopProfile) {
        if self.hops.len() < self.hop_capacity {
            self.hops.push(hop);
        } else {
            self.truncated = true;
            self.dropped_hops = self.dropped_hops.saturating_add(1);
        }
    }

    pub(crate) fn write(mut self) -> Result<(), String> {
        if let (Some(audio), Some(counters)) = (&mut self.audio, &self.audio_counters) {
            audio.callback_count = counters.callback_count.load(Ordering::Relaxed);
            audio.callback_frames_total = counters.callback_frames_total.load(Ordering::Relaxed);
            audio.underrun_frames_total = counters.underrun_frames.load(Ordering::Relaxed);
            audio.callback_errors_total = counters.callback_errors_total.load(Ordering::Relaxed);
        }
        let mut elapsed: Vec<f64> = self.hops.iter().map(|hop| hop.elapsed_us).collect();
        elapsed.sort_unstable_by(f64::total_cmp);
        self.summary = ProfileSummary {
            hop_count: self.hops.len(),
            elapsed_p99_us: if elapsed.is_empty() {
                None
            } else {
                let position = (elapsed.len() - 1) as f64 * 0.99;
                let lower = position.floor() as usize;
                let upper = position.ceil() as usize;
                Some(elapsed[lower] + (elapsed[upper] - elapsed[lower]) * position.fract())
            },
            elapsed_max_us: elapsed.last().copied(),
            over_budget_hops: self
                .hops
                .iter()
                .filter(|hop| hop.elapsed_us > self.hop_budget_us)
                .count(),
        };
        let mut writer = BufWriter::new(&self.file);
        serde_json::to_writer(&mut writer, &self).map_err(|err| format!("write profile: {err}"))?;
        writer
            .write_all(b"\n")
            .map_err(|err| format!("write profile: {err}"))?;
        writer
            .flush()
            .map_err(|err| format!("flush profile: {err}"))?;
        if self.truncated {
            return Err(format!(
                "profile truncated: {} hops exceeded capacity {}",
                self.dropped_hops, self.hop_capacity
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "profile-alloc")]
mod allocations {
    use super::AllocationCounts;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static COUNTS: Cell<Option<AllocationCounts>> = const { Cell::new(None) };
    }

    struct ProfileAllocator;

    #[global_allocator]
    static ALLOCATOR: ProfileAllocator = ProfileAllocator;

    fn record(ptr: *mut u8, bytes: usize) {
        if !ptr.is_null() {
            let _ = COUNTS.try_with(|cell| {
                if let Some(mut counts) = cell.get() {
                    counts.count = counts.count.saturating_add(1);
                    counts.bytes = counts.bytes.saturating_add(bytes as u64);
                    cell.set(Some(counts));
                }
            });
        }
    }

    // Accounting uses only const-initialized TLS scalars; it must never allocate.
    unsafe impl GlobalAlloc for ProfileAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            record(ptr, layout.size());
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            record(ptr, layout.size());
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let next = unsafe { System.realloc(ptr, layout, size) };
            record(next, size);
            next
        }
    }

    pub(super) fn begin() {
        COUNTS.with(|cell| cell.set(Some(AllocationCounts::default())));
    }

    pub(super) fn finish() -> Option<AllocationCounts> {
        COUNTS.with(Cell::take)
    }
}

pub(crate) fn begin_allocations() {
    #[cfg(feature = "profile-alloc")]
    allocations::begin();
}

pub(crate) fn finish_allocations() -> Option<AllocationCounts> {
    #[cfg(feature = "profile-alloc")]
    {
        allocations::finish()
    }
    #[cfg(not(feature = "profile-alloc"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_limit_drops_rows_without_growing_storage_and_marks_failure() {
        let path = std::env::temp_dir().join(format!(
            "conchordal-profile-capacity-{}.json",
            std::process::id()
        ));
        let mut profile = RunProfile::create(
            path.to_str().unwrap(),
            1,
            false,
            0.0,
            false,
            48_000,
            512,
            None,
        )
        .unwrap();
        profile.hop_capacity = 1;
        let capacity = profile.hops.capacity();
        for frame_idx in 0..2 {
            profile.record(HopProfile {
                frame_idx,
                time_sec: 0.0,
                alive_voice_count: 1,
                elapsed_us: 100.0,
                analysis_wait_us: 0.0,
                listener_wait_us: 0.0,
                landscape_update_us: 0.0,
                population_us: 0.0,
                reports_us: 0.0,
                render_route_us: 0.0,
                synthesis_us: 0.0,
                rendered_tone_count: 0,
                post_render_us: 0.0,
                worker_allocations: None,
                underrun_frames_total: None,
            });
        }
        assert_eq!(profile.hops.capacity(), capacity);
        assert_eq!(profile.hops.len(), 1);
        assert!(profile.write().unwrap_err().contains("profile truncated"));
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(result["truncated"], true);
        assert_eq!(result["dropped_hops"], 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn final_summary_interpolates_percentile_and_preserves_device_counters() {
        let path = std::env::temp_dir().join(format!(
            "conchordal-profile-summary-{}.json",
            std::process::id()
        ));
        let counters = Arc::new(AudioCallbackCounters::default());
        counters.callback_count.store(3, Ordering::Relaxed);
        counters
            .callback_frames_total
            .store(1024, Ordering::Relaxed);
        counters.underrun_frames.store(16, Ordering::Relaxed);
        counters.callback_errors_total.store(2, Ordering::Relaxed);
        let info = AudioDeviceInfo {
            backend: "test".into(),
            device_name: "unit-test fixture".into(),
            sample_rate: 48_000,
            channels: 2,
            ring_capacity_frames: 4800,
        };
        let mut profile = RunProfile::create(
            path.to_str().unwrap(),
            1,
            false,
            0.0,
            false,
            48_000,
            512,
            Some((info, counters)),
        )
        .unwrap();
        for frame_idx in 0..4 {
            profile.record(HopProfile {
                frame_idx,
                time_sec: 0.0,
                alive_voice_count: 1,
                elapsed_us: frame_idx as f64 + 1.0,
                analysis_wait_us: 0.0,
                listener_wait_us: 0.0,
                landscape_update_us: 0.0,
                population_us: 0.0,
                reports_us: 0.0,
                render_route_us: 0.0,
                synthesis_us: 0.0,
                rendered_tone_count: 0,
                post_render_us: 0.0,
                worker_allocations: None,
                underrun_frames_total: Some(16),
            });
        }
        profile.write().unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!((result["summary"]["elapsed_p99_us"].as_f64().unwrap() - 3.97).abs() < 1e-10);
        assert_eq!(result["audio"]["device_name"], "unit-test fixture");
        assert_eq!(result["audio"]["callback_count"], 3);
        assert_eq!(result["audio"]["callback_frames_total"], 1024);
        assert_eq!(result["audio"]["underrun_frames_total"], 16);
        assert_eq!(result["audio"]["callback_errors_total"], 2);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn allocator_counts_requested_bytes_only_on_the_measured_thread() {
        use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, realloc};
        let child = std::thread::spawn(|| {
            begin_allocations();
            let layout = Layout::from_size_align(16, 8).unwrap();
            unsafe {
                let ptr = alloc(layout);
                assert!(!ptr.is_null());
                std::hint::black_box(ptr);
                let ptr = realloc(ptr, layout, 32);
                assert!(!ptr.is_null());
                std::hint::black_box(ptr);
                dealloc(ptr, Layout::from_size_align(32, 8).unwrap());
                let ptr = alloc_zeroed(layout);
                assert!(!ptr.is_null());
                std::hint::black_box(ptr);
                dealloc(ptr, layout);
            }
            finish_allocations().unwrap()
        });
        let counts = child.join().unwrap();
        assert_eq!(counts.count, 3);
        assert_eq!(counts.bytes, 64);
        assert!(finish_allocations().is_none());
    }
}
