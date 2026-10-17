use super::*;

pub(super) struct AudioMonitor {
    pub(super) min_occupancy: Option<usize>,
    pub(super) max_peak: f32,
    pub(super) slow_chunks: u32,
    pub(super) max_analysis_wait: Duration,
    pub(super) max_listener_wait: Duration,
    pub(super) last_underrun_frames: u64,
    pub(super) last_stats_log: Instant,
    pub(super) last_clip_log: Instant,
}

impl AudioMonitor {
    pub(super) fn new() -> Self {
        let now = Instant::now();
        Self {
            min_occupancy: None,
            max_peak: 0.0,
            slow_chunks: 0,
            max_analysis_wait: Duration::ZERO,
            max_listener_wait: Duration::ZERO,
            last_underrun_frames: 0,
            last_stats_log: now,
            last_clip_log: now,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        current_time: f32,
        frame_idx: u64,
        hop: usize,
        hop_duration: Duration,
        buffer_capacity: usize,
        buffer_occupancy: usize,
        chunk_peak: f32,
        chunk_elapsed: Duration,
        analysis_wait: Duration,
        listener_wait: Duration,
        underrun_frames: Option<&AtomicU64>,
        conductor_done: bool,
    ) {
        self.min_occupancy = Some(
            self.min_occupancy
                .map_or(buffer_occupancy, |m| m.min(buffer_occupancy)),
        );
        self.max_peak = self.max_peak.max(chunk_peak);
        self.max_analysis_wait = self.max_analysis_wait.max(analysis_wait);
        self.max_listener_wait = self.max_listener_wait.max(listener_wait);

        if self.last_clip_log.elapsed() > Duration::from_millis(200) {
            if chunk_peak > 0.98 {
                warn!(
                    "[t={:.6}] Audio peak high: {:.3} at frame_idx={}. Consider more headroom.",
                    current_time, chunk_peak, frame_idx
                );
                self.last_clip_log = Instant::now();
            } else if chunk_peak > 0.9 {
                warn!(
                    "[t={:.6}] Audio peak nearing clip: {:.3} at frame_idx={}",
                    current_time, chunk_peak, frame_idx
                );
                self.last_clip_log = Instant::now();
            } else if conductor_done && chunk_peak > 1e-4 {
                warn!(
                    "[t={:.6}] Scenario done but audio active: peak={:.4}",
                    current_time, chunk_peak
                );
                self.last_clip_log = Instant::now();
            }
        }

        if chunk_elapsed > hop_duration {
            self.slow_chunks += 1;
        }

        if self.last_stats_log.elapsed() > Duration::from_secs(1) {
            if underrun_frames.is_some() && self.slow_chunks > 0 {
                warn!(
                    "[t={:.6}] Audio hop slow: {} hops over budget {:?} in stats interval",
                    current_time, self.slow_chunks, hop_duration
                );
            }
            let underrun_total = underrun_frames.map_or(0, |count| count.load(Ordering::Relaxed));
            let underrun_frames = underrun_total.saturating_sub(self.last_underrun_frames);
            self.last_underrun_frames = underrun_total;
            if underrun_frames > 0 {
                warn!(
                    "Audio output underrun: {underrun_frames} missing mono frames in stats interval"
                );
            }
            if let Some(min_occ) = self.min_occupancy.take() {
                debug!(
                    "[t={:.6}] Audio stats: min_occ={}, cap={}, hop={}, max_peak={:.3}, slow_chunks={}, max_analysis_wait={:?}, max_listener_wait={:?}, underrun_frames={}",
                    current_time,
                    min_occ,
                    buffer_capacity,
                    hop,
                    self.max_peak,
                    self.slow_chunks,
                    self.max_analysis_wait,
                    self.max_listener_wait,
                    underrun_frames
                );
            }
            self.max_peak = 0.0;
            self.slow_chunks = 0;
            self.max_analysis_wait = Duration::ZERO;
            self.max_listener_wait = Duration::ZERO;
            self.last_stats_log = Instant::now();
        }
    }
}

pub(super) fn log_guard_meter(cfg: &WorkerConfig, last_guard_log: &mut Instant, current_time: f32) {
    if let Some(meter) = cfg.guard_meter.as_ref()
        && last_guard_log.elapsed() >= Duration::from_millis(200)
        && let Some(stats) = meter.take_snapshot()
    {
        warn!(
            "[t={:.6}] Limiter: over={} max_red_db={:.2} in={:.3} out={:.3}",
            current_time,
            stats.num_over,
            stats.max_reduction_db,
            stats.max_abs_in,
            stats.max_abs_out
        );
        *last_guard_log = Instant::now();
    }
}
