use super::*;

pub(super) struct AudioMonitor {
    pub(super) min_occupancy: Option<usize>,
    pub(super) max_peak: f32,
    pub(super) slow_chunks: u32,
    pub(super) slow_hops_total: u64,
    pub(super) max_analysis_wait: Duration,
    pub(super) max_listener_wait: Duration,
    pub(super) last_underrun_frames: u64,
    pub(super) last_stats_log: Instant,
    pub(super) last_clip_log: Instant,
    hop_count: u64,
    hop_total: Duration,
    hop_max: Duration,
    last_rebuilds: u64,
    #[cfg(target_os = "linux")]
    page_size: Option<u64>,
}

impl AudioMonitor {
    pub(super) fn new() -> Self {
        let now = Instant::now();
        Self {
            min_occupancy: None,
            max_peak: 0.0,
            slow_chunks: 0,
            slow_hops_total: 0,
            max_analysis_wait: Duration::ZERO,
            max_listener_wait: Duration::ZERO,
            last_underrun_frames: 0,
            last_stats_log: now,
            last_clip_log: now,
            hop_count: 0,
            hop_total: Duration::ZERO,
            hop_max: Duration::ZERO,
            last_rebuilds: 0,
            #[cfg(target_os = "linux")]
            page_size: std::fs::read("/proc/self/auxv").ok().and_then(|bytes| {
                let word = std::mem::size_of::<usize>();
                bytes.chunks_exact(2 * word).find_map(|entry| {
                    let key = usize::from_ne_bytes(entry[..word].try_into().ok()?);
                    let value = usize::from_ne_bytes(entry[word..].try_into().ok()?);
                    (key == 6).then_some(value as u64) // AT_PAGESZ
                })
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        current_time: f64,
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
        finish_begun: bool,
        telemetry: bool,
        pop: &mut Community,
        audio_counters: Option<&crate::audio::output::AudioCallbackCounters>,
    ) {
        self.min_occupancy = Some(
            self.min_occupancy
                .map_or(buffer_occupancy, |m| m.min(buffer_occupancy)),
        );
        self.max_peak = self.max_peak.max(chunk_peak);
        self.max_analysis_wait = self.max_analysis_wait.max(analysis_wait);
        self.max_listener_wait = self.max_listener_wait.max(listener_wait);
        self.hop_count += 1;
        self.hop_total += chunk_elapsed;
        self.hop_max = self.hop_max.max(chunk_elapsed);

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
            } else if finish_begun && chunk_peak > 1e-4 {
                warn!(
                    "[t={:.6}] Scenario done but audio active: peak={:.4}",
                    current_time, chunk_peak
                );
                self.last_clip_log = Instant::now();
            }
        }

        if chunk_elapsed > hop_duration {
            self.slow_chunks += 1;
            self.slow_hops_total += 1;
        }

        if self.last_stats_log.elapsed() >= crate::audio::output::STATS_INTERVAL {
            if underrun_frames.is_some() && self.slow_chunks > 0 {
                warn!(
                    "[t={:.6}] Audio hop slow: {} hops over budget {:?} in stats interval",
                    current_time, self.slow_chunks, hop_duration
                );
            }
            let underrun_total = underrun_frames.map_or(0, |count| count.load(Ordering::Relaxed));
            let underrun_frames = underrun_total.saturating_sub(self.last_underrun_frames);
            self.last_underrun_frames = underrun_total;
            if telemetry {
                #[cfg(target_os = "linux")]
                let rss_bytes = self.page_size.and_then(|page_size| {
                    std::fs::read_to_string("/proc/self/statm")
                        .ok()?
                        .split_whitespace()
                        .nth(1)?
                        .parse::<u64>()
                        .ok()?
                        .checked_mul(page_size)
                });
                #[cfg(not(target_os = "linux"))]
                let rss_bytes = None;
                let rebuilds_total =
                    audio_counters.map_or(0, |c| c.rebuilds_total.load(Ordering::Relaxed));
                let rebuilds = rebuilds_total.saturating_sub(self.last_rebuilds);
                self.last_rebuilds = rebuilds_total;
                let (births, deaths) = pop.take_lifecycle_counts();
                println!(
                    "{}",
                    Telemetry {
                        rss_bytes,
                        hop_mean_us: self.hop_total.as_secs_f64() * 1e6 / self.hop_count as f64,
                        hop_max_us: self.hop_max.as_secs_f64() * 1e6,
                        slow_hops: self.slow_chunks,
                        underrun_frames,
                        rebuilds,
                        voices: pop.voices.iter().filter(|v| v.is_alive()).count(),
                        populations: pop.population_count(),
                        flora: pop.flora.iter().filter(|f| !f.is_finished()).count(),
                        births,
                        deaths,
                    }
                );
            }
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
            self.hop_count = 0;
            self.hop_total = Duration::ZERO;
            self.hop_max = Duration::ZERO;
            self.last_stats_log = Instant::now();
        }
    }
}

struct Telemetry {
    rss_bytes: Option<u64>,
    hop_mean_us: f64,
    hop_max_us: f64,
    slow_hops: u32,
    underrun_frames: u64,
    rebuilds: u64,
    voices: usize,
    populations: usize,
    flora: usize,
    births: u64,
    deaths: u64,
}

impl std::fmt::Display for Telemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "telemetry")?;
        if let Some(rss) = self.rss_bytes {
            write!(f, " rss_bytes={rss}")?;
        }
        write!(
            f,
            " hop_mean_us={:.3} hop_max_us={:.3} slow_hops={} underrun_frames={} rebuilds={} voices={} populations={} flora={} births={} deaths={}",
            self.hop_mean_us,
            self.hop_max_us,
            self.slow_hops,
            self.underrun_frames,
            self.rebuilds,
            self.voices,
            self.populations,
            self.flora,
            self.births,
            self.deaths,
        )
    }
}

pub(super) fn log_guard_meter(cfg: &WorkerConfig, last_guard_log: &mut Instant, current_time: f64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_is_one_line_with_optional_rss_and_interval_counts() {
        let mut summary = Telemetry {
            rss_bytes: Some(4096),
            hop_mean_us: 12.5,
            hop_max_us: 25.0,
            slow_hops: 2,
            underrun_frames: 128,
            rebuilds: 1,
            voices: 8,
            populations: 2,
            flora: 3,
            births: 4,
            deaths: 5,
        };
        assert_eq!(
            summary.to_string(),
            "telemetry rss_bytes=4096 hop_mean_us=12.500 hop_max_us=25.000 slow_hops=2 underrun_frames=128 rebuilds=1 voices=8 populations=2 flora=3 births=4 deaths=5"
        );
        summary.rss_bytes = None;
        assert!(!summary.to_string().contains("rss"));
        assert_eq!(summary.to_string().lines().count(), 1);
    }

    #[test]
    fn stats_interval_resets_hop_and_lifecycle_accumulators() {
        let mut monitor = AudioMonitor::new();
        let mut pop = Community::new(crate::core::timebase::Timebase {
            fs: 48_000.0,
            hop: 512,
        });
        let counters = crate::audio::output::AudioCallbackCounters::default();
        counters.rebuilds_total.store(2, Ordering::Relaxed);
        monitor.last_stats_log = Instant::now() - crate::audio::output::STATS_INTERVAL;
        monitor.update(
            0.0,
            0,
            512,
            Duration::from_millis(10),
            512,
            0,
            0.0,
            Duration::from_millis(20),
            Duration::ZERO,
            Duration::ZERO,
            None,
            false,
            true,
            &mut pop,
            Some(&counters),
        );
        assert_eq!(monitor.hop_count, 0);
        assert_eq!(monitor.hop_total, Duration::ZERO);
        assert_eq!(monitor.hop_max, Duration::ZERO);
        assert_eq!(monitor.last_rebuilds, 2);
        assert_eq!(pop.take_lifecycle_counts(), (0, 0));
    }
}
