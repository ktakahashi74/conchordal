//! Opt-in acquisition instrumentation; absent from instrument and render builds.
use std::time::{Duration, Instant};

thread_local! {
    static BIRTHS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn record_birth() {
    BIRTHS.with(|count| count.set(count.get() + 1));
}

pub(crate) struct Probe {
    label: &'static str,
    start: Instant,
    paced: bool,
    rows: Vec<serde_json::Value>,
}

impl Probe {
    pub(crate) fn new(label: &'static str) -> Option<Self> {
        let label =
            if label == "analysis" && std::thread::current().name() == Some("listener-analysis") {
                "listener"
            } else {
                label
            };
        std::env::var_os("L1_SAMPLE").map(|_| Self {
            label,
            start: Instant::now(),
            paced: std::env::var_os("L1_PACED").is_some(),
            rows: Vec::with_capacity(100_000),
        })
    }

    pub(crate) fn before(&mut self, frame: u64) -> Instant {
        if self.label == "audio" {
            BIRTHS.with(|count| count.set(0));
        }
        if frame == 0 {
            self.start = Instant::now();
        }
        if self.paced && self.label == "audio" {
            let scheduled = self.start + Duration::from_secs_f64(frame as f64 * 512. / 48_000.);
            std::thread::sleep(scheduled.saturating_duration_since(Instant::now()));
        }
        Instant::now()
    }

    pub(crate) fn after(&mut self, frame: u64, begin: Instant, voices: usize, backlog: usize) {
        let end = Instant::now();
        let deadline = self.start + Duration::from_secs_f64((frame + 1) as f64 * 512. / 48_000.);
        self.rows.push(serde_json::json!({
            "frame": frame, "elapsed_us": (end-begin).as_secs_f64()*1e6,
            "finish_us": (end-self.start).as_secs_f64()*1e6,
            "deadline_miss": self.paced && self.label == "audio" && end > deadline,
            "late_us": end.saturating_duration_since(deadline).as_secs_f64()*1e6,
            "voices": voices, "backlog": backlog,
            "births": BIRTHS.with(|count| count.get()),
        }));
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        let dir = std::env::var("L1_OUTPUT").expect("L1_OUTPUT for acquisition");
        let path = std::path::Path::new(&dir).join(format!("{}.json", self.label));
        serde_json::to_writer(std::fs::File::create(path).unwrap(), &self.rows).unwrap();
    }
}
