use std::fs::{self, File};
use std::io;
use std::path::PathBuf;

const ACTIVE_LOCK: &str = ".conchordal-test-output.lock";
const FAILED: &str = ".conchordal-test-output.failed";
// Keep the latest failure and two preceding cases for debugging; an operational limit.
const RETAINED_FAILURES: usize = 3;

pub struct TestOutputDirectory {
    directory: PathBuf,
    active: Option<File>,
}

impl TestOutputDirectory {
    pub fn new(directory: PathBuf) -> Self {
        fs::create_dir(&directory).unwrap();
        let active = File::create_new(directory.join(ACTIVE_LOCK)).unwrap();
        active.lock().unwrap();
        Self {
            directory,
            active: Some(active),
        }
    }

    fn retain_failure(&mut self) -> io::Result<()> {
        fs::write(self.directory.join(FAILED), b"")?;
        let parent = self.directory.parent().unwrap();
        let retention = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(parent.join(".conchordal-test-output-retention.lock"))?;
        // Serialize pruning and release this completed test while holding the parent lock.
        retention.lock()?;
        drop(self.active.take());
        let mut completed = Vec::new();
        for entry in fs::read_dir(parent)? {
            let entry = entry?;
            if !entry
                .file_name()
                .to_string_lossy()
                .starts_with("conchordal-")
                || !entry.file_type()?.is_dir()
            {
                continue;
            }
            let path = entry.path();
            let Ok(marker) = fs::metadata(path.join(FAILED)) else {
                continue;
            };
            let Ok(active) = File::options()
                .read(true)
                .write(true)
                .open(path.join(ACTIVE_LOCK))
            else {
                continue;
            };
            if active.try_lock().is_err() {
                continue;
            }
            completed.push((marker.modified()?, path, active));
        }
        completed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        for (_, path, active) in completed.into_iter().skip(RETAINED_FAILURES) {
            // The parent lock serializes pruning of these completed directories.
            drop(active);
            fs::remove_dir_all(&path)?;
        }
        Ok(())
    }
}

impl Drop for TestOutputDirectory {
    fn drop(&mut self) {
        let cleanup_error = if std::thread::panicking() {
            None
        } else {
            // Successful output has no failure marker, so pruning cannot select it.
            drop(self.active.take());
            match fs::remove_dir_all(&self.directory) {
                Ok(()) => return,
                Err(error) => Some(error),
            }
        };
        eprintln!("test output retained: {}", self.directory.display());
        if let Err(error) = self.retain_failure() {
            eprintln!("failed to prune test output: {error}");
        }
        if let Some(error) = cleanup_error {
            panic!(
                "failed to remove test output {}: {error}",
                self.directory.display()
            );
        }
    }
}
