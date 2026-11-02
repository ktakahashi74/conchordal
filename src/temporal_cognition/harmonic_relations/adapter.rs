//! One timestamp transaction over a heard support snapshot and executed records.
use super::*;
use crate::core::harmonic_support::{
    Snapshot,
    preview::{Baseline, Observer},
};

pub(crate) use crate::core::harmonic_support::preview::Errors;

pub(crate) struct Adapter {
    bins: Vec<Bin>,
    events: Vec<Event>,
    trajectories: Vec<Trajectory>,
    pub baseline: Option<Baseline>,
}

impl Adapter {
    pub(crate) fn new(coordinates: usize, sources: usize) -> Self {
        Self {
            bins: Vec::with_capacity(coordinates),
            events: Vec::with_capacity(sources + 1),
            trajectories: Vec::with_capacity(sources),
            baseline: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare<'a>(
        &'a mut self,
        snapshot: &Snapshot,
        observer: Observer,
        sample: u64,
        sample_rate: u32,
        hop: bool,
        trajectories: &[Trajectory],
        events: &[Event],
        errors: Errors,
    ) -> Result<Observation<'a>, &'static str> {
        if sample_rate == 0 || snapshot.end_sample > sample {
            return Err("support observation is from the future");
        }
        if snapshot.p.len() > self.bins.capacity()
            || trajectories.len() > self.trajectories.capacity()
            || events
                .iter()
                .enumerate()
                .filter(|(i, e)| !events[..*i].iter().any(|p| p.source == e.source))
                .count()
                > self.events.capacity()
        {
            return Err("adapter declared capacity exceeded");
        }
        if [errors.mass, errors.focus, errors.conflict, errors.shape]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err("invalid producer error bound");
        }
        if trajectories
            .iter()
            .enumerate()
            .any(|(i, t)| trajectories[..i].iter().any(|p| p.source == t.source))
            || events
                .iter()
                .any(|e| e.flags == 0 || e.flags & !(ONSET | OFFSET | OPPORTUNITY | BOUNDARY) != 0)
        {
            return Err("invalid executed trajectory or event");
        }
        self.bins.clear();
        let mut rounding = 0.;
        for (index, &p) in snapshot.p.iter().enumerate() {
            let mass = p as f32;
            rounding += (p.sqrt() - f64::from(mass).sqrt()).powi(2);
            if mass > 0. {
                self.bins.push(Bin { index, mass });
            }
        }
        self.trajectories.clear();
        self.trajectories.extend_from_slice(trajectories);
        self.trajectories.sort_unstable_by_key(|t| t.source);
        self.events.clear();
        for &event in events {
            if let Some(existing) = self.events.iter_mut().find(|e| e.source == event.source) {
                existing.flags |= event.flags;
            } else {
                self.events.push(event);
            }
        }
        self.events.sort_unstable_by_key(|e| e.source);
        let baseline = snapshot.baseline(observer);
        self.baseline = Some(baseline);
        let bounded = |value, error| Value {
            value,
            error,
            valid: true,
            bound: true,
        };
        Ok(Observation {
            time: sample as f64 / f64::from(sample_rate),
            timing_error: 0.,
            hop,
            acoustic: Acoustic {
                snapshot: snapshot.end_sample,
                sampled_at: snapshot.end_sample as f64 / f64::from(sample_rate),
                coordinate_version: baseline.coordinate_version,
                mass: bounded(snapshot.mass, errors.mass),
                focus: bounded(snapshot.focus, errors.focus),
                competition: bounded(snapshot.conflict, errors.conflict),
            },
            support: Support {
                bins: &self.bins,
                residual: 0.,
                error: errors.shape + (0.5 * rounding).sqrt(),
                bound: true,
                valid: snapshot.mass > 0.,
            },
            trajectories: &self.trajectories,
            events: &self.events,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        harmonic_support::Geometry, harmonicity_kernel::HarmonicityKernel, log2space::Log2Space,
    };

    #[test]
    fn adapter_merges_simultaneous_events_keeps_sampling_time_and_bounds() {
        let space = Log2Space::new(80., 2000., 48);
        let geometry = Geometry::new(&space, &HarmonicityKernel::new(&space, Default::default()));
        let mut snapshot = Snapshot::new(geometry);
        let mut density = vec![0.; space.n_bins()];
        density[20] = 1.;
        snapshot.observe_projection(&density, None, 1e-6, 512);
        let mut adapter = Adapter::new(snapshot.p.len(), 2);
        let source = Source {
            kind: 0,
            id: 3,
            generation: 4,
        };
        let events = [ONSET, OPPORTUNITY, OFFSET].map(|flags| Event {
            source: Some(source),
            flags,
        });
        let error = f64::EPSILON;
        let o = adapter
            .prepare(
                &snapshot,
                Observer::Mixed,
                600,
                48000,
                false,
                &[],
                &events,
                Errors {
                    mass: error,
                    focus: error,
                    conflict: error,
                    shape: error,
                },
            )
            .unwrap();
        assert_eq!(o.events.len(), 1);
        assert_eq!(o.events[0].flags, ONSET | OFFSET | OPPORTUNITY);
        assert_eq!(o.acoustic.sampled_at, 512. / 48000.);
        assert_eq!(o.time, 600. / 48000.);
        assert_eq!(o.acoustic.mass.error, error);
        assert!(o.support.error >= error && o.support.bound);
        let coordinates = (0..snapshot.p.len())
            .map(|i| {
                f64::from(snapshot.geometry.root_start_log2) + i as f64 * f64::from(space.step())
            })
            .collect();
        let mut model = Model::with_limits(
            Load {
                sample_rate: 48000,
                hop: 512,
                sources: 2,
                coordinates,
                sparse_entries: snapshot.p.len(),
            },
            Limits {
                seconds: 1.,
                records: 16,
            },
        )
        .unwrap();
        model.observe(o).unwrap();
        assert_eq!(model.resources().records, 1);
        assert!(
            adapter
                .prepare(
                    &snapshot,
                    Observer::Mixed,
                    500,
                    48000,
                    false,
                    &[],
                    &events,
                    Errors::default()
                )
                .is_err()
        );
    }

    #[test]
    #[ignore = "bounded actual-producer integration cost diagnostic; timing lock required"]
    fn hv7_actual_producer_search_cost() {
        use crate::core::stream::analysis::AnalysisStream;
        use std::time::Instant;
        let begin = Instant::now();
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let geometry = Geometry::new(nsgt.space(), &params.harmonicity_kernel);
        let mut stream = AnalysisStream::new(params, nsgt).with_nsgt_power(false);
        stream.enable_support(geometry, false);
        let hop = stream.hop_samples();
        assert_eq!(hop, 512);
        let frequencies = [211f32, 293., 397., 509., 631., 743.];
        let trajectories: Vec<_> = frequencies
            .iter()
            .enumerate()
            .map(|(i, f)| Trajectory {
                source: Source {
                    kind: u8::from(i >= 4),
                    id: i as u64,
                    generation: 0,
                },
                pitch: Value::exact(f64::from(f.log2())),
                amplitude: Value::exact(0.02),
                timbre: Value::default(),
                articulation: 0,
                sounding: true,
            })
            .collect();
        let events = [Event {
            source: Some(trajectories[0].source),
            flags: OPPORTUNITY,
        }];
        let mut audio = vec![0.; hop];
        let mut snapshots = Vec::with_capacity(12);
        for frame in 0..(32 * 12) {
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = (frame * hop + i) as f64 / 48000.;
                *sample = frequencies
                    .iter()
                    .map(|&f| (std::f64::consts::TAU * f64::from(f) * t).sin() * 0.02)
                    .sum::<f64>() as f32;
            }
            let landscape = stream.process(&audio);
            if frame % 32 == 31 {
                snapshots.push(landscape.support.as_ref().unwrap().clone());
            }
        }
        let snapshot = &snapshots[0];
        let coordinates = (0..snapshot.p.len())
            .map(|i| {
                f64::from(snapshot.geometry.root_start_log2)
                    + i as f64 * f64::from(snapshot.geometry.space.step())
            })
            .collect();
        let mut model = Model::new(Load {
            sample_rate: 48000,
            hop,
            sources: 6,
            coordinates,
            sparse_entries: snapshot.p.len(),
        })
        .unwrap();
        let adapters: Vec<_> = snapshots
            .iter()
            .map(|s| Adapter::new(s.p.len(), 6))
            .collect();
        let mut adapters = adapters;
        let observations: Vec<_> = adapters
            .iter_mut()
            .zip(&snapshots)
            .map(|(adapter, snapshot)| {
                adapter
                    .prepare(
                        snapshot,
                        Observer::Mixed,
                        snapshot.end_sample,
                        48000,
                        true,
                        &trajectories,
                        &events,
                        Errors::default(),
                    )
                    .unwrap()
            })
            .collect();
        let startup = begin.elapsed().as_secs_f64();
        for o in &observations[..8] {
            model.observe(*o).unwrap();
        }
        let resources = model.resources();
        eprintln!(
            "HV7 diagnostic startup={startup:.6}s sources=6 roots={} allocated_bytes={} retained_entries={} deadline_ms={:.6}",
            snapshot.p.len(),
            resources.allocated_bytes,
            resources.retained_support_entries,
            1000. * hop as f64 / 48000.
        );
        for n in [1, 2, 4] {
            let start = Instant::now();
            let prediction = model.predict(&observations[8..8 + n], n as u64).unwrap();
            eprintln!(
                "HV7 diagnostic query_anchors={n} endpoints={} atoms={} unknown={} elapsed_ms={:.6}",
                prediction.searched_endpoints,
                prediction.len,
                prediction.unknown,
                start.elapsed().as_secs_f64() * 1000.
            );
        }
    }
}
