use std::sync::Arc;

use conchordal_web_ui::frame::Diagnostic;

use crate::temporal_cognition::{body, observation};

#[derive(Clone, Debug, Default)]
pub(crate) struct TemporalUi {
    body: Option<Arc<body::Snapshot>>,
    pub(crate) temporal: Arc<[observation::Snapshot; 2]>,
}

impl TemporalUi {
    pub(crate) fn from_snapshots(
        body: Option<&body::Snapshot>,
        temporal: &[observation::Snapshot; 2],
    ) -> Self {
        Self {
            body: body.map(|body| Arc::new(*body)),
            temporal: Arc::new(*temporal),
        }
    }
    pub(crate) fn diagnostics(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        // The GUI draws the "Listener Twin" heading with its state bars even before the
        // first frame, so these labels come first and sit directly under it.
        if self
            .temporal
            .iter()
            .any(|s| s.state != crate::temporal_cognition::observation::ObservationState::Off)
        {
            out.push(Diagnostic::Label(
                ("Temporal context: observing; relations unavailable").into(),
            ));
            for observation in self.temporal.iter() {
                let bus = if observation.bus == 0 {
                    "Habitat"
                } else {
                    "Presentation"
                };
                out.push(Diagnostic::Label(format!(
                    "{bus}: {:?} · {} frames · {} dropped",
                    observation.state,
                    observation.received_frames,
                    observation.delivery_dropped_frames
                )));
                if let Some(trajectories) = observation.trajectories {
                    out.push(Diagnostic::Label(format!(
                        "{} spectral peaks · residual energy {:.3e}",
                        trajectories.peak_bins.iter().flatten().count(),
                        trajectories.energy[7]
                    )));
                }
                if observation.ridge_failed {
                    out.push(Diagnostic::Label(
                        ("Ridge diagnostics: failed for this source epoch").into(),
                    ));
                } else if let Some(ridges) = observation.ridges {
                    if ridges.observed {
                        out.push(Diagnostic::Label(format!(
                            "{} acoustic ridges · research scales, uncalibrated",
                            ridges.current.iter().flatten().count()
                        )));
                    } else {
                        out.push(Diagnostic::Label(
                            ("Ridge diagnostics: awaiting supported audio").into(),
                        ));
                    }
                }
                if let Some(context) = observation.group_prototypes {
                    out.push(Diagnostic::Label(format!(
                        "Prototype/group matches: {} · {} descriptors · support end {}",
                        context.assignments.iter().flatten().count(),
                        context.descriptors.iter().flatten().count(),
                        context.end_sample
                    )));
                }

                if observation.acoustic_failed {
                    out.push(Diagnostic::Label(
                        ("Acoustic diagnostics: failed for this source epoch").into(),
                    ));
                } else if let Some(acoustic) = observation.acoustic {
                    out.push(Diagnostic::Label(format!(
                        "{} sound groups · {} raw descriptors · {} accents · uncalibrated",
                        acoustic.retained_groups.iter().flatten().count(),
                        acoustic.features.iter().flatten().count(),
                        acoustic
                            .features
                            .iter()
                            .flatten()
                            .filter(|f| { f.detector.is_some_and(|d| d.accent.is_some()) })
                            .count(),
                    )));
                }
                if let Some(error) = observation.context_error {
                    out.push(Diagnostic::Label(format!("Observed context: {error}")));
                } else if let Some(context) = observation.context {
                    out.push(Diagnostic::Label(format!(
                        "Observed context · coverage {:.0}% · {} groups{}",
                        100. * context.observed_coverage,
                        context.groups.iter().flatten().count(),
                        if context.censored { " · censored" } else { "" }
                    )));
                    for g in context.groups.iter().flatten() {
                        out.push(Diagnostic::Label(format!(
                            "Group {} · {} hops · {} accents · weight {:.3} · arrival {:?}",
                            g.group.generation,
                            g.history_hops,
                            g.accents,
                            g.acoustic_weight,
                            g.arrival_probability
                        )));
                    }
                }
                if let Some(error) = observation.period_error {
                    out.push(Diagnostic::Label(format!(
                        "Period / arrival diagnostics: {error}"
                    )));
                } else if let Some(period) = observation.period {
                    for group in period.groups.iter().flatten().filter(|g| g.active) {
                        out.push(Diagnostic::Label(format!(
                            "Group {}: {} accents · {} period alternatives · uncalibrated",
                            group.ledger.group.generation,
                            group.ledger.retained_accents,
                            group.peaks.iter().flatten().count()
                        )));
                        if let Some(f) = group.forecast {
                            out.push(Diagnostic::Label(format!("{:?} arrival {:?} · elapsed {:?} s · reset unknown {} · forecast {}–{} · evidence {}–{} / available {}",f.model,f.probability,f.elapsed_seconds,f.reset_unknown,f.issued_at,f.horizon_end,f.source_start,f.source_end,f.available)));
                        }
                    }
                }
            }
        }
        if let Some(body) = &self.body {
            out.push(Diagnostic::Separator);
            out.push(Diagnostic::Heading(
                ("Private body descriptors (uncalibrated)").into(),
            ));
            if let Some(version) = body.prototype_model_version {
                out.push(Diagnostic::Label(format!(
                    "Prototype model {:02x}{:02x}{:02x}{:02x} (diagnostic)",
                    version[0], version[1], version[2], version[3]
                )));
            }
            out.push(Diagnostic::Label(format!(
            "{} frames · {} capture drops · {} outside-capacity voice hops · {} invalid bus hops",
            body.processed_frames, body.capture_drops, body.outside_voice_hops, body.invalid_hops
        )));
            for (index, record) in body
                .records
                .iter()
                .enumerate()
                .filter(|(_, r)| r.active)
                .take(4)
            {
                out.push(Diagnostic::Label(format!(
                    "Voice {} / {} · body {} · bus {} · support {}..{} · mask {:06b}",
                    record.source_id,
                    record.source_generation,
                    record.body_generation,
                    record.bus,
                    record.start,
                    record.end,
                    record.mask
                )));
                if body.prototype_model_version.is_some() {
                    if let Some(assignment) = body.prototype_assignments[index] {
                        out.push(Diagnostic::Label(format!(
                            "Descriptive medoid {} · distance {:.3} · {} coordinates",
                            assignment.key.0, assignment.distance, assignment.common_coordinates
                        )));
                    } else {
                        out.push(Diagnostic::Label(
                            ("Descriptive medoid: unsupported").into(),
                        ));
                    }
                }
            }
        }
        out
    }
}
