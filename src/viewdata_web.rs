//! Display projection shared by native acquisition and the GUI-free Web Worker.
use super::{UiFrame, VoiceStateInfo};
use conchordal_web_ui::frame::{self as wire, Diagnostic};

pub(crate) fn diagnostics(frame: &UiFrame) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if let Some(body) = &frame.body {
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
        out.push(Diagnostic::Separator);
    }
    out.push(Diagnostic::Heading(("Listener Twin").into()));
    if frame
        .temporal
        .iter()
        .any(|s| s.state != crate::temporal_cognition::observation::ObservationState::Off)
    {
        out.push(Diagnostic::Label(
            ("Temporal context: observing; relations unavailable").into(),
        ));
        for observation in frame.temporal.iter() {
            let bus = if observation.bus == 0 {
                "Habitat"
            } else {
                "Presentation"
            };
            out.push(Diagnostic::Label(format!(
                "{bus}: {:?} · {} frames · {} dropped",
                observation.state, observation.received_frames, observation.delivery_dropped_frames
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
    out
}
pub(crate) fn support_scans(voice: &VoiceStateInfo, bins: usize) -> Option<[Vec<f64>; 3]> {
    let reference = voice.support.as_ref()?;
    let mut scans = std::array::from_fn(|_| vec![0.; bins]);
    if let Some(template) = &voice.support_template {
        for i in 0..bins {
            for (scan, value) in scans
                .iter_mut()
                .zip(reference.template_features(i, template))
            {
                scan[i] = value;
            }
        }
    } else {
        reference.single_partial_scan(voice.support_single_density, &mut scans);
    }
    Some(scans)
}

impl crate::viewdata::UiFrame {
    pub fn web_snapshot(&self) -> wire::UiFrame {
        let space = |s: &crate::core::log2space::Log2Space| wire::Space {
            centers_hz: s.centers_hz.clone(),
            centers_log2: s.centers_log2.clone(),
            step_log2: s.step(),
        };
        let l = &self.landscape;
        let k = l.consonance_kernel;
        wire::UiFrame {
            wave: self.wave.clone(),
            spec: self.spec.clone(),
            listener: self.listener,
            listener_rhythm: self.listener_rhythm.clone(),
            prediction: self.prediction.clone(),
            diagnostics: diagnostics(self),
            landscape: wire::Landscape {
                space: space(&l.space),
                support: l.support.as_ref().map(|s| {
                    std::sync::Arc::new(wire::Snapshot {
                        geometry: wire::Geometry {
                            root_start_log2: s.geometry.root_start_log2,
                            space: space(&s.geometry.space),
                        },
                        p: s.p.clone(),
                        mass: s.mass,
                        focus: s.focus,
                        conflict: s.conflict,
                        end_sample: s.end_sample,
                    })
                }),
                subjective_intensity: l.subjective_intensity.clone(),
                nsgt_power: l.nsgt_power.clone(),
                consonance_field_score: l.consonance_field_score.clone(),
                consonance_field_level: l.consonance_field_level.clone(),
                harmonicity01: l.harmonicity01.clone(),
                roughness01: l.roughness01.clone(),
                consonance_kernel: wire::ConsonanceKernel {
                    a: k.a,
                    b: k.b,
                    c: k.c,
                    d: k.d,
                },
            },
            voices: self
                .voices
                .iter()
                .map(|v| wire::VoiceStateInfo {
                    id: v.id,
                    freq_hz: v.freq_hz,
                    target_freq: v.target_freq,
                    integration_window: v.integration_window,
                    consonance: v.consonance,
                    support_template: v.support_template.clone(),
                    support_scans: support_scans(v, l.space.n_bins()),
                })
                .collect(),
            meta: wire::SimulationMeta {
                script: self.meta.script.as_ref().map(|s| wire::ScriptStatus {
                    evaluated_through: s.evaluated_through,
                    done: s.done,
                    stopped: s.stopped,
                    error: s.error.clone(),
                    error_position: wire::PositionLabel(format!("{:?}", s.error_position)),
                    warnings: s
                        .warnings
                        .iter()
                        .map(|w| wire::ScriptWarning {
                            position: wire::PositionLabel(format!("{:?}", w.position)),
                            message: w.message.clone(),
                            count: w.count,
                        })
                        .collect(),
                }),
                time_sec: self.meta.time_sec,
                end: self.meta.end,
                population_count: self.meta.population_count,
                slow_hops: self.meta.slow_hops,
                voice_count: self.meta.voice_count,
                event_queue_len: self.meta.event_queue_len,
                estimator_freezes: self.meta.estimator_freezes,
                estimator_last_frozen_issue: self.meta.estimator_last_frozen_issue,
                audio_xruns: self.meta.audio_xruns,
                audio_rebuilds: self.meta.audio_rebuilds,
                audio_recovering: self.meta.audio_recovering,
                peak_level: self.meta.peak_level,
                scenario_name: self.meta.scenario_name.clone(),
                scene_name: self.meta.scene_name.clone(),
                playback_state: self.meta.playback_state,
                channel_peak: self.meta.channel_peak,
                window_peak: self.meta.window_peak,
                kuramoto_order_r: self.meta.kuramoto_order_r,
                kuramoto_active_count: self.meta.kuramoto_active_count,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_snapshot_preserves_identity_unknowns_scans_and_hop_batch() {
        let mut source = crate::viewdata::UiFrame::default();
        source.listener.generated_frame_id = (1_u64 << 53) + 19;
        source.prediction.pred_tau_tick = Some(u64::MAX - 3);
        source.meta.scene_name = Some("雨 / canopy".into());
        source.listener_rhythm.push(wire::ListenerRhythmHop {
            time_sec: 1.25,
            ..Default::default()
        });
        source.landscape.roughness01[0] = 0.125;
        let snapshot = source.web_snapshot();
        let decoded = wire::UiFrame::decode(&snapshot.encode().unwrap()).unwrap();
        assert_eq!(
            decoded.listener.generated_frame_id,
            source.listener.generated_frame_id
        );
        assert_eq!(
            decoded.prediction.pred_tau_tick,
            source.prediction.pred_tau_tick
        );
        assert_eq!(decoded.meta.scene_name, source.meta.scene_name);
        assert_eq!(decoded.listener.prediction.forecast, None);
        assert_eq!(decoded.listener_rhythm.len(), 1);
        assert_eq!(decoded.listener_rhythm[0].time_sec, 1.25);
        assert_eq!(decoded.landscape.roughness01, source.landscape.roughness01);
    }
}
