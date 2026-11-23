//! Display projection shared by native acquisition and the GUI-free Web Worker.
use super::{UiFrame, VoiceStateInfo};
use conchordal_web_ui::frame as wire;

pub(crate) fn diagnostics(frame: &UiFrame) -> Vec<wire::Diagnostic> {
    frame.observation.diagnostics()
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
                support: l.harmonic.support.as_ref().map(|s| {
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
        let mut decoder = wire::Decoder::default();
        let mut encoder = wire::Encoder::default();
        decoder
            .decode(&encoder.encode(source.web_snapshot()).unwrap())
            .unwrap();
        source.landscape.roughness01[0] = 0.125;
        // The second packet omits the frequency axes; the decoder restores them.
        let decoded = decoder
            .decode(&encoder.encode(source.web_snapshot()).unwrap())
            .unwrap();
        assert_eq!(
            decoded.landscape.space.centers_hz,
            source.landscape.space.centers_hz
        );
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
