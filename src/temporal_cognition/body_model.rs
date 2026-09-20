//! Frozen development prototypes for body/group matching, not candidate eligibility.

use super::body::Record;
use super::{observables, phrase, proposals::frontend};
use crate::config::{AppConfig, TemporalBodyConfig, TemporalBodyPrototypesConfig};

pub(crate) fn validate(config: &AppConfig) -> anyhow::Result<()> {
    let Some(model) = &config.temporal_body_prototypes else {
        return Ok(());
    };
    let body = config
        .temporal_body
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("temporal_body_prototypes requires temporal_body"))?;
    anyhow::ensure!(
        model.model_version.len() == 64
            && model
                .model_version
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "body prototype model_version must be a lowercase SHA-256"
    );
    anyhow::ensure!(
        model.sample_rate == config.audio.sample_rate
            && model.nfft == config.analysis.nfft
            && model.hop_size == config.analysis.hop_size,
        "body prototype model requires its registered sample rate, FFT and hop"
    );
    anyhow::ensure!(
        model.means == body.means
            && model.deviations == body.deviations
            && model.accent_means == body.accent_means
            && model.accent_deviations == body.accent_deviations,
        "body prototypes and extractor must share frozen descriptor and accent scales"
    );
    anyhow::ensure!(
        (1..=8).contains(&model.medoids.len()),
        "body prototype model requires 1..8 medoids"
    );
    anyhow::ensure!(
        model
            .medoids
            .windows(2)
            .all(|p| p[0].record_id < p[1].record_id),
        "body medoids must have unique record IDs in lexical order"
    );
    for medoid in &model.medoids {
        anyhow::ensure!(
            !medoid.record_id.is_empty()
                && medoid.record_id.len() <= 128
                && medoid.record_id.is_ascii(),
            "body medoid record ID must be 1..128 ASCII bytes"
        );
        anyhow::ensure!(
            (1..=63).contains(&medoid.mask) && medoid.raw_values.iter().all(|v| v.is_finite()),
            "body medoid requires finite values and a nonempty six-coordinate mask"
        );
        for i in 0..6 {
            anyhow::ensure!(
                medoid.mask & (1 << i) == 0
                    || ((medoid.raw_values[i] - body.means[i]) / body.deviations[i].max(1e-6))
                        .is_finite(),
                "body medoid standardization overflow"
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct Prototypes {
    pub version: [u8; 32],
    descriptors: [Descriptor; 8],
    count: usize,
}

impl Prototypes {
    pub(crate) fn new(model: &TemporalBodyPrototypesConfig, scales: TemporalBodyConfig) -> Self {
        let version = std::array::from_fn(|i| {
            u8::from_str_radix(&model.model_version[i * 2..i * 2 + 2], 16)
                .expect("validated model version")
        });
        let descriptors = std::array::from_fn(|index| {
            if let Some(medoid) = model.medoids.get(index) {
                Descriptor {
                    key: (index as u64, 0),
                    values: observables::standardize(
                        std::array::from_fn(|i| {
                            (medoid.mask & (1 << i) != 0).then_some(medoid.raw_values[i])
                        }),
                        scales,
                    ),
                }
            } else {
                Descriptor {
                    key: (index as u64, 0),
                    values: [None; 6],
                }
            }
        });
        Self {
            version,
            descriptors,
            count: model.medoids.len(),
        }
    }

    pub(crate) fn assign(&self, record: &Record, scales: TemporalBodyConfig) -> Option<Assignment> {
        if !record.active {
            return None;
        }
        nearest(
            &record.standardized(scales),
            &self.descriptors[..self.count],
            0.25,
        )
        .expect("validated body descriptors")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Descriptor {
    pub key: (u64, u64),
    pub values: [Option<f64>; 6],
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Shared {
    pub model_version: [u8; 32],
    pub bus: u8,
    pub epoch: u64,
    pub end_sample: u64,
    pub descriptors: [Option<observables::WindowDescriptor>; 7],
    pub assignments: [Option<Assignment>; 8],
}

impl Shared {
    pub(crate) fn refresh(
        cached: &mut Option<Self>,
        model: &Prototypes,
        scales: TemporalBodyConfig,
        phrase: &phrase::Phrase,
        acoustic: &frontend::Snapshot,
        (bus, epoch): (u8, u64),
    ) {
        let current = phrase.snapshot();
        let end = current.end_sample;
        let eligible = |descriptor: &observables::WindowDescriptor| {
            descriptor.group.bus == bus
                && descriptor.group.epoch == epoch
                && descriptor.end <= end
                && descriptor.available <= end
                && acoustic.retained_groups.contains(&Some(descriptor.group))
                && acoustic.group_handles[..7]
                    .iter()
                    .enumerate()
                    .any(|(i, h)| *h == Some(descriptor.group) && acoustic.eligible[i])
        };
        if let Some(cache) = cached.as_mut() {
            if cache.epoch != epoch
                || cache.bus != bus
                || cache.model_version != model.version
                || cache.end_sample > end
            {
                *cached = None;
            } else {
                for descriptor in &mut cache.descriptors {
                    if descriptor.as_ref().is_some_and(|d| !eligible(d)) {
                        *descriptor = None;
                    }
                }
                for assignment in &mut cache.assignments {
                    if assignment.is_some_and(|a| {
                        !cache
                            .descriptors
                            .iter()
                            .flatten()
                            .any(|d| a.key == (d.group.epoch, d.group.generation))
                    }) {
                        *assignment = None;
                    }
                }
            }
        }
        if cached.as_ref().is_some_and(|cache| {
            end - cache.end_sample < u64::from(current.sample_rate).div_ceil(10)
        }) {
            return;
        }
        let descriptors = phrase.body_descriptors().map(|d| d.filter(&eligible));
        let candidates = descriptors.map(|d| {
            d.map_or(
                Descriptor {
                    key: (u64::MAX, u64::MAX),
                    values: [None; 6],
                },
                |d| Descriptor {
                    key: (d.group.epoch, d.group.generation),
                    values: observables::standardize(d.raw_values, scales),
                },
            )
        });
        let assignments = std::array::from_fn(|i| {
            (i < model.count)
                .then(|| {
                    nearest(&model.descriptors[i].values, &candidates, 0.25)
                        .expect("finite standardized group descriptors")
                })
                .flatten()
        });
        *cached = Some(Self {
            model_version: model.version,
            bus,
            epoch,
            end_sample: end,
            descriptors,
            assignments,
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Assignment {
    pub key: (u64, u64),
    pub distance: f64,
    pub common_coordinates: usize,
}

pub(crate) fn nearest(
    values: &[Option<f64>; 6],
    candidates: &[Descriptor],
    maximum_distance: f64,
) -> Result<Option<Assignment>, &'static str> {
    if candidates.len() > 8
        || !maximum_distance.is_finite()
        || maximum_distance < 0.
        || values.iter().flatten().any(|v| !v.is_finite())
        || candidates
            .iter()
            .flat_map(|row| row.values.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err("finite standardized descriptors and nonnegative distance gate required");
    }
    let mut best: Option<Assignment> = None;
    for row in candidates {
        let mut norm: f64 = 0.;
        let mut common_coordinates = 0;
        for (left, right) in values.iter().zip(row.values) {
            if let (Some(left), Some(right)) = (left, right) {
                norm = norm.hypot(left - right);
                common_coordinates += 1;
            }
        }
        if common_coordinates == 0 {
            continue;
        }
        let distance = norm / (common_coordinates as f64).sqrt();
        if distance > maximum_distance {
            continue;
        }
        let assignment = Assignment {
            key: row.key,
            distance,
            common_coordinates,
        };
        if best.is_none_or(|old| {
            distance < old.distance
                || (distance == old.distance
                    && (common_coordinates > old.common_coordinates
                        || (common_coordinates == old.common_coordinates && row.key < old.key)))
        }) {
            best = Some(assignment);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_medoid_json_preserves_descriptor_bits() {
        // This actual medoid's flux changed by one ULP without float_roundtrip.
        let encoded = r#"{"record_id":"00-sine-110-0.04-both-bus0-cut096000","raw_values":[6.716344335166367,0.25948290271136704,-5.1351418420134545,0.054411315849938184,0.012374198076345993,-0.0],"mask":63}"#;
        let expected: [f64; 6] = [
            6.716344335166367,
            0.25948290271136704,
            -5.1351418420134545,
            0.054411315849938184,
            0.012374198076345993,
            -0.0,
        ];
        let medoid: crate::config::TemporalBodyMedoid = serde_json::from_str(encoded).unwrap();
        assert_eq!(
            medoid.raw_values.map(f64::to_bits),
            expected.map(f64::to_bits)
        );
        let value: serde_json::Value = serde_json::from_str(encoded).unwrap();
        let rewritten = serde_json::to_string(&value).unwrap();
        let recovered: crate::config::TemporalBodyMedoid =
            serde_json::from_str(&rewritten).unwrap();
        assert_eq!(
            recovered.raw_values.map(f64::to_bits),
            expected.map(f64::to_bits)
        );
    }
    use crate::config::TemporalBodyMedoid;

    fn config() -> AppConfig {
        let mut config = AppConfig::default();
        config.temporal_body = Some(TemporalBodyConfig {
            means: [0.; 6],
            deviations: [1.; 6],
            accent_means: [0.; 2],
            accent_deviations: [1.; 2],
        });
        config.temporal_body_prototypes = Some(TemporalBodyPrototypesConfig {
            model_version: "12".repeat(32),
            sample_rate: config.audio.sample_rate,
            nfft: config.analysis.nfft,
            hop_size: config.analysis.hop_size,
            means: [0.; 6],
            deviations: [1.; 6],
            accent_means: [0.; 2],
            accent_deviations: [1.; 2],
            medoids: vec![
                TemporalBodyMedoid {
                    record_id: "a".into(),
                    raw_values: [2.; 6],
                    mask: 63,
                },
                TemporalBodyMedoid {
                    record_id: "b".into(),
                    raw_values: [4.; 6],
                    mask: 63,
                },
            ],
        });
        config
    }

    #[test]
    fn model_validation_rejects_changed_scales_acquisition_ids_and_capacity() {
        let good = config();
        good.validate().unwrap();
        let roundtrip: AppConfig = toml::from_str(&toml::to_string(&good).unwrap()).unwrap();
        roundtrip.validate().unwrap();
        for change in 0..11 {
            let mut bad = good.clone();
            let model = bad.temporal_body_prototypes.as_mut().unwrap();
            match change {
                0 => model.means[0] = 1.,
                1 => model.deviations[0] = 2.,
                2 => model.accent_means[0] = 1.,
                3 => model.sample_rate += 1,
                4 => model.nfft *= 2,
                5 => model.hop_size *= 2,
                6 => model.medoids.swap(0, 1),
                7 => model.medoids[1].record_id = "a".into(),
                8 => model.medoids[0].mask = 0,
                9 => model.model_version = "invalid".into(),
                10 => bad.temporal_body = None,
                _ => unreachable!(),
            }
            assert!(bad.validate().is_err(), "change {change}");
        }
        let mut bad = good.clone();
        bad.temporal_body_prototypes
            .as_mut()
            .unwrap()
            .medoids
            .clear();
        assert!(bad.validate().is_err());
        let mut bad = good;
        bad.temporal_body_prototypes.as_mut().unwrap().medoids = (0..9)
            .map(|i| TemporalBodyMedoid {
                record_id: format!("{i}"),
                raw_values: [0.; 6],
                mask: 63,
            })
            .collect();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn matching_uses_current_record_and_never_supplies_missing_coordinates() {
        let config = config();
        let scales = config.temporal_body.unwrap();
        let model = Prototypes::new(config.temporal_body_prototypes.as_ref().unwrap(), scales);
        assert_eq!(model.version, [0x12; 32]);
        let mut record = Record {
            active: true,
            raw_values: [2.; 6],
            mask: 63,
            ..Record::default()
        };
        assert_eq!(model.assign(&record, scales).unwrap().key, (0, 0));
        record.raw_values = [4.; 6];
        record.body_generation += 1;
        assert_eq!(model.assign(&record, scales).unwrap().key, (1, 0));
        record.mask = 4;
        assert_eq!(model.assign(&record, scales).unwrap().common_coordinates, 1);
        record.mask = 0;
        assert!(model.assign(&record, scales).is_none());
        record.mask = 63;
        record.raw_values = [3.; 6];
        assert!(model.assign(&record, scales).is_none());
        record.raw_values = [2.; 6];
        record.active = false;
        assert!(model.assign(&record, scales).is_none());
    }
}
