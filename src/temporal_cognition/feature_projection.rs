//! Shared raw-feature arithmetic; hypothetical inputs never update an observation stream.

use crate::core::log2space::Log2Space;

pub(super) mod window;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "origin", content = "value", rename_all = "snake_case")]
pub(super) enum Feature {
    Unsupported,
    Observed(f64),
    Projected(f64),
}

impl Feature {
    pub(super) fn value(self) -> Option<f64> {
        match self {
            Self::Unsupported => None,
            Self::Observed(value) | Self::Projected(value) => Some(value),
        }
    }

    fn projected(self) -> bool {
        matches!(self, Self::Projected(_))
    }

    fn from(value: Option<f64>, projected: bool) -> Self {
        match value {
            None => Self::Unsupported,
            Some(value) if projected => Self::Projected(value),
            Some(value) => Self::Observed(value),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Spectrum<'a> {
    pub energy_scan: &'a [f64],
    pub projected: bool,
}

#[derive(Clone, Copy)]
pub(super) struct Frame<'a> {
    pub energy: Feature,
    pub bus_energy: Feature,
    pub spectrum: Option<Spectrum<'a>>,
}

pub(super) struct Projection {
    /// Same order and floors as RawDescriptor; provenance follows each dependency.
    pub values: [Feature; 10],
    /// The detector's alternate floor must not change the descriptor's fixed floor.
    pub accent_flux: f64,
}

/// Callers supply only physically supported inputs and an eligible adjacent frame.
/// This function has no clock, group, accent, or learning state to advance.
pub(super) fn evaluate(
    space: &Log2Space,
    current: Frame<'_>,
    previous: Option<Frame<'_>>,
    accent_rms_floor: f64,
) -> Result<Projection, &'static str> {
    if !accent_rms_floor.is_finite() || accent_rms_floor <= 0. {
        return Err("invalid projection accent floor");
    }
    for frame in std::iter::once(current).chain(previous) {
        if let Some(spectrum) = frame.spectrum {
            space.assert_scan_len_named(spectrum.energy_scan, "feature_projection_energy_scan");
        }
        if [frame.energy.value(), frame.bus_energy.value()]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite() || v < 0.)
            || frame
                .energy
                .value()
                .zip(frame.bus_energy.value())
                .is_some_and(|(energy, bus)| energy > bus + 1e-12)
            || frame
                .spectrum
                .is_some_and(|s| s.energy_scan.iter().any(|v| !v.is_finite() || *v < 0.))
        {
            return Err("invalid assigned raw-feature energy");
        }
    }
    let current_scan = current.spectrum.map(|s| s.energy_scan);
    let previous_scan = previous.and_then(|p| p.spectrum).map(|s| s.energy_scan);
    let (mut mass, mut weighted, mut previous_mass, mut flux, mut accent_flux) =
        (0., 0., 0., 0., 0.);
    let spectral_floor = accent_rms_floor * accent_rms_floor;
    for (i, &coordinate) in space.centers_log2.iter().enumerate() {
        let now = current_scan.map(|scan| scan[i]);
        let old = previous_scan.map(|scan| scan[i]);
        if let Some(value) = now {
            mass += value;
            weighted += value * f64::from(coordinate);
        }
        if let Some(value) = old {
            previous_mass += value;
        }
        if let (Some(now), Some(old)) = (now, old) {
            flux += (0.5 * now.max(1e-12).log2() - 0.5 * old.max(1e-12).log2()).max(0.);
            if spectral_floor != 1e-12 {
                accent_flux += (0.5 * now.max(spectral_floor).log2()
                    - 0.5 * old.max(spectral_floor).log2())
                .max(0.);
            }
        }
    }
    if spectral_floor == 1e-12 {
        accent_flux = flux;
    }
    let mut values = [Feature::Unsupported; 10];
    values[2] = Feature::from(
        current.energy.value().map(|v| v.sqrt().max(1e-6).log2()),
        current.energy.projected(),
    );
    values[6] = Feature::from(
        current
            .energy
            .value()
            .zip(current.bus_energy.value())
            .map(|(energy, bus)| if bus > 0. { energy / bus } else { 0. }),
        current.energy.projected() || current.bus_energy.projected(),
    );
    if mass > 0. {
        let center = weighted / mass;
        let mut spread = 0.;
        let mut fractions = [0.; 3];
        for (&coordinate, &energy) in space.centers_log2.iter().zip(current_scan.unwrap()) {
            let delta = f64::from(coordinate) - center;
            let weight = energy / mass;
            spread += weight * delta * delta;
            fractions[if delta < -0.5 {
                0
            } else if delta > 0.5 {
                2
            } else {
                1
            }] += weight;
        }
        let projected = current.spectrum.unwrap().projected;
        values[0] = Feature::from(Some(center), projected);
        values[1] = Feature::from(Some(spread.sqrt()), projected);
        for (dst, value) in values[7..].iter_mut().zip(fractions) {
            *dst = Feature::from(Some(value), projected);
        }
    }
    if let Some(previous) = previous {
        if let Some((old, now)) = previous.energy.value().zip(values[2].value()) {
            let delta = now - old.sqrt().max(1e-6).log2();
            let projected = current.energy.projected() || previous.energy.projected();
            values[3] = Feature::from(Some(delta.max(0.)), projected);
            values[4] = Feature::from(Some((-delta).max(0.)), projected);
        }
        if let (Some(now), Some(old)) = (current.spectrum, previous.spectrum)
            && !current.energy.value().is_some_and(|e| e > 0. && mass == 0.)
            && !previous
                .energy
                .value()
                .is_some_and(|e| e > 0. && previous_mass == 0.)
        {
            values[5] = Feature::from(
                Some(flux / space.n_bins() as f64),
                now.projected
                    || old.projected
                    || current.energy.projected()
                    || previous.energy.projected(),
            );
        }
    }
    if values
        .iter()
        .filter_map(|v| v.value())
        .any(|v| !v.is_finite())
    {
        return Err("raw-feature accumulation overflow");
    }
    Ok(Projection {
        values,
        accent_flux,
    })
}

#[cfg(test)]
mod tests;
