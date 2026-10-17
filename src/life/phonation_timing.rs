use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateBoundary {
    pub gate: u64,
    pub tick: Tick,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThetaGrid {
    pub boundaries: Vec<GateBoundary>,
}

impl ThetaGrid {
    #[cfg(test)]
    pub fn from_candidates(candidates: &[CandidatePoint]) -> Self {
        let mut grid = Self::default();
        grid.rebuild_from_candidates(candidates);
        grid
    }

    /// Rebuild the grid in place, reusing the boundary buffer. Runs every hop on
    /// the audio thread, so it must not allocate once warm.
    /// Input candidates must be sorted by tick ascending.
    pub fn rebuild_from_candidates(&mut self, candidates: &[CandidatePoint]) {
        debug_assert!(
            candidates.windows(2).all(|p| p[0].tick <= p[1].tick),
            "candidates must be sorted by tick"
        );
        self.boundaries.clear();
        let mut last_gate = None;
        let mut last_tick = None;
        for candidate in candidates {
            if let Some(prev_gate) = last_gate {
                if candidate.gate < prev_gate {
                    // A clock emitting a decreasing gate is a producer bug. Drop
                    // the candidate instead of aborting the performance; dev
                    // builds trip the assert. A repeated gate is legitimate and
                    // is skipped silently below.
                    debug_assert!(false, "candidate gates must be non-decreasing");
                    continue;
                }
                if candidate.gate == prev_gate || candidate.gate != prev_gate.saturating_add(1) {
                    continue;
                }
                if last_tick.is_some_and(|t| candidate.tick < t) {
                    continue;
                }
            }
            self.boundaries.push(GateBoundary {
                gate: candidate.gate,
                tick: candidate.tick,
            });
            last_gate = Some(candidate.gate);
            last_tick = Some(candidate.tick);
        }
    }

    fn boundaries_for_gate(&self, gate: u64) -> Option<(GateBoundary, GateBoundary)> {
        if self.boundaries.is_empty() {
            return None;
        }
        let start_gate = self.boundaries[0].gate;
        let idx = gate.checked_sub(start_gate)? as usize;
        let next_idx = idx.checked_add(1)?;
        let b0 = *self.boundaries.get(idx)?;
        let b1 = *self.boundaries.get(next_idx)?;
        if b0.gate != gate || b1.gate != gate.saturating_add(1) {
            return None;
        }
        Some((b0, b1))
    }

    /// Returns a tick by rounding the interpolated position, clamping interior phases away from
    /// gate boundary ticks.
    pub fn tick_at(&self, gate: u64, phase_in_gate: f32) -> Option<Tick> {
        let (b0, b1) = self.boundaries_for_gate(gate)?;
        let phase = phase_in_gate.clamp(0.0, 1.0);
        if phase > 0.0 && phase < 1.0 && b1.tick <= b0.tick.saturating_add(1) {
            return None;
        }
        let dt = b1.tick.saturating_sub(b0.tick) as f64;
        let tick_f = b0.tick as f64 + dt * phase as f64;
        let tick = tick_f.round() as Tick;
        if phase > 0.0 && phase < 1.0 {
            let lo = b0.tick.saturating_add(1);
            let hi = b1.tick.saturating_sub(1);
            if lo > hi {
                return None;
            }
            return Some(tick.clamp(lo, hi));
        }
        Some(tick)
    }

    /// Extend the gate grid up to `target_gate_plus_1`. `fixed_period`, when set,
    /// extends at a constant tick spacing (the coupling clock's nominal onset
    /// period) so a note-off lands on the same grid that generated the onset.
    /// When `None`, the grid follows the adaptive theta (a `ThetaGate` clock).
    /// Using theta to extend a coupling-clock voice would make its note length
    /// drift with theta.
    pub fn ensure_boundaries_until(
        &mut self,
        ctx: &CoreTickCtx,
        target_gate_plus_1: u64,
        fixed_period: Option<f64>,
    ) {
        let Some(mut last) = self.boundaries.last().copied() else {
            return;
        };
        if last.gate >= target_gate_plus_1 {
            return;
        }
        let mut cursor_tick = last.tick;
        let mut next_gate = last.gate.saturating_add(1);
        let mut added = 0u64;
        while last.gate < target_gate_plus_1 && added < MAX_EXTRA_GATES {
            let mut next_tick = match fixed_period {
                Some(period) => {
                    let step = (period.round() as Tick).max(1);
                    cursor_tick.saturating_add(step)
                }
                None => {
                    // The cursor is already a gate boundary, so the next gate
                    // is a full theta cycle away, not the frame's phase remainder.
                    let mut theta = ctx.rhythms.theta;
                    theta.phase = 0.0;
                    match next_gate_tick(cursor_tick, ctx.fs, theta, 0.0) {
                        Some(tick) => tick,
                        None => break,
                    }
                }
            };
            // Failsafe: if the clock still returns a non-advancing tick, force monotonic progress.
            if next_tick <= cursor_tick {
                next_tick = cursor_tick.saturating_add(1);
            }
            if next_tick <= cursor_tick {
                break;
            }
            let boundary = GateBoundary {
                gate: next_gate,
                tick: next_tick,
            };
            self.boundaries.push(boundary);
            last = boundary;
            cursor_tick = next_tick;
            next_gate = next_gate.saturating_add(1);
            added += 1;
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimingField {
    pub start_gate: u64,
    pub e_gate: Vec<f32>,
}

impl TimingField {
    pub fn from_values(start_gate: u64, e_gate: Vec<f32>) -> Self {
        Self { start_gate, e_gate }
    }

    #[cfg(test)]
    pub fn build_from(
        ctx: &CoreTickCtx,
        grid: &ThetaGrid,
        social: Option<(&SocialDensityTrace, f32)>,
        extra_gate_gain: f32,
    ) -> Self {
        let mut field = Self::default();
        field.rebuild_from(ctx, grid, social, extra_gate_gain);
        field
    }

    /// Uniform field (weight 1.0 at every gate), ignoring `env_open`/social.
    /// Used by fixed-rate clocks so their onsets are not gated by the adaptive
    /// theta envelope. Rebuilt in place: this runs every hop on the audio thread.
    pub fn rebuild_flat(&mut self, grid: &ThetaGrid) {
        self.e_gate.clear();
        let Some(start_gate) = grid.boundaries.first().map(|b| b.gate) else {
            self.start_gate = 0;
            return;
        };
        let last_gate = grid.boundaries.last().map(|b| b.gate).unwrap_or(start_gate);
        let len = (last_gate.saturating_sub(start_gate) as usize).saturating_add(1);
        self.start_gate = start_gate;
        self.e_gate.resize(len, 1.0);
    }

    /// Rebuild in place, reusing the gate buffer (per-hop audio-thread path).
    pub fn rebuild_from(
        &mut self,
        ctx: &CoreTickCtx,
        grid: &ThetaGrid,
        social: Option<(&SocialDensityTrace, f32)>,
        extra_gate_gain: f32,
    ) {
        const MAX_GATES_PER_HOP: u64 = 4096;
        self.e_gate.clear();
        if grid.boundaries.is_empty() {
            self.start_gate = 0;
            return;
        }
        debug_assert!(
            grid.boundaries
                .windows(2)
                .all(|pair| pair[0].tick <= pair[1].tick)
        );
        debug_assert!(
            grid.boundaries
                .windows(2)
                .all(|pair| pair[0].gate < pair[1].gate)
        );
        let mut rhythms = ctx.rhythms;
        let mut cursor_tick = ctx.now_tick;
        let mut expected_gate = grid.boundaries[0].gate;
        let mut last_weight = None;
        for boundary in &grid.boundaries {
            if boundary.tick > cursor_tick {
                let dt_sec = (boundary.tick - cursor_tick) as f32 / ctx.fs;
                rhythms.advance_in_place(dt_sec);
                cursor_tick = boundary.tick;
            }
            let mut weight = rhythms.env_open.clamp(0.0, 1.0);
            match social {
                Some((trace, coupling)) if coupling != 0.0 => {
                    let density = trace.density_at(boundary.tick);
                    if density.is_finite() {
                        let arg = (coupling * density).clamp(-10.0, 10.0);
                        weight = (weight * arg.exp()).max(0.0);
                        if !weight.is_finite() {
                            weight = 0.0;
                        }
                    }
                }
                _ => {}
            }
            let extra_gain = if extra_gate_gain.is_finite() {
                extra_gate_gain.max(0.0)
            } else {
                1.0
            };
            weight *= extra_gain;
            if !weight.is_finite() {
                weight = 0.0;
            }
            if boundary.gate > expected_gate {
                let fill_weight = last_weight.unwrap_or(1.0);
                let missing = boundary.gate - expected_gate;
                if missing > MAX_GATES_PER_HOP {
                    // Pathological gap: stop filling instead of aborting the
                    // performance. Gates past the end of `e_gate` read the 1.0
                    // fallback in `e()`, so truncating costs nothing a clamped
                    // fill would have preserved, and it keeps the reused buffer
                    // from growing by thousands of dead entries.
                    debug_assert!(false, "timing field gap too large");
                    break;
                }
                // Fill gaps to keep gate-indexed lookups contiguous.
                self.e_gate
                    .extend(std::iter::repeat_n(fill_weight, missing as usize));
            }
            self.e_gate.push(weight);
            last_weight = Some(weight);
            expected_gate = boundary.gate.saturating_add(1);
        }
        self.start_gate = grid.boundaries[0].gate;
    }

    pub fn e(&self, gate: u64) -> f32 {
        if gate < self.start_gate {
            return 1.0;
        }
        let idx = gate - self.start_gate;
        self.e_gate.get(idx as usize).copied().unwrap_or(1.0)
    }
}
