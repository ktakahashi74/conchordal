// RS-7 (line side of RS-1): the line and the moss each respond to heard onsets.
// This file differs from demo_p1r_w8_s20261113.rhai (the moss-only response, where the moss
// recipe adds .respond_to_onsets(8) to the control demo_p1_s20261113.rhai) in one call
// only: the line recipe adds .respond_to_onsets(1). The line's pitch re-choice clock
// advances after onsets it hears outside its own sound, read in its self-excluding
// observer; after it moves, a note keeps one ordinary cycle. It writes no pitch and no
// onset time; the field still decides whether and where the line moves.
// respond_to_onsets(1) on the line: one rung of the ladder 1, 2, 4, 8, 16 rendered for
// the record gates of RS-6 advice section 6, which select one value by rule. It is
// neither a default nor a recommended value. The moss keeps respond_to_onsets(8).
//
// Demo P1 (WK-3): a moving line and moss. No configuration file beyond the defaults
// (rendered with the all-default FD-1 configuration, AMB-1/inputs/v6b.toml).
//
// WARNING (authored harmony): none is written. No progression, scale, root, key, pitch or
// interval set appears in this score. The line starts at 196 Hz and then chooses every
// later pitch from the consonance field it hears without itself; the moss and the lichen
// grow on the field around them.
//
// Value origins:
// - The whole score is the pace-1 version of RG-9 (AMB-1/inputs/a-none.rhai), unchanged:
//   output_gain 9, temporal_basin(1.6, 2.4), meter_stability(0.55), the line, the moss and
//   the lichen recipes, the six sections, their lengths and the values changed at each
//   boundary (moss amp 0.035 -> 0.04 -> 0.05 -> 0.04, lichen amp 0.025 -> 0.035, line
//   temperature 0.10 -> 0.3 -> 0.08 -> 0.05). The author heard this form on 2026-10-30.
// - seed 20261113: one of five consecutive seeds 20261109-20261113 rendered for the RS-2
//   record gates (20261109 is the GT material's seed; the others have no meaning).
// - at(196.0) is only the line's starting position: seek_consonance() moves it.
// Section boundaries and the values changed there are the composer's macro form; they are
// not evidence of self-organisation.
seed(20261113);
output_gain(9.0);
temporal_basin(1.6, 2.4);
meter_stability(0.55);

let wanderer = harmonic().brain("drone").sustain().seek_consonance()
    .modes(harmonic_modes().count(8)).amp(0.04).brightness(0.38)
    .temperature(0.10)
    .pitch_apply_mode("rearticulate").leave_self_out(true).adsr(1.5, 0.8, 0.9, 3.0)
    .respond_to_onsets(1);
let moss = flora().amp(0.035).fade(6.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25).respond_to_onsets(8);
let lichen = flora().tone("resonant").amp(0.025).fade(8.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25);

let drift = place(wanderer, at(196.0));
wait(4.0);
let field = place(moss, random(98.0, 1568.0));
section("germination", || { wait(46.0); });

section("second light", || { field.amp(0.04); wait(56.0); });

let crust = place(lichen, consonance(392.0, 3136.0));
section("pulse", || { wait(80.0); });

section("canopy", || {
    field.amp(0.05);
    crust.amp(0.035);
    drift.temperature(0.3);
    wait(72.0);
});

section("thinning", || {
    drift.temperature(0.08);
    field.amp(0.04);
    wait(52.0);
});

release(crust);
section("after", || { drift.temperature(0.05); wait(36.0); });

release(field);
release(drift);
wait(8.0);
