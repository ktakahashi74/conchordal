// Line and moss, sharper response: TDEMO-1 pattern 4, which the author heard as temporal
// DCC on 2026-11-17, extended to about two minutes (the author, 2026-11-17: six minutes is
// long, two will do).
//
// The line attacks crisply and chooses each move itself; the moss's ecology runs faster
// after new onsets it hears outside its own sound (respond_to_onsets 16), and its births
// stay on the beats it hears. The line is away twice for 12 s, so the moss is also heard
// without the onsets it answers, and answering again when the line returns.
//
// WARNING (authored harmony): none is written. No progression, scale, root, key, pitch or
// interval set appears in this score. Each time the line enters, the field chooses its
// place; it then chooses every later pitch from the consonance field it hears without
// itself. No anchor or beat carrier is placed.
//
// Value origins:
// - The line and the moss: TDEMO-1 pattern 4 unchanged (attack 0.02 s, brightness 0.45,
//   temperature 0.3, respond_to_onsets 16, the 12 s absence and the entry range).
// - Lichen and the amplitudes changed at each boundary: from "Line and moss". The line's
//   temperature stays at pattern 4's value instead of that score's schedule. The first
//   entry keeps pattern 4's timing (24 s, then 12 s away); the lichen's entry, the second
//   absence and the section lengths are the drafter's composer choices.
// - No seed is pinned on this page, so every performance differs.
output_gain(9.0);
temporal_basin(1.6, 2.4);
meter_stability(0.55);

let line = harmonic().brain("drone").sustain().seek_consonance()
    .modes(harmonic_modes().count(8)).amp(0.04).brightness(0.45)
    .temperature(0.3)
    .pitch_apply_mode("rearticulate").leave_self_out(true).adsr(0.02, 0.5, 0.8, 2.0);
let moss = flora().amp(0.035).fade(6.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25).respond_to_onsets(16);
let lichen = flora().tone("resonant").amp(0.025).fade(8.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25);

let field = place(moss, random(98.0, 1568.0));
wait(2.0);
// Placements stay outside section scopes, which release what they place when they return.
let first = place(line, consonance(147.0, 587.0));
section("germination", || { wait(24.0); });

release(first);
wait(12.0);
let second = place(line, consonance(147.0, 587.0));
section("second light", || { field.amp(0.04); wait(20.0); });

let crust = place(lichen, consonance(392.0, 3136.0));
section("canopy", || { field.amp(0.05); wait(26.0); });

release(second);
wait(12.0);
let third = place(line, consonance(147.0, 587.0));
release(crust);
section("after", || { field.amp(0.04); wait(21.0); });

release(field);
release(third);
// The Flora tails end within about two seconds of their release.
wait(3.0);
