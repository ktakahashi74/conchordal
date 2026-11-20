// Line and moss, sharper response: TDEMO-1 pattern 4, which the author heard as temporal
// DCC on 2026-11-17, extended to about two minutes (the author, 2026-11-17: six minutes is
// long, two will do).
//
// The line attacks crisply and draws no lottery: it moves when the field it hears without
// itself, eroded where sound has tired the modeled listener, is better elsewhere. The
// moss's ecology runs faster
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
// - The line and the moss: TDEMO-1 pattern 4 (attack 0.02 s, brightness 0.45,
//   respond_to_onsets 16, the 12 s absence and the entry range), except the line's
//   temperature and movement cost below.
// - temperature(0.0) and move_cost(0.125): version V3 of the DR-2 record comparison
//   (2026-11-19). Pattern 4 as heard used temperature 0.3, the default movement cost and
//   the former habituation drive. Under the current default drive, 0.125 is the largest
//   cost at which a line left its place without the lottery (HB-2). The version was taken
//   by a rule fixed before rendering; it is neither a default nor a recommended value.
//   The heard form: temperature(0.3), no move_cost, settings root-projection.toml.
// - The line's sound (2026-11-19): brightness(0.0), the partial structure the author found
//   least rough among five (DR-8, the line was heard as rough), and unison(3).spread(1/3),
//   the smallest layering the author heard with the moss (DR-11). Copies sit within +-8 cents
//   and the companions have a third of the centre's amplitude.
// - Lichen and the amplitudes changed at each boundary: from "Line and moss". The first
//   entry keeps pattern 4's timing (24 s, then 12 s away); the lichen's entry, the second
//   absence and the section lengths are the drafter's composer choices.
// - No seed is pinned on this page, so every performance differs.
output_gain(9.0);
temporal_basin(1.6, 2.4);
meter_stability(0.55);

let line = harmonic().brain("drone").sustain().seek_consonance()
    .modes(harmonic_modes().count(8)).amp(0.04).brightness(0.0)
    .unison(3).spread(0.3333333)
    .temperature(0.0).move_cost(0.125)
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
