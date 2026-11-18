// Rain and moss: TDEMO-1 pattern 5, which the author heard as temporal DCC on 2026-11-17,
// extended to about two minutes (the author, 2026-11-17: six minutes is long, two will do).
//
// The rain gathers its drops where the beat heard in the habitat is expected
// (follow_meter 2, research word, RW-13). The meter hears those drops, so a beat can rise
// from the rain itself and loosen as its 1/f sway thins it. The moss and the lichen are
// born only on beats they hear, so they ride the same beat; the moss's ecology also runs
// faster after new onsets it hears outside its own sound (respond_to_onsets 8).
//
// WARNING (authored harmony): none is written. No progression, scale, root, key, pitch or
// interval set appears in this score, and no anchor or beat carrier is placed. The rain
// falls on the field the moss and the lichen make; they grow on the field the rain makes.
//
// Value origins:
// - Moss, rain and tempo prior: TDEMO-1 pattern 5 unchanged (the rain recipe is B2's, with
//   follow_meter 2 chosen by rule from rendered records: at 4 most drops land on the beat,
//   at 1 no beat rises).
// - Lichen and the amplitudes changed at each boundary: from "Line and moss". The rain runs
//   about 110 s so its beat can rise and loosen more than once (two-minute records of
//   pattern 5 rose at 30-40 s and 60-70 s for seed 1). The lichen enters halfway so the
//   second half changes. Section lengths are the drafter's composer choices.
// - No seed is pinned on this page, so every performance differs.
output_gain(9.0);
temporal_basin(1.6, 2.4);
meter_stability(0.55);

let moss = flora().amp(0.035).fade(6.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25).respond_to_onsets(8);
let lichen = flora().tone("resonant").amp(0.025).fade(8.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25);
let rainfall = rain().amp(0.012).rate(6).sway(2).objects(4).fade(4)
    .leave_drops_out(true).objects_follow_drops(true).ring(2)
    .width(0.0052).impact_modes(harmonic_modes()).object_modes(harmonic_modes())
    .follow_meter(2.0);

let field = place(moss, random(98.0, 1568.0));
wait(3.0);
let shower = place(rainfall, consonance(150.0, 3520.0));
section("germination", || { wait(47.0); });

// Placements stay outside section scopes, which release what they place when they return.
let crust = place(lichen, consonance(392.0, 3136.0));
section("canopy", || { field.amp(0.05); wait(45.0); });

section("thinning", || { field.amp(0.04); wait(16.0); });

release(crust);
release(shower);
wait(6.0);
release(field);
// The Flora tails end within about two seconds of their release.
wait(3.0);
