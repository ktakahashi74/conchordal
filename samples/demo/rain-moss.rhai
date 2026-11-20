// Rain

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
let shower = place(rainfall.rate(0.5), consonance(150.0, 3520.0));
section("germination", || {
    for step in 1..=46 {
        wait(1.0);
        shower.rate(0.5 * 12.0 ** (step.to_float() / 46.0));
    }
    wait(1.0);
});

let crust = place(lichen, consonance(392.0, 3136.0));
section("canopy", || { field.amp(0.05); wait(45.0); });

section("thinning", || { field.amp(0.04); wait(16.0); });

release(crust);
release(shower);
wait(6.0);
release(field);
wait(3.0);
