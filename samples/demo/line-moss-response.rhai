// Line

output_gain(9.0);
temporal_basin(1.6, 2.4);
meter_stability(0.55);

let line = harmonic().brain("drone").sustain().seek_consonance()
    .modes(harmonic_modes().count(8)).amp(0.04).brightness(0.0).attack_noise(-12.0)
    .unison(3).spread(0.3333333)
    .temperature(0.0).move_cost(0.125)
    .pitch_apply_mode("rearticulate").leave_self_out(true).adsr(0.02, 0.5, 0.8, 2.0);
let moss = flora().amp(0.035).fade(6.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25).respond_to_onsets(16);
let lichen = flora().tone("resonant").amp(0.025).fade(8.0).pace(1.0)
    .pitch_apply_mode("rearticulate").rest_on_habituation(0.25);

let field = place(moss, random(98.0, 1568.0));
wait(2.0);
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
wait(3.0);
