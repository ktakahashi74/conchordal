// DEMO-B1 draft (research copy in .orchestration, not a published sample).
// Built only from parts the author heard as music: Flora at pace 1, rain with 1/f sway,
// and a held, anchored field. No Voice searches for pitch; no progression is written.
// The single founder pitch, register ranges, amplitudes, rain rates and the roughness-aversion
// arc are the drafter's composer choices for the author to reshape.

output_gain(4.5);
temporal_basin(1.6, 2.4);

let root_hz = 110.0;

let field = harmonic().brain("drone").sustain().anchor()
    .modes(harmonic_modes().count(8)).amp(0.045).brightness(0.32)
    .adsr(6.0, 1.0, 0.95, 8.0);
let rainfall = rain().amp(0.028).rate(6).sway(1).objects(4).fade(6);
let moss = flora().amp(0.03).fade(8.0).pace(1.0).pitch_apply_mode("rearticulate");
let lichen = flora().tone("resonant").amp(0.02).fade(10.0).pace(1.0)
    .pitch_apply_mode("rearticulate");

// The field is first sensed only by the ecology; the rain reveals it.
let hidden = place(field.send(habitat_bus), at(root_hz));
let shower = place(rainfall, gap(150, 5000));
section("rain", || { wait(30.0); });

let ground = place(moss, consonance(root_hz, 880.0));
section("germination", || { wait(45.0); });

let canopy = place(lichen, consonance(440.0, 3520.0));
section("canopy", || {
    shower.rate(14);
    ground.amp(0.04);
    set_roughness_aversion(0.6);
    wait(40.0);
});

section("storm", || {
    shower.rate(36);
    canopy.amp(0.032);
    set_roughness_aversion(0.25);
    wait(35.0);
});

section("clearing", || {
    set_roughness_aversion(1.6);
    shower.rate(8);
    canopy.amp(0.02);
    ground.amp(0.035);
    wait(40.0);
});

release(canopy);
release(shower);
section("after", || {
    set_roughness_aversion(1.0);
    ground.amp(0.025);
    wait(30.0);
});

release(ground);
release(hidden);
wait(10.0);
