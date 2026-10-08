seed(42);

let ground = place(harmonic().brain("drone").amp(0.035).sustain().anchor()
    .modes(harmonic_modes().count(4)).spread(0.0).unison(1), at(220.0));
wait(0.4);

let body = harmonic().amp(0.025).sustain().anchor()
    .modes(harmonic_modes().count(3)).spread(0.0).unison(1)
    .endurance(0.12).recovery(0.4).consonance_viability(0.6, 0.95)
    .respawn_random().respawn_capacity(4).respawn_min_c_level(0.0)
    .respawn_settle(consonance(160.0, 600.0).spacing(0.1))
    .respawn_background_death_rate(2.0).adsr(0.01, 0.01, 0.9, 0.02);
let colony = place(body, consonance(160.0, 600.0).count(4).spacing(0.1));
wait(0.3);
let later = place(sine().amp(0.02).sustain().anchor(),
    consonance(200.0, 700.0).peak().count(2));
wait(0.7);
release(later);
release(colony);
release(ground);
wait(0.12);
print("TEST_OK: body_birth_metabolism_flags");
