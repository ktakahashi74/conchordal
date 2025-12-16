print(">>> Rhythm vitality demo: modulation self-oscillation <<<");

// Phase 0: baseline decay (no vitality)
scene("baseline");
print("0) Vitality=0.0 — modulation decays once the driver stops.");
set_rhythm_vitality(0.0);
add_agent("driver", 110.0, 0.35, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.2 });
wait(1.0);
remove("driver");
wait(1.0); // should hear the modulation fade out

// Phase 1: enable vitality to maintain a limit cycle
scene("self_oscillation");
print("1) Vitality=0.6 — short burst drives the bank, then it keeps oscillating.");
set_rhythm_vitality(0.6);
add_agent("driver", 110.0, 0.35, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.2 });
wait(0.8);
remove("driver"); // after this, no input; the modulation should continue audibly
wait(4.0);

// Phase 2: turn off vitality to let it decay again
scene("cooldown");
print("2) Vitality reset to 0.0 — modulation now decays.");
set_rhythm_vitality(0.0);
wait(2.0);

finish();
