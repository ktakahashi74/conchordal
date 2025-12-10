// Minimal Rhai scenario demonstrating two agents that trade dynamics over time.
// Uses only the supported scripting API: add_agent, wait, set_freq, set_amp, remove, finish.

let decay_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 };

// Start a steady pulse
add_agent("pulse", 330.0, 0.3, decay_life);

// After 1.5s, introduce a "ghost" agent that we accent with amplitude bumps
wait(1.5);
add_agent("ghost", 495.0, 0.2, decay_life);

// Simple dynamic exchange: lift ghost while ducking pulse, then swap back.
for _i in 0..6 {
    // ghost up, pulse down
    set_amp("ghost", 0.35);
    set_amp("pulse", 0.15);
    wait(0.5);
    // swap back
    set_amp("ghost", 0.12);
    set_amp("pulse", 0.28);
    wait(0.5);
    // micro retune ghost for variation (deterministic offsets)
    set_freq("ghost", 495.0 + 10.0);
    wait(0.1);
    set_freq("ghost", 495.0 - 5.0);
}

// Fade out
set_amp("ghost", 0.05);
set_amp("pulse", 0.1);
wait(1.0);
remove("ghost");
wait(0.5);
remove("pulse");
finish();
