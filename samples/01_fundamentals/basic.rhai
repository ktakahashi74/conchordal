let drones_tag = "drones";
spawn(drones_tag, 5, #{
    amp: 0.15,
    method: #{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 },
    life: #{
        body: #{ core: "sine" },
        articulation: #{ core: "drone" },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    }
});

wait(10.0);
release(drones_tag, 3.0);
wait(4.0);
end();
