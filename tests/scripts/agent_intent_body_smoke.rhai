print("TEST_OK: agent_intent_body_smoke");

let sine_life = #{
    body: #{ core: "sine" }
};

let bright_life = #{
    body: #{
        core: "harmonic",
        mode: "harmonic",
        stiffness: 0.0,
        brightness: 0.9,
        comb: 0.0,
        damping: 0.5,
        vibrato_rate: 5.0,
        vibrato_depth: 0.0,
        jitter: 0.2,
        unison: 0.0
    }
};

spawn("agent_sine", 1, #{ amp: 0.3, life: sine_life });
spawn("agent_bright", 1, #{ amp: 0.3, life: bright_life });
wait(3.0);
end();
