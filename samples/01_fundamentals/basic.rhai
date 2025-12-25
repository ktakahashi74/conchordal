let life = #{
    type: "sustain",
    initial_energy: 1.0,
    metabolism_rate: 0.05,
    recharge_rate: 0.0, 
    action_cost: 0.0,   
    envelope: #{
        attack_sec: 0.5,
        decay_sec: 0.5,
        sustain_level: 1.0
    }
};

let method = #{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 };

spawn_agents("my_swarm", method, life, 5, 0.15);

wait(20.0);

finish();
