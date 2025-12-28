spawn_agents("drones",
			#{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 },
 			#{
                body: #{ core: "sine" },
                temporal: #{ core: "drone" },
                field: #{ core: "pitch_hill_climb" },
                modulation: #{ core: "static", persistence: 0.5, habituation_sensitivity: 1.0, exploration: 0.0 }
            }, 5, 0.15);

wait(10.0);
release("my_drone", 3.0);
wait(4.0);
finish();
