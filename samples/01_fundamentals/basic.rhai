spawn_agents("drones",
			#{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 },
 			#{ brain: "drone" }, 5, 0.15);

wait(10.0);
release("my_drone", 3.0);
wait(4.0);
finish();
