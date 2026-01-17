print("TEST_OK: stable_order_same_time_actions");
let drones = create(sine, 1).amp(0.2);
flush();
drones.freq(200.0);
drones.freq(300.0);
flush();
wait(0.2);
