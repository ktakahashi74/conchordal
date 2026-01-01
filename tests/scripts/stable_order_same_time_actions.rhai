print("TEST_OK: stable_order_same_time_actions");
let drones = spawn("drones", 1);
drones[0].set_freq(200.0);
drones[0].set_freq(300.0);
finish();
