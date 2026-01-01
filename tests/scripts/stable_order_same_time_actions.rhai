print("TEST_OK: stable_order_same_time_actions");
let drones = spawn("drones", 1);
set(drones[0], #{ freq: 200.0 });
set(drones[0], #{ freq: 300.0 });
end();
