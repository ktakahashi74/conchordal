print("TEST_OK: stable_order_same_time");
let drones = spawn("drones", 2);
set(drones[0], #{ freq: 200.0 });
set(drones[0], #{ freq: 300.0 });
end();
