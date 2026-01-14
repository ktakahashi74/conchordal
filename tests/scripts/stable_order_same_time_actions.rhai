print("TEST_OK: stable_order_same_time_actions");
spawn("drones", 1);
set("drones", #{ pitch: #{ freq: 200.0 } });
set("drones", #{ pitch: #{ freq: 300.0 } });
end();
