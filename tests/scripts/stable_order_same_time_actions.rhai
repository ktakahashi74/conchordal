print("TEST_OK: stable_order_same_time_actions");
spawn("drones", 1);
set("drones", #{ pitch: #{ center_hz: 200.0 } });
set("drones", #{ pitch: #{ center_hz: 300.0 } });
end();
