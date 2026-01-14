print("TEST_OK: stable_order_for_loop_expansion");
spawn("drones", 2);
set("drones", #{ body: #{ amp: 0.1 } });
set("drones", #{ pitch: #{ freq: 220.0 } });
end();
