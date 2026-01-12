print("TEST_OK: handle_index_and_iter");
spawn("drones", 5);
set("drones", #{ pitch: #{ center_hz: 220.0 } });
set("drones", #{ body: #{ amp: 0.05 } });
end();
