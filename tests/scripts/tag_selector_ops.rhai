print("TEST_OK: tag_selector_ops");
spawn("drones", 3);
wait(1);
set("drones", #{ body: #{ amp: 0.1 } });
wait(1);
remove("drones");
end();
