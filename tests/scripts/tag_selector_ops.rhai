print("TEST_OK: tag_selector_ops");
spawn("drones", 3);
wait(1);
release("drones", 2.0);
wait(1);
remove("drones");
end();
