print("TEST_OK: tag_selector_ops");
spawn("drones", 3);
wait(1);
tag("drones").release(2.0);
wait(1);
tag("drones").remove();
finish();
