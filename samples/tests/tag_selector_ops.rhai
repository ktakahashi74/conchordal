spawn("drones", 3);
wait(1);
tag("drones").release(2.0);
wait(1);
tag("drones").remove();
finish();
