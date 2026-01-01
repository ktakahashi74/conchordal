print("TEST_OK: stable_order_for_loop_expansion");
let drones = spawn("drones", 2);
for a in drones {
    set(a, #{ amp: 0.1 });
    set(a, #{ freq: 220.0 });
}
end();
