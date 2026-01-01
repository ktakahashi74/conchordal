print("TEST_OK: handle_index_and_iter");
let drones = spawn("drones", 5);
set(drones[0], #{ freq: 220.0 });
for a in drones {
    set(a, #{ amp: 0.05 });
}
end();
