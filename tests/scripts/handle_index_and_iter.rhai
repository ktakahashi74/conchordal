print("TEST_OK: handle_index_and_iter");
let groups = [];
for i in 0..3 {
    groups.push(place(sine(), at(220.0 + i * 30.0).count(1)));
}
flush();
for g in groups {
    g.amp(0.05);
}
flush();
wait(0.2);
