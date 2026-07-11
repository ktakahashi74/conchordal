print("TEST_OK: handle_index_and_iter");
let populations = [];
for i in 0..3 {
    populations.push(place(sine(), at(220.0 + i * 30.0).count(1)));
}
flush();
for population in populations {
    population.amp(0.05);
}
flush();
wait(0.2);
