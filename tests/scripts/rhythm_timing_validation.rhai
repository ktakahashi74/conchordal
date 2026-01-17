print("TEST_OK: rhythm_timing_validation");
let metro = create(sine, 1).amp(0.6).phonation("decay");
flush();
for i in 0..4 {
    wait(0.25);
}
release(metro);
wait(0.2);
