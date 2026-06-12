print("TEST_OK: minimal_plan_rate_smoke");
section("plan", || {
    let g = place(sine(), at(440.0).count(2)).amp(0.3);
    flush();
    wait(0.5);
});
