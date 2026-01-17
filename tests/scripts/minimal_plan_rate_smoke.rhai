print("TEST_OK: minimal_plan_rate_smoke");
scene("plan", || {
    let g = create(sine, 2).amp(0.3);
    flush();
    wait(0.5);
});
