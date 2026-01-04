print("TEST_OK: minimal_plan_rate_smoke");

spawn("plan_test", 2, #{ amp: 0.3 });
set_plan_rate_tag("plan_test", 0.5);

wait(3.0);
end();
