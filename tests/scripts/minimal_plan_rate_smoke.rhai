print("TEST_OK: minimal_plan_rate_smoke");

spawn("plan_test", 2, #{ body: #{ amp: 0.3 } });

wait(3.0);
end();
