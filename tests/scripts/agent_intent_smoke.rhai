print("TEST_OK: agent_intent_smoke");
spawn("agents", 3, #{ amp: 0.3 });
set_plan_rate_tag("agents", 1.0);
wait(4.0);
end();
