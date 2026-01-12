print("TEST_OK: agent_intent_smoke");
spawn("agents", 3, #{ body: #{ amp: 0.3 } });
wait(4.0);
end();
