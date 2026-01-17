print("TEST_OK: agent_intent_smoke");
let agents = create(sine, 3).amp(0.3);
flush();
wait(1.0);
release(agents);
wait(0.2);
