print(">>> Experiment: Agent Interaction <<<");

spawn("anchor", 1, #{ body: #{ amp: 0.6 }, pitch: #{ center_hz: 65.41 } });
wait(1.5);
spawn("seeker", 2, #{ body: #{ amp: 0.4 }, pitch: #{ center_hz: 140.0 } });
wait(2.0);
set("anchor", #{ pitch: #{ center_hz: 87.31 } });
wait(2.0);
end();
