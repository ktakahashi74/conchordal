print("Emergent harmony");

spawn("anchor", 1, #{ body: #{ amp: 0.3 }, pitch: #{ center_hz: 65.406 } });
wait(1.0);
spawn("seeker_fsharp", 1, #{ body: #{ amp: 0.08 }, pitch: #{ center_hz: 92.499 } });
spawn("seeker_csharp", 1, #{ body: #{ amp: 0.08 }, pitch: #{ center_hz: 69.296 } });
wait(2.5);
set("anchor", #{ pitch: #{ center_hz: 87.307 } });
wait(2.0);
end();
