print("Cadence ii-V-I");

let root = 110.0;
spawn("bass_ii", 1, #{ body: #{ amp: 0.36 }, pitch: #{ center_hz: root * 1.33 } });
spawn("pad_ii", 2, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: root * 2.0 } });
wait(2.0);

spawn("bass_v", 1, #{ body: #{ amp: 0.36 }, pitch: #{ center_hz: root * 1.5 } });
spawn("pad_v", 2, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: root * 2.5 } });
wait(2.0);

spawn("bass_i", 1, #{ body: #{ amp: 0.36 }, pitch: #{ center_hz: root } });
spawn("pad_i", 2, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: root * 2.0 } });
wait(2.0);

end();
