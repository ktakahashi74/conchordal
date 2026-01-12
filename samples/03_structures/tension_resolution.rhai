print("Tension and resolution");

let root = 110.0;
spawn("bass", 1, #{ body: #{ amp: 0.4 }, pitch: #{ center_hz: root } });
spawn("pad", 2, #{ body: #{ amp: 0.25 }, pitch: #{ center_hz: root * 2.0 } });
wait(2.0);

spawn("tension", 2, #{ body: #{ amp: 0.3 }, pitch: #{ center_hz: root * 1.5 } });
wait(1.5);

spawn("resolve", 2, #{ body: #{ amp: 0.25 }, pitch: #{ center_hz: root * 2.0 } });
wait(2.0);

end();
