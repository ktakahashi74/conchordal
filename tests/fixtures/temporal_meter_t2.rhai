seed(21);
let a = place(sine().amp(0.04).metric().entrainment(0), at(440.0));
let b = place(sine().amp(0.04).metric(), at(110.0));
wait(3.0);
release(a);
release(b);
wait(0.1);
