"""Independent Decimal reference: Todd 1992 eqs.7–10, Friberg 1999 eqs.2–4.

No rendered-performance measurements or final-note stimulus corrections.
Run from repository root; the Rust implementation uses different, rationalized forms.
"""
from decimal import Decimal as D, localcontext
import json
from pathlib import Path

rows = []
with localcontext() as ctx:
    ctx.prec = 400
    for weight in ["0", "5e-324", "1e-30", "2.220446049250313e-16", "0.5", "1", "100"]:
        r = min(D(weight), D(1))
        k = (1 - D(".7") * r) ** 3 - 1
        for position in ["0", ".125", ".25", ".5", ".75", ".9", "1"]:
            x = D(position)
            v = (1 + k*x) ** (D(1)/3)
            f = 3 * ((1+k*x) ** (D(2)/3)-1)/(2*k) if k else x
            inverse = ((1+2*k*f/3) ** (D(3)/2)-1)/k if k else f
            potential = -D(".5") * (x/D(".5") if x <= D(".5") else (1-x)/D(".5"))
            intensity = 100 * 2 * (0-potential)
            gain = (1+r*intensity/100).sqrt()
            rows.append(dict(weight=weight, effective_weight=str(r), x=str(x), v=format(v, ".36g"), F=format(f, ".36g"), inverse=format(inverse, ".36g"),
                             gain=format(gain, ".36g"), origin="17", length="300", fs=48000,
                             mapped_seconds=format(17+300*f, ".36g"), mapped_sample=int(((17+300*f)*48000+D('.5')).to_integral_value(rounding='ROUND_FLOOR'))))
Path("tests/research/pe1_reference.json").write_text(json.dumps(rows, indent=2)+"\n")
