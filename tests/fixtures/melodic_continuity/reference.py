"""Independent Decimal-80 evaluation of text-consistent von Hippel (2000) AR(1).
No corpus fit: means/variances are algebraic and bins use the Gaussian integral.
Run with Python 3.12+ to regenerate reference.json.
"""
import decimal
import json
from pathlib import Path
from decimal import Decimal as D

decimal.getcontext().prec = 80
PI = D('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899')

def phi(z):
    # Integrate exp(-t*t/2) term by term, including tail cases.
    term = z
    total = term
    for k in range(1, 1000):
        term *= -z*z / (2*k)
        delta = term / (2*k+1)
        total += delta
        if abs(delta) < D('1e-65'):
            break
    else:
        raise RuntimeError('series did not converge')
    return D('0.5') + total/(2*PI).sqrt()

cases = []
for m,s,r,last in [(60,2,'0.5',62), (72,2,'0.5',74), (60,4,'0.5',62), (60,2,0,62), (60,2,'-0.5',62), (60,2,'0.99',62)]:
    m,s,r,last = map(D, map(str,(m,s,r,last)))
    mu = m+r*(last-m)
    v = s*s*(1-r*r)
    for lo,hi in [(-1,1),(0, D('0.5')),(7,8)]:
        a,b = mu+D(lo)*v.sqrt(), mu+D(hi)*v.sqrt()
        q = phi(D(hi))-phi(D(lo))
        cases.append(dict(m=float(m),s2=float(s*s),r=float(r),last=float(last),a=float(a),b=float(b),mu=float(mu),variance=float(v),q=float(q)))
Path(__file__).with_name('reference.json').write_text(json.dumps(dict(reference='Python Decimal, precision 80; Gaussian integral power series; no fitted parameters', cases=cases),indent=2)+'\n')
