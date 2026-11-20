"""Generate the signed 3-D Jeub et al. (2011) coherence grid used during FIR synthesis."""
from pathlib import Path

import numpy as np
from scipy.special import j0, roots_legendre


def curve(frequencies, order):
    # Gauss-Legendre quadrature in u=cos(theta) and radius. Azimuth integrates to 2*pi*J0.
    nodes, weights = roots_legendre(order)
    u, wu = (nodes[:, None] + 1) / 2, weights[:, None] / 2
    r, wr = (nodes + 1) * 0.075 / 2, weights * 0.075 / 2
    gap = (0.17 - 0.15) / 2
    lm = np.hypot(r, gap)
    radial = r * wr / lm
    argument = r * np.sqrt(1 - u*u)
    result = []
    for frequency in frequencies:
        k = 2 * np.pi * frequency / 340
        h2 = np.exp(1j*k*0.085*u)
        factor = (1j*k*(u + gap/lm) + gap/lm**2) / 2
        plate = np.sum(factor * np.exp(-1j*k*lm) * j0(k*argument) * radial, axis=1, keepdims=True)
        h1 = h2.conj() - np.exp(-1j*k*0.075*u) * plate
        result.append(2*np.sum(wu*(h1*h2.conj()).real) / np.sum(wu*(abs(h1)**2 + 1)))
    return np.array(result)


if __name__ == '__main__':
    grid = np.arange(0, 24001, 25)
    values = curve(grid, 64)
    assert np.max(abs(values - curve(grid, 96))) < 1e-12
    assert np.max(abs(curve(grid[:-1] + 12.5, 64) - (values[:-1] + values[1:])/2)) < 0.001
    dest = Path(__file__).resolve().parents[2] / 'src/core/room_coherence.f64le'
    dest.write_bytes(values.astype('<f8').tobytes())
