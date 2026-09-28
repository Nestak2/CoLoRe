"""Validate the P(k) estimator on the full periodic box.

colore_dens_gaussian_0.dat is the z=0 linear Gaussian field over the whole
periodic box, so its measured P(k) must reproduce the input CAMB-format P(k)
that CoLoRe was given. If that works, the estimator is right and any residual
in the lightcone sub-cube test is geometry, not normalisation.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pk_lightcone import read_density_grid, measure_pk

run = sys.argv[1]
pkfile = sys.argv[2]

g, lbox, ngrid = read_density_grid(os.path.join(run, 'colore_dens_gaussian_0.dat'))
print(f"# gaussian grid: {g.shape} lbox={lbox:.2f} Mpc/h  <d>={g.mean():+.5f} rms={g.std():.5f}")
d = g - g.mean()
k, p, n = measure_pk(d, lbox, nbin=22, kmin=2*np.pi/lbox, kmax=np.pi*ngrid/lbox)

kh, pkh = np.loadtxt(pkfile, unpack=True)
pin = np.interp(np.log(k), np.log(kh), np.log(pkh))
pin = np.exp(pin)

print(f"{'k[h/Mpc]':>9s} {'P_meas':>11s} {'P_input':>11s} {'ratio':>7s} {'Nmodes':>8s}")
for i in range(len(k)):
    print(f"{k[i]:9.4f} {p[i]:11.2f} {pin[i]:11.2f} {p[i]/pin[i]:7.3f} {n[i]:8.0f}")
