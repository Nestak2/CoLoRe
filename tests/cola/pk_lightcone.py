#!/usr/bin/env python
"""Measure P(k) from CoLoRe density grids and compare against linear/halofit.

The grid is a lightcone, so we cut a sub-cube at a fixed comoving radius from
the observer (box centre) and measure P(k) there, at the redshift corresponding
to that radius. This mirrors the subcube_pk approach in
field_tests_colore_vs_theory.ipynb.
"""
import sys, os
import numpy as np

def read_density_grid(fname):
    with open(fname, 'rb') as f:
        nnodes    = np.fromfile(f, dtype=np.int32,   count=1)[0]
        size_flou = np.fromfile(f, dtype=np.int32,   count=1)[0]
        lbox      = np.fromfile(f, dtype=np.float64, count=1)[0]
        ngrid     = np.fromfile(f, dtype=np.int32,   count=1)[0]
        nz_here   = np.fromfile(f, dtype=np.int32,   count=1)[0]
        iz0_here  = np.fromfile(f, dtype=np.int32,   count=1)[0]
        dt = np.float32 if size_flou == 4 else np.float64
        d = np.fromfile(f, dtype=dt, count=nz_here*ngrid*ngrid)
    return d.reshape(nz_here, ngrid, ngrid), lbox, ngrid


def measure_pk(delta, lsub, nbin=24, kmin=None, kmax=None):
    """Isotropic P(k) of a cubic real field of side lsub (Mpc/h)."""
    n = delta.shape[0]
    dk = 2*np.pi/lsub
    fk = np.fft.rfftn(delta) * (lsub/n)**3
    kz = np.fft.fftfreq(n, d=1.0/n)*dk
    kx = np.fft.rfftfreq(n, d=1.0/n)*dk
    KZ, KY, KX = np.meshgrid(kz, kz, kx, indexing='ij')
    kmod = np.sqrt(KX**2+KY**2+KZ**2)
    p3d = np.abs(fk)**2 / lsub**3
    if kmin is None: kmin = dk
    if kmax is None: kmax = 0.5*n*dk       # Nyquist
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbin+1)
    idx = np.digitize(kmod.ravel(), bins)-1
    w = (kmod.ravel() > 0) & (idx >= 0) & (idx < nbin)
    kk = np.zeros(nbin); pp = np.zeros(nbin); nn = np.zeros(nbin)
    np.add.at(nn, idx[w], 1.0)
    np.add.at(kk, idx[w], kmod.ravel()[w])
    np.add.at(pp, idx[w], p3d.ravel()[w])
    good = nn > 0
    return kk[good]/nn[good], pp[good]/nn[good], nn[good]


def subcubes(grid, lbox, ngrid, r_target, nsub):
    """Six nsub^3 sub-cubes centred at comoving radius r_target from the
    observer, one along each of +-x, +-y, +-z. All six sit at the same radius,
    hence the same redshift, so their spectra can be averaged."""
    cen = ngrid//2                          # observer cell (box centre)
    dx = lbox/ngrid
    ioff = int(round(r_target/dx))
    lo, hi = cen - nsub//2, cen + nsub - nsub//2
    out = []
    for axis in range(3):                   # grid axes are (z, y, x)
        for sgn in (+1, -1):
            j0 = cen + sgn*ioff - nsub//2
            if j0 < 0 or j0+nsub > ngrid:
                raise ValueError(f"sub-cube [{j0},{j0+nsub}) does not fit in {ngrid}")
            sl = [slice(lo, hi)]*3
            sl[axis] = slice(j0, j0+nsub)
            out.append(grid[tuple(sl)])
    return out


if __name__ == '__main__':
    import pyccl as ccl

    runs = sys.argv[1:]
    H0H = 0.6736
    OM, OB = 0.3153, 0.0493
    PKFILE = os.environ.get('PKFILE', 'pk.txt')

    # Build the theory from CoLoRe's own linear P(k) input, so "linear" means
    # exactly what CoLoRe was given, and halofit is computed on top of it.
    kh, pkh = np.loadtxt(PKFILE, unpack=True)       # k in h/Mpc, P in (Mpc/h)^3
    k_ccl = kh*H0H                                   # 1/Mpc
    p_ccl = pkh/H0H**3                               # Mpc^3

    base = ccl.Cosmology(Omega_c=OM-OB, Omega_b=OB, h=H0H, n_s=0.9649,
                         sigma8=0.8111, transfer_function='eisenstein_hu',
                         matter_power_spectrum='linear')
    a_arr = np.linspace(0.2, 1.0, 81)
    chi = ccl.comoving_radial_distance(base, a_arr)
    hoh0 = ccl.h_over_h0(base, a_arr)
    gf = ccl.growth_factor(base, a_arr)

    pk_lin_2d = np.array([g*g*p_ccl for g in gf])    # D(a)^2 P_lin(k, z=0)

    calc = ccl.CosmologyCalculator(
        Omega_c=OM-OB, Omega_b=OB, h=H0H, n_s=0.9649, sigma8=0.8111,
        background={'a': a_arr, 'chi': chi, 'h_over_h0': hoh0},
        growth={'a': a_arr, 'growth_factor': gf,
                'growth_rate': ccl.growth_rate(base, a_arr)},
        pk_linear={'a': a_arr, 'k': k_ccl,
                   'delta_matter:delta_matter': pk_lin_2d},
        nonlinear_model='halofit')

    NSUB     = int(os.environ.get('NSUB', 48))
    R_TARGET = float(os.environ.get('RTARGET', 1700.0))

    import scipy.optimize as op
    z_ref = op.brentq(
        lambda z: ccl.comoving_radial_distance(base, 1.0/(1+z))*H0H - R_TARGET,
        0.001, 3.0)
    a_ref = 1.0/(1+z_ref)
    print(f"# sub-cube: {NSUB}^3 cells centred at r = {R_TARGET} Mpc/h  ->  z = {z_ref:.4f}")

    out = {}
    for run in runs:
        fn = os.path.join(run, 'colore_dens_lightcone_0.dat')
        g, lbox, ngrid = read_density_grid(fn)
        cubes = subcubes(g, lbox, ngrid, R_TARGET, NSUB)
        lsub = NSUB*lbox/ngrid
        ps = []
        for sub in cubes:
            d = sub - sub.mean()
            k, p, nmode = measure_pk(d, lsub, nbin=int(os.environ.get('NBIN', 14)))
            ps.append(p)
        p = np.mean(ps, axis=0)
        out[run] = (k, p)
        allc = np.concatenate([c.ravel() for c in cubes])
        dd = allc - allc.mean()
        print(f"# {os.path.basename(run):8s} lsub={lsub:.1f} Mpc/h  <d>={allc.mean():+.4f} "
              f"rms={dd.std():.4f} min(1+d)={1+allc.min():.4f} "
              f"skew={float(((dd/dd.std())**3).mean()):+.3f}")

    kref = out[runs[0]][0]
    pl = calc.linear_matter_power(kref*H0H, a_ref)*H0H**3
    ph = calc.nonlin_matter_power(kref*H0H, a_ref)*H0H**3

    print("\n# ratio to halofit  (k_Nyquist of the parent grid = %.3f h/Mpc)"
          % (np.pi*ngrid/lbox))
    hdr = f"{'k[h/Mpc]':>9s} {'lin/hf':>7s}"
    for run in runs:
        hdr += f" {os.path.basename(run):>8s}"
    print(hdr)
    for i, kv in enumerate(kref):
        row = f"{kv:9.4f} {pl[i]/ph[i]:7.3f}"
        for run in runs:
            row += f" {out[run][1][i]/ph[i]:8.3f}"
        print(row)

    np.savez(os.environ.get('SAVE', 'pk_results.npz'),
             k=kref, p_lin=pl, p_hf=ph, z_ref=z_ref,
             **{os.path.basename(r): out[r][1] for r in runs})
