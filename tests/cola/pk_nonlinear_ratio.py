"""Non-linear enhancement of P(k) for each CoLoRe density type.

The lightcone makes an absolute comparison against halofit awkward: a sub-cube
cut out of the periodic box is not periodic (which costs ~20% of the power at
low k, identically for every density type), and the growth factor varies across
the cube.

Both problems cancel if the reference is CoLoRe's own Gaussian grid multiplied
cell-by-cell by the local growth factor D(r) -- that is exactly the linear
lightcone field, measured through exactly the same window. So we compare

    R_meas(k)   = P[dens_type] / P[linear lightcone]
    R_theory(k) = P_halofit(z_eff) / P_linear(z_eff)

The LPT/COLA grids are built by CIC deposit and the Gaussian grid is not, so the
CIC assignment window is divided out of the former.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pk_lightcone import read_density_grid, subcubes


def measure_pk_win(delta, lsub, nbin, kmin, kmax, dx_cic=None):
    """P(k) of a cubic field; if dx_cic is given, divide out the CIC window of a
    parent grid with cell size dx_cic (W = prod sinc^2(k_j dx/2), P /= W^2)."""
    n = delta.shape[0]
    dk = 2*np.pi/lsub
    fk = np.fft.rfftn(delta) * (lsub/n)**3
    kz = np.fft.fftfreq(n, d=1.0/n)*dk
    kx = np.fft.rfftfreq(n, d=1.0/n)*dk
    KZ, KY, KX = np.meshgrid(kz, kz, kx, indexing='ij')
    kmod = np.sqrt(KX**2+KY**2+KZ**2)
    p3d = np.abs(fk)**2 / lsub**3
    if dx_cic is not None:
        w = np.ones_like(p3d)
        for K in (KX, KY, KZ):
            u = K*dx_cic/2.0
            s = np.where(u == 0, 1.0, np.sin(np.where(u == 0, 1.0, u))/np.where(u == 0, 1.0, u))
            w *= s**2                      # CIC field window
        p3d = p3d/np.maximum(w**2, 1e-3)   # power window is W^2
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbin+1)
    idx = np.digitize(kmod.ravel(), bins)-1
    sel = (kmod.ravel() > 0) & (idx >= 0) & (idx < nbin)
    kk = np.zeros(nbin); pp = np.zeros(nbin); nn = np.zeros(nbin)
    np.add.at(nn, idx[sel], 1.0)
    np.add.at(kk, idx[sel], kmod.ravel()[sel])
    np.add.at(pp, idx[sel], p3d.ravel()[sel])
    good = nn > 0
    return kk[good]/nn[good], pp[good]/nn[good], nn[good]


if __name__ == '__main__':
    import pyccl as ccl
    import scipy.optimize as op

    runs = sys.argv[1:]
    H0H, OM, OB = 0.6736, 0.3153, 0.0493
    NSUB     = int(os.environ.get('NSUB', 64))
    R_TARGET = float(os.environ.get('RTARGET', 300.0))
    NBIN     = int(os.environ.get('NBIN', 12))
    PKFILE   = os.environ.get('PKFILE', 'pk.txt')
    CIC      = os.environ.get('CIC', '1') == '1'

    base = ccl.Cosmology(Omega_c=OM-OB, Omega_b=OB, h=H0H, n_s=0.9649,
                         sigma8=0.8111, transfer_function='eisenstein_hu',
                         matter_power_spectrum='linear')
    kh, pkh = np.loadtxt(PKFILE, unpack=True)
    a_arr = np.linspace(0.2, 1.0, 81)
    gf = ccl.growth_factor(base, a_arr)
    calc = ccl.CosmologyCalculator(
        Omega_c=OM-OB, Omega_b=OB, h=H0H, n_s=0.9649, sigma8=0.8111,
        background={'a': a_arr,
                    'chi': ccl.comoving_radial_distance(base, a_arr),
                    'h_over_h0': ccl.h_over_h0(base, a_arr)},
        growth={'a': a_arr, 'growth_factor': gf,
                'growth_rate': ccl.growth_rate(base, a_arr)},
        pk_linear={'a': a_arr, 'k': kh*H0H,
                   'delta_matter:delta_matter': np.array([g*g*pkh/H0H**3 for g in gf])},
        nonlinear_model='halofit')

    def r_of_z(z):
        return ccl.comoving_radial_distance(base, 1.0/(1+z))*H0H
    z_eff = op.brentq(lambda z: r_of_z(z)-R_TARGET, 1e-4, 3.0)
    a_eff = 1.0/(1+z_eff)

    # --- linear lightcone reference: gaussian grid x D(r) ---
    gg, lbox, ngrid = read_density_grid(os.path.join(runs[0], 'colore_dens_gaussian_0.dat'))
    dx = lbox/ngrid
    cen = ngrid//2
    ax = (np.arange(ngrid)-cen)*dx
    RR = np.sqrt(ax[:, None, None]**2 + ax[None, :, None]**2 + ax[None, None, :]**2)
    zgrid = np.linspace(0.0, 4.0, 4001)
    rgrid = r_of_z(zgrid)
    Dgrid = ccl.growth_factor(base, 1.0/(1+zgrid))
    Dr = np.interp(RR.ravel(), rgrid, Dgrid).reshape(RR.shape)
    lin_lc = gg*Dr

    lsub = NSUB*lbox/ngrid
    kmin, kmax = 2*np.pi/lsub, np.pi*ngrid/lbox

    def pk_of(field, cic):
        ps = []
        for sub in subcubes(field, lbox, ngrid, R_TARGET, NSUB):
            d = sub - sub.mean()
            k, p, n = measure_pk_win(d, lsub, NBIN, kmin, kmax,
                                     dx_cic=(dx if cic else None))
            ps.append(p)
        return k, np.mean(ps, axis=0)

    k, p_ref = pk_of(lin_lc, False)
    print(f"# z_eff = {z_eff:.4f} at r = {R_TARGET} Mpc/h; sub-cube {NSUB}^3 "
          f"= {lsub:.1f} Mpc/h; parent dx = {dx:.3f} Mpc/h; "
          f"CIC deconvolution {'on' if CIC else 'off'}")
    print(f"# shot noise if Poisson: dx^3 = {dx**3:.1f} (Mpc/h)^3")

    out = {}
    for run in runs:
        g, lb, ng = read_density_grid(os.path.join(run, 'colore_dens_lightcone_0.dat'))
        name = os.path.basename(run)
        cic = name not in ('lgnr', 'clip')
        _, p = pk_of(g, cic and CIC)
        out[name] = p

    rt = calc.nonlin_matter_power(k*H0H, a_eff)/calc.linear_matter_power(k*H0H, a_eff)

    print(f"\n# P(k) / P_linear-lightcone(k)   [theory column = halofit/linear at z_eff]")
    hdr = f"{'k[h/Mpc]':>9s} {'theory':>7s}"
    for name in out: hdr += f" {name:>8s}"
    print(hdr)
    for i, kv in enumerate(k):
        row = f"{kv:9.4f} {rt[i]:7.3f}"
        for name in out: row += f" {out[name][i]/p_ref[i]:8.3f}"
        print(row)

    np.savez(os.environ.get('SAVE', 'ratio_results.npz'),
             k=k, theory=rt, p_ref=p_ref, z_eff=z_eff, **out)
