# COLA validation

Scripts used to validate the COLA density field (`dens_type = 4`) against theory.
They need `numpy`, `scipy` and `pyccl`.

All three read CoLoRe's native density grid output, so the runs must have
`output_density = true`.

## `pk_estimator_check.py`

Measures P(k) of `colore_dens_gaussian_0.dat` -- the z=0 linear Gaussian field
over the whole periodic box -- and compares it against the input CAMB-format
P(k) file. This validates the estimator itself and should agree to ~1% wherever
there are enough modes.

```
python pk_estimator_check.py <run_dir> <pk_file>
```

## `pk_nonlinear_ratio.py`

The main test. Reports the non-linear enhancement `P(k)/P_linear(k)` for each
density type and compares it against `halofit/linear` at the relevant redshift.

An absolute comparison against halofit is awkward on a lightcone: a sub-cube cut
out of the periodic box is not itself periodic (which costs ~20% of the power at
low k, identically for every density type), and the growth factor varies across
the cube. Both effects cancel if the reference is CoLoRe's own Gaussian grid
multiplied cell-by-cell by the local growth factor D(r) -- that is exactly the
linear lightcone field, measured through exactly the same window. The LPT and
COLA grids are built by CIC deposit and the Gaussian grid is not, so the CIC
assignment window is divided out of the former.

```
NSUB=64 RTARGET=150 NBIN=12 PKFILE=pk.txt \
  python pk_nonlinear_ratio.py <run_dir> [<run_dir> ...]
```

`RTARGET` is the comoving radius (Mpc/h) of the sub-cube centres and sets the
effective redshift; `NSUB` is the sub-cube side in cells. Six sub-cubes are
placed at +-x, +-y, +-z at that radius and their spectra averaged. Runs whose
directory basename is `lgnr` or `clip` are treated as not CIC-deposited.

## `pk_lightcone.py`

Absolute P(k) against linear and halofit, for reference. Provides
`read_density_grid`, `measure_pk` and `subcubes`, which the other two import.

## Regression check

The lognormal/1LPT/2LPT/clip paths must stay byte-identical when COLA is
changed, and `dens_type = 4` with `cola_n_steps = 0` and
`cola_lightcone_mode = 0` must reproduce `dens_type = 2` with `lpt_vels = 1`
bit-for-bit. Both were checked with `md5sum` on
`colore_dens_lightcone_0.dat`.

Note that CoLoRe seeds one RNG per OpenMP thread
(`seed_rng + IThread0 + ithr`, `src/fourier.c`), so the realisation depends on
`OMP_NUM_THREADS`. Any bit-for-bit comparison has to use the same thread count
on both sides.
