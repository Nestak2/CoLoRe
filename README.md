# CoLoRe

CoLoRe is a parallelized code to generate fast 3D realization of a wide variety of cosmological observations.

## 1 Methods

The methods used by CoLoRe are described in [Ramirez-Perez et al. 2021](https://arxiv.org/abs/2111.05069).

When in doubt, bear in mind that by default CoLoRe uses the following units:
 - Lenghts: Mpc/h
 - Angles: degrees


## 2 Compilation and usage

To compile CoLoRe, open the Makefile and edit it according to your
system. The default options (except for the paths to the external
libraries) should work for most systems.

### 2.1 Compilation options
The following compilation flags, tunable in the Makefile, can be modified
to control the behaviour of CoLoRe:

- CoLoRe may be very memory-demanding. To minimize the memory overhead use single precision floating point (`USE_PRECISION = yes`).
- OpenMP parallelization is enabled by setting the option `USE_OMP` to "yes".
- MPI parallelization is enabled by setting the option `USE_MPI` to "yes".
- The default bias model in CoLoRe is exponential. To select a different bias model add `-D_BIAS_MODEL_2` (exp-truncated) or `-D_BIAS_MODEL_3` (truncated).

### 2.2 Dependencies
CoLoRe uses 4 external packages:
 - GSL. The GNU Scientific Library (tested for versions 3.*)
 - FFTW. The Fastest Fourier Transform of the West (versions 3.*)
 - CFITSIO. FITS format library. This package is optional.
 - HDF5. HDF5 format library. This package is optional.
The paths to the corresponding headers and libraries should be correctly
set in the Makefile.

### 2.3 Running the code
Once the Makefile has been editted, typing 'make' should generate
the executable 'CoLoRe'. To run CoLoRe just type

> mpirun -np <number-of-nodes> ./CoLoRe <param_file>

where <param_file> is the path to the parameter file described in
section 3.


## 2.4 Density field types (`dens_type`)

The `dens_type` parameter selects how the physical matter density is grown from
the Gaussian field:

| `dens_type` | method |
|---|---|
| 0 | lognormal |
| 1 | first-order Lagrangian perturbation theory (1LPT) |
| 2 | second-order Lagrangian perturbation theory (2LPT) |
| 3 | clipped linear field |
| 4 | COLA |

Types 0-3 are single-shot: the displacement kernels are computed once and
applied with the growth factor evaluated at each cell's own comoving distance
from the observer.

**COLA** ([Tassev, Zaldarriaga & Eisenstein
2013](https://arxiv.org/abs/1301.0322)) instead evolves particles with a
particle-mesh force solver in the frame that comoves with their 2LPT
trajectory. Large scales stay exact by construction, while a few tens of
timesteps recover the non-linear small scales. Because CoLoRe's box *is* the
lightcone volume, each particle is deposited into the density and momentum grids
at the instant it crosses its own lightcone radius, following the approach of
[L-PICOLA](https://github.com/CullanHowlett/l-picola) (Howlett & Manera 2015).

Measured against halofit at z = 0.05 in a 590 Mpc/h box with `n_grid = 256`
(cell size 2.3 Mpc/h), the non-linear enhancement `P(k)/P_linear` is

| k [h/Mpc] | halofit | 2LPT | COLA (20 steps) |
|---|---|---|---|
| 0.21 | 1.22 | 0.88 | 1.21 |
| 0.29 | 1.46 | 0.86 | 1.42 |
| 0.38 | 1.85 | 0.79 | 1.70 |

so COLA tracks halofit to a few percent out to k ~ 0.3 h/Mpc, where 2LPT is
already ~40% low. The remaining deficit at higher k is the particle-mesh force
resolution and shrinks as `n_grid` is increased. The results converge by about
10 timesteps and are insensitive to `cola_z_init` at the ~2% level.

COLA is configured by the `cola_*` options documented in
[param_colore_cola.cfg](param_colore_cola.cfg). Two constraints are enforced:

- `r_smooth` must be negative. The Gaussian pre-smoothing of the initial field
  removes exactly the small-scale power the PM steps exist to evolve, so with
  smoothing on COLA would simply reproduce 2LPT.
- `lpt_vels` is forced to 1, so that the redshift-space distortions come from
  the actual particle velocities rather than from linear theory.

Setting `cola_n_steps = 0` together with `cola_lightcone_mode = 0` selects a
self-test path that reproduces `dens_type = 2` with `lpt_vels = 1`
bit-for-bit; it is useful for checking a build.

COLA is currently **serial/OpenMP only** and will refuse to run on more than one
MPI node: the particle migration between slabs and the grid halo exchanges are
not written yet. Gaussian skewers (`gaussian_skewers`) are only available for
`dens_type = 0`, and `output_lpt` is not supported for COLA because particles
cross the lightcone incrementally rather than all at once.

Memory is about 18% above the 2LPT requirement; run
`./CoLoRe --test-memory <param_file>` to get an estimate before a large run.


## 3 Parameter file and examples.

The behaviour of CoLoRe is mainly controlled by the input param file. The
param file is basically a set of name-value pairs. Any blank lines, and
anything beyond a #-symbol will be ignored.   We provide a [sample param
file](param_example.cfg) that includes all the input parameters
needed by CoLoRe. The comments included in this file explain the meaning
and functionality of these parameters.

We also provide an ipython [notebook](example_CoLoRe.ipynb) that demonstrates
how to generate all the different probes implemented in CoLoRe. The notebook
also exemplifies how to interpret the different outputs.


## 4 License

CoLoRe is distributed under the GPL license (see COPYING in the root
directory). We kindly ask you to cite the companion paper
[Ramirez-Perez et al. 2021](TBD) when using the code.


## 5 Contact

Regarding bugs, suggestions, questions or petitions, feel free to contact
the authors:
    David Alonso: david.alonso@physics.ox.ac.uk
    Cesar Ramirez: cramirez@ifae.es
