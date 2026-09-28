///////////////////////////////////////////////////////////////////////
//                                                                   //
//   COLA (COmoving Lagrangian Acceleration) density field for CoLoRe //
//                                                                   //
// This file is part of CoLoRe.                                      //
//                                                                   //
// CoLoRe is free software: you can redistribute it and/or modify it //
// under the terms of the GNU General Public License as published by //
// the Free Software Foundation, either version 3 of the License, or //
// (at your option) any later version.                               //
//                                                                   //
// CoLoRe is distributed in the hope that it will be useful, but     //
// WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU //
// General Public License for more details.                          //
//                                                                   //
// You should have received a copy of the GNU General Public License //
// along with CoLoRe.  If not, see <http://www.gnu.org/licenses/>.   //
//                                                                   //
///////////////////////////////////////////////////////////////////////
//
// The COLA method (Tassev, Zaldarriaga & Eisenstein 2013, JCAP 06:036)
// evolves particles with a particle-mesh force solver in the frame that
// comoves with their 2LPT trajectory. Large scales therefore stay exact by
// construction, and a handful of timesteps is enough to recover the
// non-linear small scales. The reference parallel C implementation is
// L-PICOLA (Howlett & Manera 2015), which is also GPLv3.
//
// Notation and units (all consistent with the rest of CoLoRe):
//   - lengths in Mpc/h; csm_hubble returns H(a)/c in h/Mpc, so
//     par->hubble_0 = H0/c and par->prefac_lensing = (3/2)*Omega_M*H0^2.
//   - the code momentum is  P = a^2 dx/dt_code = a^3 H(a) dx/da  (dimensionless),
//     so the velocity written into grid_velx/y/z is v = P/a^2, which is
//     (1+z)*v_peculiar/c -- exactly the convention that srcs.c and beaming.c
//     consume when lpt_vels is set.
//   - the PM "force field" is  Phi_a(k) = i k_a delta_k / k^2  [Mpc/h], i.e. the
//     same kernel lpt_2 uses for Psi1, so that in linear theory Phi = D1*Psi1.
//     With it the equation of motion is
//         dx/da    = P / (a^3 H)
//         dP/da    = (3/2) Omega_M H0^2 Phi / (a^2 H)
//   - splitting x = x_LPT + x_res with x_LPT = q + D1(a) Psi1 + D2(a) Psi2 and
//     storing the residual P_res = P - [G1(a) Psi1 + G2(a) Psi2], where
//     G_i(a) = a^3 H dD_i/da = a^2 H(a) D_i(a) f_i(a), gives the COLA equation
//         dP_res/da = (3/2) Omega_M H0^2 [ Phi - D1 Psi1 - (D2 - D1^2) Psi2 ] / (a^2 H)
//     (the (D2 - D1^2) bracket is L-PICOLA's q2 term rewritten for CoLoRe's
//     standard, negative D2 convention).
//
// Discretisation. The drift is exact in its LPT part, because G_i(a) is known
// analytically and integrates to D_i(a_f) - D_i(a_i). The kick is not: the PM
// force is only available at one time a_c per step, so Phi is frozen there --
// and the LPT terms subtracted from it MUST be frozen at the same a_c, or the
// two stop cancelling and every step injects a spurious error proportional to
// Psi1, i.e. straight into the large scales COLA is supposed to keep exact.
// Using the exact increment G_i(a_hi) - G_i(a_lo) instead looks more accurate
// but is wrong for precisely this reason; it shows up as P(k) losing large-scale
// power as cola_z_init is pushed earlier (45% at z_init = 49 with 20 steps).
// So, following L-PICOLA:
//     dP_res = (3/2) Omega_M H0^2 * K(a_lo,a_hi)
//              * [ Phi(a_c) - D1(a_c) Psi1 - (D2(a_c) - D1(a_c)^2) Psi2 ]
// which cancels identically at first order for any step size.
//
// Lightcone handling: CoLoRe's box is the lightcone volume with the observer at
// its centre, so every particle is observed at its own cosmic time. We run one
// global time loop over the whole box and, during each drift, deposit each
// particle into the density grid at the instant it crosses its own lightcone
// radius r_lc(a). Particles are never removed after crossing -- they keep
// contributing to the PM force, they are just not deposited twice.
///////////////////////////////////////////////////////////////////////
#include "common.h"
#include "cosmo_mad.h"

//Number of entries in the a-indexed COLA background tables
#define COLA_NA 4001

//////
// a-indexed background tables.
//
// get_bg() in cosmo.c is indexed by comoving distance, which would mean
// double-interpolating here, and -- more importantly -- its growth_f2_arr is
// 2*Omega_m^(4/7), which is NOT the logarithmic derivative of the D2 that
// cosmo.c:727 actually tabulates (they differ by ~3% at z=0). That is harmless
// for 2LPT RSDs, but for COLA it would mean G2 != a^3 H dD2/da, so the kick's
// dG2 and the drift's dD2 would no longer telescope and every step would inject
// spurious residual momentum. So COLA keeps its own tables, with f2 obtained by
// differentiating the tabulated D2.
struct ColaBg {
  int na;
  double a0,af,da,ida;
  double *a;
  double *h;   //H(a) in h/Mpc
  double *d1;  //D1(a), normalised to 1 at a=1
  double *f1;  //dlnD1/dlna
  double *d2;  //D2(a) (negative, same convention as cosmo.c:727)
  double *f2;  //dln|D2|/dlna, obtained by differentiating d2
  double *g1;  //G1 = a^2 H D1 f1
  double *g2;  //G2 = a^2 H D2 f2
  double *rlc; //comoving distance to a, in Mpc/h
  double *i2;  //cumulative integral of da/(a^2 H)
  double *i3;  //cumulative integral of da/(a^3 H)
  double *in;  //cumulative integral of a^(nlpt-3)/H da (only if nlpt != 0)
};

//////
// Persistent particle state. One particle per Lagrangian grid cell.
typedef struct {
  unsigned long long np;       //particles held by this node
  unsigned long long np_alloc; //allocated slots per array
  flouble *x[3];               //Eulerian position [Mpc/h], absolute box coordinates
  flouble *p[3];               //residual code momentum P_res
  flouble *s1[3];              //Psi1 (Lagrangian, travels with the particle)
  flouble *s2[3];              //Psi2
  unsigned char *done;         //0 = not yet deposited, 1 = deposited, 2 = crossing now
  dftw_complex *dk;            //PM density: real (padded) on input, delta_k after the r2c
  dftw_complex *scr;           //PM scratch: Phi_a(k), then the real force grid in place
  flouble *pmd;                //= (flouble *)dk,  the padded PM density grid
  flouble *pmf;                //= (flouble *)scr, the padded PM force grid
} ColaState;

//All of CoLoRe's FFTs are in-place: the real array is the complex array
//reinterpreted, with the x dimension padded to 2*(n_grid/2+1). Calling
//fftw_wrap_r2c out-of-place would make FFTW expect a *dense* n_grid^3 real
//array instead, silently reading the wrong memory. So the two PM transforms
//below are both in-place, which is why the density needs its own complex
//buffer rather than sharing the scratch one.

//////
// Small helpers

static dftw_complex *cola_alloc_complex(ptrdiff_t dsize)
{
  dftw_complex *p;
#ifdef _SPREC
  p=fftwf_alloc_complex(dsize);
#else //_SPREC
  p=fftw_alloc_complex(dsize);
#endif //_SPREC
  if(p==NULL)
    report_error(1,"COLA ran out of memory allocating a %ld-element FFT grid\n",(long)dsize);
  return p;
}

static void cola_free_complex(dftw_complex *p)
{
  if(p==NULL) return;
#ifdef _SPREC
  fftwf_free(p);
#else //_SPREC
  fftw_free(p);
#endif //_SPREC
}

//Linear interpolation in a uniform-in-a table
static double cbg(struct ColaBg *bg,double a,double *tab)
{
  double u=(a-bg->a0)*bg->ida;
  int i=(int)u;
  if(i<0) i=0;
  if(i>bg->na-2) i=bg->na-2;
  u-=i;
  return tab[i]*(1-u)+tab[i+1]*u;
}

//////
// Background tables

typedef struct {
  Csm_params *cp;
  double nlpt;
} ColaIntPar;

static double integ_i2(double a,void *vp)
{
  ColaIntPar *ip=(ColaIntPar *)vp;
  return 1./(a*a*csm_hubble(ip->cp,a));
}

static double integ_i3(double a,void *vp)
{
  ColaIntPar *ip=(ColaIntPar *)vp;
  return 1./(a*a*a*csm_hubble(ip->cp,a));
}

static double integ_in(double a,void *vp)
{
  ColaIntPar *ip=(ColaIntPar *)vp;
  return pow(a,ip->nlpt-3)/csm_hubble(ip->cp,a);
}

//Cumulative integral of f over the table nodes, starting at 0
static void cola_cumint(struct ColaBg *bg,double *out,
			double (*f)(double,void *),ColaIntPar *ip)
{
  int ii;
  double result,error;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
  F.function=f;
  F.params=ip;
  out[0]=0;
  for(ii=1;ii<bg->na;ii++) {
    gsl_integration_qag(&F,bg->a[ii-1],bg->a[ii],0,1E-10,1000,6,w,&result,&error);
    out[ii]=out[ii-1]+result;
  }
  gsl_integration_workspace_free(w);
}

static void cola_bg_init(ParamCoLoRe *par)
{
  int ii;
  struct ColaBg *bg=my_malloc(sizeof(struct ColaBg));
  Csm_params *cp=csm_params_new();
  ColaIntPar ip;
  double a_init=1./(1+par->cola_z_init);
  double growth0;

  csm_unset_gsl_eh();
  csm_background_set(cp,par->OmegaM,par->OmegaL,par->OmegaB,par->weos,0,par->hhub,2.275);

  bg->na=COLA_NA;
  //Extend below a_init so that interpolation near the first step is safe
  bg->a0=0.5*a_init;
  bg->af=1.0;
  bg->da=(bg->af-bg->a0)/(bg->na-1);
  bg->ida=1./bg->da;
  bg->a  =my_malloc(bg->na*sizeof(double));
  bg->h  =my_malloc(bg->na*sizeof(double));
  bg->d1 =my_malloc(bg->na*sizeof(double));
  bg->f1 =my_malloc(bg->na*sizeof(double));
  bg->d2 =my_malloc(bg->na*sizeof(double));
  bg->f2 =my_malloc(bg->na*sizeof(double));
  bg->g1 =my_malloc(bg->na*sizeof(double));
  bg->g2 =my_malloc(bg->na*sizeof(double));
  bg->rlc=my_malloc(bg->na*sizeof(double));
  bg->i2 =my_malloc(bg->na*sizeof(double));
  bg->i3 =my_malloc(bg->na*sizeof(double));
  bg->in =NULL;

  growth0=csm_growth_factor(cp,1.);
  for(ii=0;ii<bg->na;ii++) {
    double a=bg->a0+ii*bg->da;
    double om=csm_omega_m(cp,a);
    bg->a[ii]=a;
    bg->h[ii]=csm_hubble(cp,a);
    bg->d1[ii]=csm_growth_factor(cp,a)/growth0;
    bg->f1[ii]=csm_f_growth(cp,a);
    //Same fitting formula as cosmo.c:727, so that COLA's D2 and the 2LPT path's
    //D2 agree exactly
    bg->d2[ii]=-0.42857142857*bg->d1[ii]*bg->d1[ii]*pow(om,-0.00699300699);
    bg->rlc[ii]=csm_radial_comoving_distance(cp,a);
  }

  //f2 = dln|D2|/dlna by central differences on the tabulated D2. This is the
  //derivative of the D2 we actually use, which is what COLA needs (see the
  //comment on struct ColaBg above).
  for(ii=0;ii<bg->na;ii++) {
    int im=(ii>0 ? ii-1 : 0);
    int ipl=(ii<bg->na-1 ? ii+1 : bg->na-1);
    double dlna=log(bg->a[ipl])-log(bg->a[im]);
    bg->f2[ii]=(log(fabs(bg->d2[ipl]))-log(fabs(bg->d2[im])))/dlna;
  }

  for(ii=0;ii<bg->na;ii++) {
    double a=bg->a[ii];
    bg->g1[ii]=a*a*bg->h[ii]*bg->d1[ii]*bg->f1[ii];
    bg->g2[ii]=a*a*bg->h[ii]*bg->d2[ii]*bg->f2[ii];
  }

  ip.cp=cp;
  ip.nlpt=par->cola_nlpt;
  cola_cumint(bg,bg->i2,&integ_i2,&ip);
  cola_cumint(bg,bg->i3,&integ_i3,&ip);
  if(par->cola_nlpt!=0) {
    bg->in=my_malloc(bg->na*sizeof(double));
    cola_cumint(bg,bg->in,&integ_in,&ip);
  }

  csm_params_free(cp);
  par->cola_bg=bg;
}

void cola_free(ParamCoLoRe *par)
{
  struct ColaBg *bg=par->cola_bg;
  if(bg==NULL) return;
  free(bg->a); free(bg->h); free(bg->d1); free(bg->f1); free(bg->d2);
  free(bg->f2); free(bg->g1); free(bg->g2); free(bg->rlc);
  free(bg->i2); free(bg->i3);
  if(bg->in!=NULL) free(bg->in);
  free(bg);
  par->cola_bg=NULL;
}

//////
// Kick and drift factors.
//
// Standard (nlpt == 0):
//   K(alo,ahi)  = int_alo^ahi da/(a^2 H)
//   D(ai,af)    = int_ai^af  da/(a^3 H)
// Tassev variant (nlpt != 0), derived from the ansatz P_res(a) ~ (a/ac)^nlpt:
//   K_n = ac^(1-n) (ahi^n - alo^n) / (n ac^2 H(ac))
//   D_n = ac^(-n) int_ai^af a^(n-3)/H da
// Both reduce to the standard factors as the step shrinks.

static double cola_kick_factor(ParamCoLoRe *par,double alo,double ahi,double ac)
{
  struct ColaBg *bg=par->cola_bg;
  double n=par->cola_nlpt;
  if(n==0)
    return cbg(bg,ahi,bg->i2)-cbg(bg,alo,bg->i2);
  return pow(ac,1-n)*(pow(ahi,n)-pow(alo,n))/(n*ac*ac*cbg(bg,ac,bg->h));
}

static double cola_drift_factor(ParamCoLoRe *par,double ai,double af,double ac)
{
  struct ColaBg *bg=par->cola_bg;
  double n=par->cola_nlpt;
  if(n==0)
    return cbg(bg,af,bg->i3)-cbg(bg,ai,bg->i3);
  return pow(ac,-n)*(cbg(bg,af,bg->in)-cbg(bg,ai,bg->in));
}

//////
// Psi1 and Psi2.
//
// The k-space kernels below are a deliberate, verbatim copy of lpt_2's
// (src/density.c:961-1105). Refactoring lpt_2 to share them would risk changing
// its floating-point operation order, and the 2LPT path has to stay
// bit-for-bit identical. TODO: unify once COLA is validated.

static void cola_compute_psi(ParamCoLoRe *par,ColaState *st)
{
  int axis;
  ptrdiff_t dsize=par->nz_here*((long)(par->n_grid*(par->n_grid/2+1)));
  dftw_complex *(cdisp[3]),*(cdigrad[6]);
  flouble *(disp[3]),*(digrad[6]);
  int ngx=2*(par->n_grid/2+1);

  print_info(" - Transforming density field\n");
  //cdisp[2] aliases the density grid, exactly as lpt_2 does: delta_k is no
  //longer needed once the displacement kernels have been built.
  for(axis=0;axis<2;axis++) {
    cdisp[axis]=cola_alloc_complex(dsize);
    disp[axis]=(flouble *)cdisp[axis];
  }
  cdisp[2]=par->grid_dens_f;
  disp[2]=par->grid_dens;
  if(par->cola_use_2lpt) {
    for(axis=0;axis<6;axis++) {
      cdigrad[axis]=cola_alloc_complex(dsize);
      digrad[axis]=(flouble *)cdigrad[axis];
    }
  }
  else {
    for(axis=0;axis<6;axis++) {
      cdigrad[axis]=NULL;
      digrad[axis]=NULL;
    }
  }

  fftw_wrap_r2c(par->n_grid,par->grid_dens,par->grid_dens_f);

  print_info(" - Computing displacement field\n");
#ifdef _HAVE_OMP
#pragma omp parallel default(none)		\
  shared(par,cdisp,cdigrad)
#endif //_HAVE_OMP
  {
    int ii;
    double dk=2*M_PI/par->l_box;
    double kv[3];
    double fftnorm=(double)(par->n_grid*((long)(par->n_grid*par->n_grid)));

#ifdef _HAVE_OMP
#pragma omp for
#endif //_HAVE_OMP
    for(ii=0;ii<par->nz_here;ii++) {
      int jj,ii_true;
      ii_true=par->iz0_here+ii;
      if(2*ii_true<=par->n_grid)
	kv[2]=ii_true*dk;
      else
	kv[2]=-(par->n_grid-ii_true)*dk;
      for(jj=0;jj<par->n_grid;jj++) {
	int kk;
	if(2*jj<=par->n_grid)
	  kv[1]=jj*dk;
	else
	  kv[1]=-(par->n_grid-jj)*dk;
	for(kk=0;kk<=par->n_grid/2;kk++) {
	  int ax;
	  double k_mod2;
	  long index=kk+(par->n_grid/2+1)*((long)(jj+par->n_grid*ii));
	  if(2*kk<=par->n_grid)
	    kv[0]=kk*dk;
	  else
	    kv[0]=-(par->n_grid-kk)*dk;

	  k_mod2=fftnorm*(kv[0]*kv[0]+kv[1]*kv[1]+kv[2]*kv[2]);

	  for(ax=0;ax<3;ax++) {
	    if(k_mod2<=0)
	      cdisp[ax][index]=0;
	    else
	      cdisp[ax][index]=I*kv[ax]*par->grid_dens_f[index]/k_mod2;
	  }

	  if(par->cola_use_2lpt) {
	    cdigrad[0][index]=I*kv[0]*cdisp[0][index]; //SIGN
	    cdigrad[1][index]=I*kv[1]*cdisp[0][index];
	    cdigrad[2][index]=I*kv[2]*cdisp[0][index];
	    cdigrad[3][index]=I*kv[1]*cdisp[1][index];
	    cdigrad[4][index]=I*kv[2]*cdisp[1][index];
	    cdigrad[5][index]=I*kv[2]*cdisp[2][index];
	  }
	}
      }
    } //end omp for
  } //end omp parallel

  if(par->cola_use_2lpt) {
    print_info(" - Transform digradient\n");
    for(axis=0;axis<6;axis++)
      fftw_wrap_c2r(par->n_grid,cdigrad[axis],digrad[axis]);

    print_info(" - Computing second-order potential\n");
#ifdef _HAVE_OMP
#pragma omp parallel default(none)		\
  shared(par,digrad)
#endif //_HAVE_OMP
    {
      int iz;
      int ngxl=2*(par->n_grid/2+1);

#ifdef _HAVE_OMP
#pragma omp for
#endif //_HAVE_OMP
      for(iz=0;iz<par->nz_here;iz++) {
	int iy;
	long indexz=iz*((long)(ngxl*par->n_grid));
	for(iy=0;iy<par->n_grid;iy++) {
	  int ix;
	  long indexy=iy*ngxl;
	  for(ix=0;ix<par->n_grid;ix++) {
	    long index=ix+indexy+indexz; //SIGN
	    digrad[5][index]=
	      digrad[0][index]*digrad[3][index]+//xx*yy
	      digrad[0][index]*digrad[5][index]+//xx*zz
	      digrad[3][index]*digrad[5][index]-//+yy*zz
	      digrad[1][index]*digrad[1][index]-//-xy*xy
	      digrad[2][index]*digrad[2][index]-//-xz*xz
	      digrad[4][index]*digrad[4][index];//-yz*yz
	  }
	}
      } //end omp for
    } //end omp parallel

    print_info(" - Transforming second-order potential\n");
    fftw_wrap_r2c(par->n_grid,digrad[5],cdigrad[5]);

    print_info(" - Computing 2nd-order displacement field\n");
#ifdef _HAVE_OMP
#pragma omp parallel default(none)		\
  shared(par,cdigrad)
#endif //_HAVE_OMP
    {
      int ii;
      double dk=2*M_PI/par->l_box;
      double kv[3];
      double fftnorm=(double)(par->n_grid*((long)(par->n_grid*par->n_grid)));

#ifdef _HAVE_OMP
#pragma omp for
#endif //_HAVE_OMP
      for(ii=0;ii<par->nz_here;ii++) {
	int jj,ii_true;
	ii_true=par->iz0_here+ii;
	if(2*ii_true<=par->n_grid)
	  kv[2]=ii_true*dk;
	else
	  kv[2]=-(par->n_grid-ii_true)*dk;
	for(jj=0;jj<par->n_grid;jj++) {
	  int kk;
	  if(2*jj<=par->n_grid)
	    kv[1]=jj*dk;
	  else
	    kv[1]=-(par->n_grid-jj)*dk;
	  for(kk=0;kk<=par->n_grid/2;kk++) {
	    int ax;
	    double k_mod2;
	    long index=kk+(par->n_grid/2+1)*((long)(jj+par->n_grid*ii));
	    if(2*kk<=par->n_grid)
	      kv[0]=kk*dk;
	    else
	      kv[0]=-(par->n_grid-kk)*dk;

	    k_mod2=fftnorm*(kv[0]*kv[0]+kv[1]*kv[1]+kv[2]*kv[2]);

	    for(ax=0;ax<3;ax++) {
	      if(k_mod2<=0)
		cdigrad[ax][index]=0;
	      else
		cdigrad[ax][index]=-I*kv[ax]*cdigrad[5][index]/k_mod2; //SIGN, 1/k^2, normalization
	    }
	  }
	}
      } //end omp for
    } //end omp parallel
  }

  print_info(" - Transform 1st- and 2nd-order displacement fields\n");
  for(axis=0;axis<3;axis++) {
    fftw_wrap_c2r(par->n_grid,cdisp[axis],disp[axis]);
    if(par->cola_use_2lpt)
      fftw_wrap_c2r(par->n_grid,cdigrad[axis],digrad[axis]);
  }

  //Copy from the padded FFT layout into the dense per-particle arrays
  print_info(" - Undoing padding\n");
#ifdef _HAVE_OMP
#pragma omp parallel default(none)		\
  shared(par,st,disp,digrad,ngx)
#endif //_HAVE_OMP
  {
    long iz;
#ifdef _HAVE_OMP
#pragma omp for schedule(static)
#endif //_HAVE_OMP
    for(iz=0;iz<par->nz_here;iz++) {
      int iy;
      long indexz=iz*((long)(ngx*par->n_grid));
      long indexz_u=iz*((long)(par->n_grid*par->n_grid));
      for(iy=0;iy<par->n_grid;iy++) {
	int ix;
	long indexy=iy*ngx;
	long indexy_u=iy*par->n_grid;
	for(ix=0;ix<par->n_grid;ix++) {
	  int ax;
	  long index=ix+indexy+indexz;
	  long index_u=ix+indexy_u+indexz_u;
	  for(ax=0;ax<3;ax++) {
	    st->s1[ax][index_u]=disp[ax][index];
	    st->s2[ax][index_u]=(par->cola_use_2lpt ? digrad[ax][index] : 0);
	  }
	}
      }
    } //end omp for
  } //end omp parallel

  for(axis=0;axis<2;axis++)
    cola_free_complex(cdisp[axis]);
  if(par->cola_use_2lpt) {
    for(axis=0;axis<6;axis++)
      cola_free_complex(cdigrad[axis]);
  }
}

//////
// Allocation

static void cola_alloc(ParamCoLoRe *par,ColaState *st)
{
  int ax;
  ptrdiff_t dsize=par->nz_here*((long)(par->n_grid*(par->n_grid/2+1)));
  unsigned long long np=(unsigned long long)(par->nz_here)*
    ((unsigned long long)(par->n_grid))*((unsigned long long)(par->n_grid));
#ifdef _HAVE_MPI
  //Room for particles migrating in from neighbouring slabs
  unsigned long long np_alloc=(unsigned long long)(np*(1+par->lpt_buffer_fraction));
#else //_HAVE_MPI
  unsigned long long np_alloc=np;
#endif //_HAVE_MPI

  st->np=np;
  st->np_alloc=np_alloc;
  for(ax=0;ax<3;ax++) {
    st->x[ax] =my_malloc(np_alloc*sizeof(flouble));
    st->p[ax] =my_calloc(np_alloc,sizeof(flouble));
    st->s1[ax]=my_malloc(np_alloc*sizeof(flouble));
    st->s2[ax]=my_malloc(np_alloc*sizeof(flouble));
  }
  st->done=my_calloc(np_alloc,sizeof(unsigned char));
  st->dk=NULL;
  st->scr=NULL;
  st->pmd=NULL;
  st->pmf=NULL;
  if(par->cola_n_steps>0) {
    st->dk =cola_alloc_complex(dsize);
    st->scr=cola_alloc_complex(dsize);
    st->pmd=(flouble *)(st->dk);
    st->pmf=(flouble *)(st->scr);
  }
}

static void cola_state_free(ColaState *st)
{
  int ax;
  for(ax=0;ax<3;ax++) {
    free(st->x[ax]); free(st->p[ax]); free(st->s1[ax]); free(st->s2[ax]);
  }
  free(st->done);
  cola_free_complex(st->dk);
  cola_free_complex(st->scr);
}

//////
// CIC deposit of a single particle into the density and momentum grids.
// This is pos_2_dens_2lpt's body (src/density.c:190-262) for one particle.

static void cola_deposit_one(ParamCoLoRe *par,flouble *xc,flouble *vc)
{
  int ax,i0[3],i1[3];
  flouble a0[3],a1[3];
  flouble i_agrid=par->n_grid/par->l_box;
  long ngx=2*(par->n_grid/2+1);

  for(ax=0;ax<3;ax++) {
    i0[ax]=(int)(xc[ax]*i_agrid);
    a1[ax]=xc[ax]*i_agrid-i0[ax];
    a0[ax]=1-a1[ax];
    i1[ax]=i0[ax]+1;
    if(i0[ax]<0) i0[ax]+=par->n_grid;
    if(i1[ax]<0) i1[ax]+=par->n_grid;
    if(i0[ax]>=par->n_grid) i0[ax]-=par->n_grid;
    if(i1[ax]>=par->n_grid) i1[ax]-=par->n_grid;
  }
  i0[2]-=par->iz0_here;
  i1[2]-=par->iz0_here;

#define COLA_DEP(IX,IY,IZ,W)						\
  {									\
    long _id=(IX)+ngx*((IY)+par->n_grid*(IZ));				\
    par->grid_dens[_id]+=(W);						\
    par->grid_velx[_id]+=(W)*vc[0];					\
    par->grid_vely[_id]+=(W)*vc[1];					\
    par->grid_velz[_id]+=(W)*vc[2];					\
  }

  if((i0[2]>=0) && (i0[2]<par->nz_here)) {
    COLA_DEP(i0[0],i0[1],i0[2],a0[0]*a0[1]*a0[2]);
    COLA_DEP(i1[0],i0[1],i0[2],a1[0]*a0[1]*a0[2]);
    COLA_DEP(i0[0],i1[1],i0[2],a0[0]*a1[1]*a0[2]);
    COLA_DEP(i1[0],i1[1],i0[2],a1[0]*a1[1]*a0[2]);
  }
  if((i1[2]>=0) && (i1[2]<par->nz_here)) {
    COLA_DEP(i0[0],i0[1],i1[2],a0[0]*a0[1]*a1[2]);
    COLA_DEP(i1[0],i0[1],i1[2],a1[0]*a0[1]*a1[2]);
    COLA_DEP(i0[0],i1[1],i1[2],a0[0]*a1[1]*a1[2]);
    COLA_DEP(i1[0],i1[1],i1[2],a1[0]*a1[1]*a1[2]);
  }
#undef COLA_DEP
}

//////
// CIC deposit of all particles into the PM density grid (weight 1 each).
// This is pos_2_cic (src/density.c:64-105) reading from the ColaState arrays.

static void cola_pm_deposit(ParamCoLoRe *par,ColaState *st)
{
  unsigned long long ii;
  flouble i_agrid=par->n_grid/par->l_box;
  long ngx=2*(par->n_grid/2+1);
  long ntot=par->nz_here*((long)(par->n_grid*ngx));

  for(ii=0;ii<ntot;ii++)
    st->pmd[ii]=0;

  //Scatter, so kept serial for now. TODO (performance): counting-sort the
  //particles by i0[2] and colour the z-groups so this can be threaded.
  for(ii=0;ii<st->np;ii++) {
    int ax,i0[3],i1[3];
    flouble a0[3],a1[3];
    for(ax=0;ax<3;ax++) {
      i0[ax]=(int)(st->x[ax][ii]*i_agrid);
      a1[ax]=st->x[ax][ii]*i_agrid-i0[ax];
      a0[ax]=1-a1[ax];
      i1[ax]=i0[ax]+1;
      if(i0[ax]<0) i0[ax]+=par->n_grid;
      if(i1[ax]<0) i1[ax]+=par->n_grid;
      if(i0[ax]>=par->n_grid) i0[ax]-=par->n_grid;
      if(i1[ax]>=par->n_grid) i1[ax]-=par->n_grid;
    }
    i0[2]-=par->iz0_here;
    i1[2]-=par->iz0_here;

    if((i0[2]>=0) && (i0[2]<par->nz_here)) {
      st->pmd[i0[0]+ngx*(i0[1]+par->n_grid*i0[2])]+=a0[0]*a0[1]*a0[2];
      st->pmd[i1[0]+ngx*(i0[1]+par->n_grid*i0[2])]+=a1[0]*a0[1]*a0[2];
      st->pmd[i0[0]+ngx*(i1[1]+par->n_grid*i0[2])]+=a0[0]*a1[1]*a0[2];
      st->pmd[i1[0]+ngx*(i1[1]+par->n_grid*i0[2])]+=a1[0]*a1[1]*a0[2];
    }
    if((i1[2]>=0) && (i1[2]<par->nz_here)) {
      st->pmd[i0[0]+ngx*(i0[1]+par->n_grid*i1[2])]+=a0[0]*a0[1]*a1[2];
      st->pmd[i1[0]+ngx*(i0[1]+par->n_grid*i1[2])]+=a1[0]*a0[1]*a1[2];
      st->pmd[i0[0]+ngx*(i1[1]+par->n_grid*i1[2])]+=a0[0]*a1[1]*a1[2];
      st->pmd[i1[0]+ngx*(i1[1]+par->n_grid*i1[2])]+=a1[0]*a1[1]*a1[2];
    }
  }

  //rho -> delta. There is exactly one particle per cell, so the mean density is
  //1. Only the real cells are touched: the FFT padding must stay at zero, or it
  //would inject a spurious comb into delta_k.
  {
    long iz;
#ifdef _HAVE_OMP
#pragma omp parallel for default(none) shared(par,st,ngx) schedule(static)
#endif //_HAVE_OMP
    for(iz=0;iz<par->nz_here;iz++) {
      int iy;
      long indexz=iz*((long)(ngx*par->n_grid));
      for(iy=0;iy<par->n_grid;iy++) {
	int ix;
	long indexy=iy*ngx;
	for(ix=0;ix<par->n_grid;ix++)
	  st->pmd[ix+indexy+indexz]-=1.0;
      }
    }
  }
}

//////
// CIC read-back: the transpose of the deposit above. Read-only on the grid, so
// it threads trivially.

static void cola_cic_read(ParamCoLoRe *par,ColaState *st,flouble *grid,flouble *out)
{
  flouble i_agrid=par->n_grid/par->l_box;
  long ngx=2*(par->n_grid/2+1);
  unsigned long long np=st->np;

#ifdef _HAVE_OMP
#pragma omp parallel for default(none) shared(par,st,grid,out,i_agrid,ngx,np) schedule(static)
#endif //_HAVE_OMP
  for(unsigned long long ii=0;ii<np;ii++) {
    int ax,i0[3],i1[3];
    flouble a0[3],a1[3];
    for(ax=0;ax<3;ax++) {
      i0[ax]=(int)(st->x[ax][ii]*i_agrid);
      a1[ax]=st->x[ax][ii]*i_agrid-i0[ax];
      a0[ax]=1-a1[ax];
      i1[ax]=i0[ax]+1;
      if(i0[ax]<0) i0[ax]+=par->n_grid;
      if(i1[ax]<0) i1[ax]+=par->n_grid;
      if(i0[ax]>=par->n_grid) i0[ax]-=par->n_grid;
      if(i1[ax]>=par->n_grid) i1[ax]-=par->n_grid;
    }
    //Serial build: the slab is the whole box, so wrap in z as well.
    i0[2]-=par->iz0_here;
    i1[2]-=par->iz0_here;
    if(i0[2]>=par->nz_here) i0[2]-=par->nz_here;
    if(i1[2]>=par->nz_here) i1[2]-=par->nz_here;

    out[ii]=
      a0[0]*a0[1]*a0[2]*grid[i0[0]+ngx*(i0[1]+par->n_grid*i0[2])]+
      a1[0]*a0[1]*a0[2]*grid[i1[0]+ngx*(i0[1]+par->n_grid*i0[2])]+
      a0[0]*a1[1]*a0[2]*grid[i0[0]+ngx*(i1[1]+par->n_grid*i0[2])]+
      a1[0]*a1[1]*a0[2]*grid[i1[0]+ngx*(i1[1]+par->n_grid*i0[2])]+
      a0[0]*a0[1]*a1[2]*grid[i0[0]+ngx*(i0[1]+par->n_grid*i1[2])]+
      a1[0]*a0[1]*a1[2]*grid[i1[0]+ngx*(i0[1]+par->n_grid*i1[2])]+
      a0[0]*a1[1]*a1[2]*grid[i0[0]+ngx*(i1[1]+par->n_grid*i1[2])]+
      a1[0]*a1[1]*a1[2]*grid[i1[0]+ngx*(i1[1]+par->n_grid*i1[2])];
  }
}

//////
// One PM force evaluation plus the COLA kick.
//
// The Poisson solve is done one axis at a time and the kick applied
// immediately, so no per-particle acceleration arrays are ever needed and the
// PM working set stays at two complex grids.

static void cola_force_and_kick(ParamCoLoRe *par,ColaState *st,
				double alo,double ahi,double ac,flouble *phi,
				int diagnose)
{
  int ax;
  struct ColaBg *bg=par->cola_bg;
  double kfac=par->prefac_lensing*cola_kick_factor(par,alo,ahi,ac);
  double d1c=cbg(bg,ac,bg->d1);
  double d2c=cbg(bg,ac,bg->d2);
  //LPT terms frozen at a_c, exactly as Phi is (see the discretisation note above)
  double sub1=d1c;
  double sub2=d2c-d1c*d1c;
  unsigned long long np=st->np;

  cola_pm_deposit(par,st);
  //In-place: pmd IS the real alias of dk
  fftw_wrap_r2c(par->n_grid,st->pmd,st->dk);

  for(ax=0;ax<3;ax++) {
    unsigned long long ii;
    double phimean=0;

    //Phi_ax(k) = i k_ax delta_k / k^2, optionally with the CIC window divided out
#ifdef _HAVE_OMP
#pragma omp parallel default(none) shared(par,st,ax)
#endif //_HAVE_OMP
    {
      int ii2;
      double dkk=2*M_PI/par->l_box;
      double kv[3];
      double fftnorm=(double)(par->n_grid*((long)(par->n_grid*par->n_grid)));
      int m[3];

#ifdef _HAVE_OMP
#pragma omp for
#endif //_HAVE_OMP
      for(ii2=0;ii2<par->nz_here;ii2++) {
	int jj,ii_true;
	ii_true=par->iz0_here+ii2;
	if(2*ii_true<=par->n_grid) { kv[2]=ii_true*dkk; m[2]=ii_true; }
	else { kv[2]=-(par->n_grid-ii_true)*dkk; m[2]=ii_true-par->n_grid; }
	for(jj=0;jj<par->n_grid;jj++) {
	  int kk;
	  if(2*jj<=par->n_grid) { kv[1]=jj*dkk; m[1]=jj; }
	  else { kv[1]=-(par->n_grid-jj)*dkk; m[1]=jj-par->n_grid; }
	  for(kk=0;kk<=par->n_grid/2;kk++) {
	    double k_mod2,w2=1.0;
	    long index=kk+(par->n_grid/2+1)*((long)(jj+par->n_grid*ii2));
	    if(2*kk<=par->n_grid) { kv[0]=kk*dkk; m[0]=kk; }
	    else { kv[0]=-(par->n_grid-kk)*dkk; m[0]=kk-par->n_grid; }

	    k_mod2=fftnorm*(kv[0]*kv[0]+kv[1]*kv[1]+kv[2]*kv[2]);

	    if(par->cola_deconvolve_cic) {
	      int a2;
	      double w=1.0;
	      for(a2=0;a2<3;a2++) {
		if(m[a2]!=0) {
		  double u=M_PI*m[a2]/(double)(par->n_grid);
		  w*=sin(u)/u;
		}
	      }
	      w2=w*w;
	      //Clamp so the Nyquist corner is not amplified out of control
	      if(w2<0.05) w2=0.05;
	    }

	    if(k_mod2<=0)
	      st->scr[index]=0;
	    else
	      st->scr[index]=I*kv[ax]*st->dk[index]/(k_mod2*w2);
	  }
	}
      } //end omp for
    } //end omp parallel

    //In-place: pmf IS the real alias of scr
    fftw_wrap_c2r(par->n_grid,st->scr,st->pmf);
    cola_cic_read(par,st,st->pmf,phi);

    if(diagnose) {
      //Linear-force test. In linear theory Phi = D1(a)*Psi1, so regressing the
      //PM force on D1*Psi1 must give a slope close to 1 at the starting
      //redshift. This is the sharpest available check on the sign and the
      //normalisation of the Poisson solve, and it is worth running before
      //trusting a single timestep.
      double sxy=0,syy=0,sxx=0;
      unsigned long long jj;
#ifdef _HAVE_OMP
#pragma omp parallel for default(none) \
  shared(st,phi,np,ax,d1c) reduction(+:sxy,syy,sxx) schedule(static)
#endif //_HAVE_OMP
      for(jj=0;jj<np;jj++) {
	double ref=d1c*st->s1[ax][jj];
	sxy+=ref*phi[jj];
	sxx+=ref*ref;
	syy+=phi[jj]*phi[jj];
      }
      print_info("   linear-force test, axis %d: slope = %.4lf, "
		 "rms(Phi) = %.4lE, rms(D1*Psi1) = %.4lE Mpc/h\n",
		 ax,(sxx>0 ? sxy/sxx : 0),sqrt(syy/np),sqrt(sxx/np));
    }

    if(par->cola_subtract_mean) {
      //The k=0 mode is zero analytically, but the CIC round trip leaves a small
      //residual. A net force would integrate into a global velocity offset,
      //i.e. a spurious all-sky RSD dipole, so remove it.
      double sum=0;
#ifdef _HAVE_OMP
#pragma omp parallel for default(none) shared(phi,np) reduction(+:sum) schedule(static)
#endif //_HAVE_OMP
      for(ii=0;ii<np;ii++)
	sum+=phi[ii];
#ifdef _HAVE_MPI
      MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      phimean=sum/(double)(par->n_grid*((long)(par->n_grid*par->n_grid)));
#else //_HAVE_MPI
      phimean=sum/(double)np;
#endif //_HAVE_MPI
    }

#ifdef _HAVE_OMP
#pragma omp parallel for default(none) \
  shared(st,phi,np,ax,kfac,sub1,sub2,phimean) schedule(static)
#endif //_HAVE_OMP
    for(ii=0;ii<np;ii++) {
      st->p[ax][ii]+=kfac*(phi[ii]-phimean
			   -sub1*st->s1[ax][ii]-sub2*st->s2[ax][ii]);
    }
  }
}

//////
// Drift with on-the-fly lightcone deposition.
//
// r_lc(a) shrinks as a grows while the particle radius barely changes, so each
// particle is crossed exactly once, when r_p - r_lc(a) flips sign from - to +.
// Both radii are close to linear across a step, which gives the crossing
// fraction in closed form; the position at the crossing is then evaluated
// exactly at a_cross rather than interpolated.

static void cola_drift(ParamCoLoRe *par,ColaState *st,double ai,double af,double ac,
		       unsigned long long *n_crossed)
{
  struct ColaBg *bg=par->cola_bg;
  double rlc_i=cbg(bg,ai,bg->rlc),rlc_f=cbg(bg,af,bg->rlc);
  double d1i=cbg(bg,ai,bg->d1),d2i=cbg(bg,ai,bg->d2);
  double dd1=cbg(bg,af,bg->d1)-d1i;
  double dd2=cbg(bg,af,bg->d2)-d2i;
  double dr=cola_drift_factor(par,ai,af,ac);
  unsigned long long ii,np=st->np,ncr=0;
  double lbox=par->l_box;

  //Pass A: flag the particles that cross during this step
#ifdef _HAVE_OMP
#pragma omp parallel for default(none) \
  shared(par,st,np,dr,dd1,dd2,rlc_i,rlc_f) reduction(+:ncr) schedule(static)
#endif //_HAVE_OMP
  for(ii=0;ii<np;ii++) {
    int ax;
    double r2i=0,r2f=0;
    if(st->done[ii]) continue;
    for(ax=0;ax<3;ax++) {
      double xo=st->x[ax][ii]-par->pos_obs[ax];
      double xn=xo+st->p[ax][ii]*dr+dd1*st->s1[ax][ii]+dd2*st->s2[ax][ii];
      r2i+=xo*xo;
      r2f+=xn*xn;
    }
    if((sqrt(r2i)<rlc_i) && (sqrt(r2f)>=rlc_f)) {
      st->done[ii]=2;
      ncr++;
    }
  }

  //Pass B: deposit the crossers. Serial, because the deposit is a scatter, but
  //only a thin shell crosses per step.
  for(ii=0;ii<np;ii++) {
    int ax;
    double ri,rf,rn,s,as,drs,dd1s,dd2s,g1s,g2s;
    flouble xc[3],vc[3];
    if(st->done[ii]!=2) continue;

    ri=0; rf=0;
    for(ax=0;ax<3;ax++) {
      double xo=st->x[ax][ii]-par->pos_obs[ax];
      double xn=xo+st->p[ax][ii]*dr+dd1*st->s1[ax][ii]+dd2*st->s2[ax][ii];
      ri+=xo*xo;
      rf+=xn*xn;
    }
    ri=sqrt(ri); rf=sqrt(rf);
    rn=(rf-ri)-(rlc_f-rlc_i);
    if(fabs(rn)<1E-10) s=0;
    else s=(rlc_i-ri)/rn;
    if(s<0) s=0;
    if(s>1) s=1;
    as=ai+s*(af-ai);

    drs=cola_drift_factor(par,ai,as,ac);
    dd1s=cbg(bg,as,bg->d1)-d1i;
    dd2s=cbg(bg,as,bg->d2)-d2i;
    g1s=cbg(bg,as,bg->g1);
    g2s=cbg(bg,as,bg->g2);

    for(ax=0;ax<3;ax++) {
      double xx=st->x[ax][ii]+st->p[ax][ii]*drs+dd1s*st->s1[ax][ii]+dd2s*st->s2[ax][ii];
      if(xx<0) xx+=lbox;
      if(xx>=lbox) xx-=lbox;
      xc[ax]=xx;
      vc[ax]=(st->p[ax][ii]+g1s*st->s1[ax][ii]+g2s*st->s2[ax][ii])/(as*as);
    }
    cola_deposit_one(par,xc,vc);
    st->done[ii]=1;
  }

  //Pass C: advance every particle, crossed or not. Particles are never removed
  //after crossing -- they must keep contributing to the PM force.
#ifdef _HAVE_OMP
#pragma omp parallel for default(none) \
  shared(st,np,dr,dd1,dd2,lbox) schedule(static)
#endif //_HAVE_OMP
  for(ii=0;ii<np;ii++) {
    int ax;
    for(ax=0;ax<3;ax++) {
      double xx=st->x[ax][ii]+st->p[ax][ii]*dr+dd1*st->s1[ax][ii]+dd2*st->s2[ax][ii];
      if(xx<0) xx+=lbox;
      if(xx>=lbox) xx-=lbox;
      st->x[ax][ii]=xx;
    }
  }

  *n_crossed=ncr;
}

//////
// Initial conditions at a_init: exact 2LPT, so the residual momentum vanishes.

static void cola_init_particles(ParamCoLoRe *par,ColaState *st,double a0)
{
  struct ColaBg *bg=par->cola_bg;
  double d1=cbg(bg,a0,bg->d1),d2=cbg(bg,a0,bg->d2);
  double g1=cbg(bg,a0,bg->g1),g2=cbg(bg,a0,bg->g2);
  double rlc0=cbg(bg,a0,bg->rlc);
  unsigned long long nearly=0;
  long iz;
  flouble dx=par->l_box/par->n_grid;

#ifdef _HAVE_OMP
#pragma omp parallel for default(none) \
  shared(par,st,d1,d2,dx) schedule(static)
#endif //_HAVE_OMP
  for(iz=0;iz<par->nz_here;iz++) {
    int iy;
    long indexz=iz*((long)(par->n_grid*par->n_grid));
    flouble q2=(iz+par->iz0_here+0.0)*dx;
    for(iy=0;iy<par->n_grid;iy++) {
      int ix;
      long indexy=iy*par->n_grid;
      flouble q1=(iy+0.0)*dx;
      for(ix=0;ix<par->n_grid;ix++) {
	int ax;
	long index=ix+indexy+indexz;
	flouble q[3];
	q[0]=(ix+0.0)*dx;
	q[1]=q1;
	q[2]=q2;
	for(ax=0;ax<3;ax++) {
	  flouble xx=q[ax]+d1*st->s1[ax][index]+d2*st->s2[ax][index];
	  if(xx<0) xx+=par->l_box;
	  if(xx>=par->l_box) xx-=par->l_box;
	  st->x[ax][index]=xx;
	  st->p[ax][index]=0;
	}
      }
    }
  } //end omp for

  //Particles that are already inside the lightcone shell at a_init (the box
  //corners reach sqrt(3)*r_max) would otherwise never be deposited.
  {
    unsigned long long ii;
    for(ii=0;ii<st->np;ii++) {
      int ax;
      double r2=0;
      flouble xc[3],vc[3];
      for(ax=0;ax<3;ax++) {
	double d=st->x[ax][ii]-par->pos_obs[ax];
	r2+=d*d;
      }
      if(sqrt(r2)>=rlc0) {
	for(ax=0;ax<3;ax++) {
	  xc[ax]=st->x[ax][ii];
	  vc[ax]=(st->p[ax][ii]+g1*st->s1[ax][ii]+g2*st->s2[ax][ii])/(a0*a0);
	}
	cola_deposit_one(par,xc,vc);
	st->done[ii]=1;
	nearly++;
      }
    }
  }
  print_info(" - %llu particles already inside the lightcone at z = %.3lf\n",
	     nearly,1./a0-1);
}

//////
// Pure-LPT reference path (cola_n_steps == 0, cola_lightcone_mode == 0).
//
// This reproduces lpt_2's particle loop (src/density.c:1112-1208) exactly,
// using get_bg (not the COLA tables) so the comparison is apples-to-apples.
// It is the Stage-1 self-test: with these settings dens_type=4 must match
// dens_type=2 with lpt_vels=1.

static void cola_lpt_reference(ParamCoLoRe *par,ColaState *st)
{
  long iz;
  flouble dx=par->l_box/par->n_grid;

  print_info(" - Pure-LPT reference path (Lagrangian growth factors)\n");
  for(iz=0;iz<par->nz_here;iz++) {
    int iy;
    long indexz=iz*((long)(par->n_grid*par->n_grid));
    flouble xv2=(iz+par->iz0_here+0.0)*dx-par->pos_obs[2];
    for(iy=0;iy<par->n_grid;iy++) {
      int ix;
      long indexy=iy*par->n_grid;
      flouble xv1=(iy+0.0)*dx-par->pos_obs[1];
      for(ix=0;ix<par->n_grid;ix++) {
	int ax;
	long index=ix+indexy+indexz;
	double r,dg,d2g,f1g,f2g,invhz;
	flouble xv[3],xc[3],vc[3];
	xv[0]=(ix+0.0)*dx-par->pos_obs[0];
	xv[1]=xv1;
	xv[2]=xv2;
	r=sqrt(xv[0]*xv[0]+xv[1]*xv[1]+xv[2]*xv[2]);
	dg=get_bg(par,r,BG_D1,0);
	d2g=get_bg(par,r,BG_D2,0);
	f1g=get_bg(par,r,BG_F1,0);
	f2g=get_bg(par,r,BG_F2,0);
	invhz=get_bg(par,r,BG_IH,0);
	for(ax=0;ax<3;ax++) {
	  flouble p=xv[ax]+dg*st->s1[ax][index]+d2g*st->s2[ax][index]+par->pos_obs[ax];
	  vc[ax]=(dg*f1g*st->s1[ax][index]+d2g*f2g*st->s2[ax][index])/invhz;
	  if(p<0) p+=par->l_box;
	  if(p>=par->l_box) p-=par->l_box;
	  xc[ax]=p;
	}
	cola_deposit_one(par,xc,vc);
	st->done[index]=1;
      }
    }
  }
}

//////
// Turn the accumulated grids into what the rest of CoLoRe expects:
// momentum -> velocity, then rho -> delta.

static void cola_finalize(ParamCoLoRe *par)
{
  long ii;
  long ngx=2*(par->n_grid/2+1);
  long ntot=par->nz_here*((long)(par->n_grid*ngx));
#ifdef _DEBUG
  double numtot=0;
#endif //_DEBUG

  //Momentum-weighted average, exactly as pos_2_dens_2lpt (src/density.c:263-274)
#ifdef _HAVE_OMP
#pragma omp parallel for default(none) shared(par,ntot) schedule(static)
#endif //_HAVE_OMP
  for(ii=0;ii<ntot;ii++) {
    if(par->grid_dens[ii]!=0) {
      par->grid_velx[ii]/=par->grid_dens[ii];
      par->grid_vely[ii]/=par->grid_dens[ii];
      par->grid_velz[ii]/=par->grid_dens[ii];
    }
    else {
      par->grid_velx[ii]=0;
      par->grid_vely[ii]=0;
      par->grid_velz[ii]=0;
    }
  }

  print_info(" - Normalizing density field\n");
  //One particle per cell, so the mean density is exactly 1
#ifdef _HAVE_OMP
#ifdef _DEBUG
#pragma omp parallel default(none) shared(par,numtot,ngx)
#else //_DEBUG
#pragma omp parallel default(none) shared(par,ngx)
#endif //_DEBUG
#endif //_HAVE_OMP
  {
    long iz;
#ifdef _DEBUG
    double numtot_thr=0;
#endif //_DEBUG
#ifdef _HAVE_OMP
#pragma omp for schedule(static)
#endif //_HAVE_OMP
    for(iz=0;iz<par->nz_here;iz++) {
      int iy;
      long indexz=iz*((long)(ngx*par->n_grid));
      for(iy=0;iy<par->n_grid;iy++) {
	int ix;
	long indexy=iy*ngx;
	for(ix=0;ix<par->n_grid;ix++) {
	  long index=ix+indexy+indexz;
#ifdef _DEBUG
	  numtot_thr+=par->grid_dens[index];
#endif //_DEBUG
	  par->grid_dens[index]=par->grid_dens[index]-1.;
	}
      }
    } //end omp for
#ifdef _DEBUG
#ifdef _HAVE_OMP
#pragma omp critical
#endif //_HAVE_OMP
    {
      numtot+=numtot_thr;
    }
#endif //_DEBUG
  } //end omp parallel

#ifdef _DEBUG
#ifdef _HAVE_MPI
  if(NodeThis==0)
    MPI_Reduce(MPI_IN_PLACE,&numtot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  else
    MPI_Reduce(&numtot,&numtot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
#endif //_HAVE_MPI
  print_info(" Total density : %lE\n",
	     numtot-(double)(par->n_grid*((long)(par->n_grid*par->n_grid))));
#endif //_DEBUG
}

//////
// Driver

void cola_compute_density_field(ParamCoLoRe *par)
{
  int i,ns=par->cola_n_steps;
  ColaState st;
  double a_init=1./(1+par->cola_z_init);
  double a_end=1./(1+par->z_min);
  double *aa,*ah;
  long ngx=2*(par->n_grid/2+1);
  long ntot=par->nz_here*((long)(par->n_grid*ngx));
  long ii;
  flouble *phi=NULL;
  unsigned long long nleft=0;

#ifdef _HAVE_MPI
  if(NNodes>1) {
    report_error(1,"COLA is not MPI-parallel yet: the particle migration and grid halo "
		 "exchanges are still to be written. Run on a single node (or build with "
		 "USE_MPI=no) for now.\n");
  }
#endif //_HAVE_MPI

  print_info(" COLA: %d steps, z_init = %.3lf -> z_min = %.3lf\n",
	     ns,par->cola_z_init,par->z_min);

  cola_bg_init(par);

  //The velocity grids are guaranteed to exist: io.c forces lpt_vels=1 for COLA
  if((par->grid_velx==NULL) || (par->grid_vely==NULL) || (par->grid_velz==NULL))
    report_error(1,"COLA needs the velocity grids (lpt_vels)\n");

  cola_alloc(par,&st);
  cola_compute_psi(par,&st);

  //grid_dens currently holds Psi1_z (it doubled as the FFT buffer); clear the
  //output grids before anything is deposited.
  for(ii=0;ii<ntot;ii++) {
    par->grid_dens[ii]=0;
    par->grid_velx[ii]=0;
    par->grid_vely[ii]=0;
    par->grid_velz[ii]=0;
  }

  if((ns==0) && !(par->cola_lightcone_mode)) {
    //Stage-1 self-test path: pure 2LPT at the Lagrangian radius
    cola_lpt_reference(par,&st);
    cola_finalize(par);
    cola_state_free(&st);
    return;
  }

  cola_init_particles(par,&st,a_init);

  //Step ladder
  aa=my_malloc((ns+1)*sizeof(double));
  ah=my_malloc((ns>0 ? ns : 1)*sizeof(double));
  for(i=0;i<=ns;i++) {
    if(par->cola_step_dist)
      aa[i]=a_init*exp(i*(log(a_end)-log(a_init))/(ns>0 ? ns : 1));
    else
      aa[i]=a_init+i*(a_end-a_init)/(ns>0 ? ns : 1);
  }
  for(i=0;i<ns;i++) {
    if(par->cola_step_dist)
      ah[i]=sqrt(aa[i]*aa[i+1]);
    else
      ah[i]=0.5*(aa[i]+aa[i+1]);
  }

  if(ns>0)
    phi=my_malloc(st.np_alloc*sizeof(flouble));

  for(i=0;i<ns;i++) {
    unsigned long long ncr=0;
    double alo=(i==0 ? aa[0] : ah[i-1]);
    print_info(" - Step %d/%d: a = %.4lf -> %.4lf (z = %.3lf -> %.3lf)\n",
	       i+1,ns,aa[i],aa[i+1],1./aa[i]-1,1./aa[i+1]-1);
#ifdef _DEBUG
    cola_force_and_kick(par,&st,alo,ah[i],aa[i],phi,(i==0));
#else //_DEBUG
    cola_force_and_kick(par,&st,alo,ah[i],aa[i],phi,0);
#endif //_DEBUG
    cola_drift(par,&st,aa[i],aa[i+1],ah[i],&ncr);
    print_info("   %llu particles crossed the lightcone\n",ncr);
  }

  if(ns>0) {
    //Closing half kick, so the leftover particles get the right velocity
    cola_force_and_kick(par,&st,ah[ns-1],aa[ns],aa[ns],phi,0);
    free(phi);
  }

  //Particles at radii smaller than r(z_min) are never crossed; deposit them at
  //the final time so that mass is conserved exactly.
  {
    struct ColaBg *bg=par->cola_bg;
    double aN=aa[ns];
    double g1=cbg(bg,aN,bg->g1),g2=cbg(bg,aN,bg->g2);
    unsigned long long jj;
    for(jj=0;jj<st.np;jj++) {
      int ax;
      flouble xc[3],vc[3];
      if(st.done[jj]) continue;
      for(ax=0;ax<3;ax++) {
	xc[ax]=st.x[ax][jj];
	vc[ax]=(st.p[ax][jj]+g1*st.s1[ax][jj]+g2*st.s2[ax][jj])/(aN*aN);
      }
      cola_deposit_one(par,xc,vc);
      st.done[jj]=1;
      nleft++;
    }
  }
  print_info(" - %llu particles deposited at the final time (inside r(z_min))\n",nleft);

  free(aa);
  free(ah);
  cola_finalize(par);
  cola_state_free(&st);
}
