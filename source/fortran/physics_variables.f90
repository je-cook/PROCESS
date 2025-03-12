module physics_variables
  !! author: J. Morris (UKAEA)
  !!
  !! Module containing global variables relating to the plasma physics
  !!
  !!### References
  !!
  !! -
#ifndef dp
  use, intrinsic :: iso_fortran_env, only: dp=>real64
#endif

  implicit none

  public

  integer, parameter :: n_confinement_scalings = 51
  !! number of energy confinement time scaling laws

  real(dp) :: m_beam_amu
  !! beam ion mass (amu)

  real(dp) :: m_fuel_amu
  !! average mass of fuel portion of ions (amu)

  real(dp) :: m_ions_total_amu
  !! average mass of all ions (amu)

  real(dp) :: alphaj
  !! current profile index (calculated from q_0 and q if `iprofile=1`)

  real(dp) :: alphan
  !! density profile index

  real(dp) :: alphap
  !! pressure profile index

  real(dp) :: alpha_rate_density_total
  !! Alpha particle production rate per unit volume, from plasma and beams [particles/m3/sec]

  real(dp) :: alpha_rate_density_plasma
  !! Alpha particle production rate per unit volume, just from plasma [particles/m3/sec]

  real(dp) :: alphat
  !! temperature profile index

  real(dp) :: aspect
  !! aspect ratio (`iteration variable 1`)

  real(dp) :: beamfus0
  !! multiplier for beam-background fusion calculation

  real(dp) :: beta
  !! total plasma beta (`iteration variable 5`) (calculated if stellarator)

  real(dp) :: beta_fast_alpha
  !! fast alpha beta component

  real(dp) :: beta_max
  !! Max allowable beta

  real(dp) :: beta_min
  !! allowable lower beta

  real(dp) :: beta_beam
  !! neutral beam beta component

  real(dp) :: beta_poloidal
  !! poloidal beta

  real(dp) :: beta_poloidal_eps
  !! Poloidal beta and inverse aspcet ratio product

  real(dp) :: beta_toroidal
  !! toroidal beta

  real(dp) :: beta_thermal
  !! thermal beta

  real(dp) :: beta_thermal_poloidal
  !! poloidal thermal beta

  real(dp) :: beta_thermal_toroidal
  !! poloidal thermal beta

  real(dp) :: beta_norm_total
  !! normaised total beta

  real(dp) :: beta_norm_thermal
  !! normaised thermal beta

  real(dp) :: beta_norm_toroidal
  !! normaised toroidal beta

  real(dp) :: beta_norm_poloidal
  !! normaised poloidal beta

  real(dp) :: e_plasma_beta_thermal
  !! Plasma thermal energy derived from thermal beta

  real(dp) :: betbm0
  !! leading coefficient for NB beta fraction

  real(dp) :: bp
  !! poloidal field (T)

  real(dp) :: bt
  !! toroidal field on axis (T) (`iteration variable 2`)

  real(dp) :: btot
  !! total toroidal + poloidal field (T)

  real(dp) :: burnup
  !! fractional plasma burnup

  real(dp) :: burnup_in
  !! fractional plasma burnup user input

  real(dp) :: bvert
  !! vertical field at plasma (T)

  real(dp) :: c_beta
  !! Destabalisation parameter for iprofile=6 beta limit

  real(dp) :: csawth
  !! coeff. for sawteeth effects on burn V-s requirement

  real(dp) :: f_vol_plasma
  !! multiplying factor for the plasma volume (normally=1)

  real(dp) :: f_r_conducting_wall
  !! maximum ratio of conducting wall distance to plasma minor radius for
  !! vertical stability (`constraint equation 23`)

  real(dp) :: dene
  !! electron density (/m3) (`iteration variable 6`)

  real(dp) :: nd_fuel_ions
  !! fuel ion density (/m3)

  real(dp) :: dlamee
  !! electron-electron coulomb logarithm

  real(dp) :: dlamie
  !! ion-electron coulomb logarithm

  real(dp), dimension(8) :: dlimit
  !! density limit (/m3) as calculated using various models

  real(dp) :: nd_alphas
  !! thermal alpha density (/m3)

  real(dp) :: nd_beam_ions
  !! hot beam ion density, variable (/m3)

  real(dp) :: beam_density_out
  !! hot beam ion density from calculation (/m3)

  real(dp) :: beta_norm_max
  !! Troyon-like coefficient for beta scaling

  real(dp) :: dnelimt
  !! density limit (/m3)

  real(dp) :: nd_ions_total
  !! total ion density (/m3)

  real(dp) :: dnla
  !! line averaged electron density (/m3)

  real(dp) :: nd_protons
  !! proton ash density (/m3)

  real(dp) :: ntau
  !! Fusion double product (s/m3)

  real(dp) :: nTtau
  !! Lawson triple product [keV s / m3]

  real(dp) :: nd_impurities
  !! high Z ion density (/m3)

  real(dp) :: gradient_length_ne
  !! Max. normalized gradient length in el. density (ipedestal==0 only)

  real(dp) :: gradient_length_te
  !! Max. normalized gradient length in el. temperature (ipedestal==0 only)

  real(dp) :: beta_poloidal_eps_max
  !! maximum (eps*beta_poloidal) (`constraint equation 6`). Note: revised issue #346
  !! "Operation at the tokamak equilibrium poloidal beta-limit in TFTR", 1992 Nucl. Fusion 32 1468

  real(dp) :: eps
  !! inverse aspect ratio

  real(dp) :: aux_current_fraction
  !! fraction of plasma current produced by auxiliary current drive

  real(dp) :: inductive_current_fraction
  !! fraction of plasma current produced inductively

  real(dp) :: f_alpha_electron
  !! fraction of alpha energy to electrons

  real(dp) :: f_alpha_plasma
  !! Fraction of alpha power deposited in plasma. Default of 0.95 taken from https://doi.org/10.1088/0029-5515/39/12/305.

  real(dp) :: f_alpha_ion
  !! fraction of alpha power to ions

  real(dp) :: f_deuterium
  !! deuterium fuel fraction

  real(dp) :: ftar
  !! fraction of power to the lower divertor in double null configuration
  !! (`i_single_null = 0` only) (default assumes SN)

  real(dp) :: ffwal
  !! factor to convert plasma surface area to first wall area in neutron wall
  !! load calculation (`iwalld=1`)

  real(dp) :: fgwped
  !! fraction of Greenwald density to set as pedestal-top density. If `<0`, pedestal-top
  !! density set manually using neped (`ipedestal==1`).
  !! (`iteration variable 145`)

  real(dp) :: fgwsep
  !! fraction of Greenwald density to set as separatrix density. If `<0`, separatrix
  !! density set manually using nesep (`ipedestal==1`).
  !! (`iteration variable 152`)

  real(dp) :: f_helium3
  !! helium-3 fuel fraction

  real(dp) :: figmer
  !! physics figure of merit (= plasma_current*aspect**sbar, where `sbar=1`)

  real(dp) :: fkzohm
  !! Zohm elongation scaling adjustment factor (`i_plasma_geometry=2, 3`)

  real(dp) :: fplhsep
  !! F-value for Psep >= Plh + Paux (`constraint equation 73`)

  real(dp) :: fpdivlim
  !! F-value for minimum pdivt (`constraint equation 80`)

  real(dp) :: fne0
  !! f-value for the constraint ne(0) > ne(ped) (`constraint equation 81`)
  !! (`Iteration variable 154`)

  real(dp) :: f_tritium
  !! tritium fuel fraction

  real(dp) :: fusion_rate_density_total
  !! fusion reaction rate, from beams and plasma (reactions/m3/sec)

  real(dp) :: fusion_rate_density_plasma
  !! fusion reaction rate, just from plasma (reactions/m3/sec)

  real(dp) :: fvsbrnni
  !! fraction of the plasma current produced by non-inductive means (`iteration variable 44`)

  real(dp) :: ejima_coeff
  !! Ejima coefficient for resistive startup V-s formula

  real(dp) :: f_beta_alpha_beam_thermal
  !! ratio of (fast alpha + neutral beam beta) to thermal beta

  real(dp), dimension(n_confinement_scalings) :: hfac
  !! H factors for an ignited plasma for each energy confinement time scaling law

  real(dp) :: hfact
  !! H factor on energy confinement times, radiation corrected (`iteration variable 10`).

  real(dp) :: taumax
  !! Maximum allowed energy confinement time (s)

  integer :: i_bootstrap_current
  !! switch for bootstrap current scaling
  !!
  !! - =1 ITER 1989 bootstrap scaling (high R/a only)
  !! - =2 for Nevins et al general scaling
  !! - =3 for Wilson et al numerical scaling
  !! - =4 for Sauter et al scaling
  !! - =5 for Sakai et al scaling
  !! - =6 for ARIES scaling
  !! - =7 for Andrade et al scaling
  !! - =8 for Hoang et al scaling
  !! - =9 for Wong et al scaling
  !! - =10 for Gi-I et al scaling
  !! - =11 for Gi-II et al scaling

  integer :: i_beta_component
  !! switch for beta limit scaling (`constraint equation 24`)
  !!
  !! - =0 apply limit to total beta
  !! - =1 apply limit to thermal beta
  !! - =2 apply limit to thermal + neutral beam beta
  !! - =3 apply limit to toroidal beta

  integer :: i_plasma_current
  !! switch for plasma current scaling to use
  !!
  !! - =1 Peng analytic fit
  !! - =2 Peng double null divertor scaling (ST)
  !! - =3 simple ITER scaling (k = 2.2, d = 0.6)
  !! - =4 later ITER scaling, a la Uckan
  !! - =5 Todd empirical scaling I
  !! - =6 Todd empirical scaling II
  !! - =7 Connor-Hastie model
  !! - =8 Sauter scaling allowing negative triangularity
  !! - =9 FIESTA ST fit

  integer :: i_diamagnetic_current
  !! switch for diamagnetic current scaling
  !!
  !! - =0 Do not calculate
  !! - =1 Use original TART scaling
  !! - =2 Use SCENE scaling

  integer :: i_density_limit
  !! switch for density limit to enforce (`constraint equation 5`)
  !!
  !! - =1 old ASDEX
  !! - =2 Borrass model for ITER (I)
  !! - =3 Borrass model for ITER (II)
  !! - =4 JET edge radiation
  !! - =5 JET simplified
  !! - =6 Hugill-Murakami Mq limit
  !! - =7 Greenwald limit
  !! - =8 ASDEX New

  integer :: idivrt
  !! number of divertors (calculated from `i_single_null`)

  integer :: i_beta_fast_alpha
  !! switch for fast alpha pressure calculation
  !!
  !! - =0 ITER physics rules (Uckan) fit
  !! - =1 Modified fit (D. Ward) - better at high temperature

  integer :: ignite
  !! switch for ignition assumption. Obviously, ignite must be zero if current drive
  !! is required. If ignite is 1, any auxiliary power is assumed to be used only during
  !! plasma start-up, and is excluded from all steady-state power balance calculations.
  !!
  !! - =0 do not assume plasma ignition
  !! - =1 assume ignited (but include auxiliary power in costs)</UL

  integer :: ipedestal
  !! switch for pedestal profiles:
  !!
  !! - =0 use original parabolic profiles
  !! - =1 use pedestal profile

  integer :: i_pfirsch_schluter_current
  !! switch for Pfirsch-Schlüter current scaling (issue #413):
  !!
  !! - =0 Do not calculate
  !! - =1 Use SCENE scaling

  real(dp) :: neped
  !! electron density of pedestal [m-3] (`ipedestal==1)

  real(dp) :: nesep
  !! electron density at separatrix [m-3] (`ipedestal==1)

  real(dp) :: alpha_crit
  !! critical ballooning parameter value

  real(dp) :: nesep_crit
  !! critical electron density at separatrix [m-3]

  real(dp) :: plasma_res_factor
  !! plasma resistivity pre-factor

  real(dp) :: rhopedn
  !! r/a of density pedestal (`ipedestal==1`)

  real(dp) :: rhopedt
  !! r/a of temperature pedestal (`ipedestal==1`)

  real(dp) :: rho_te_max
  !! r/a where the temperature gradient is largest (`ipedestal==0`)

  real(dp) :: rho_ne_max
  !! r/a where the density gradient is largest (`ipedestal==0`)

  real(dp) :: tbeta
  !! temperature profile index beta  (`ipedestal==1)

  real(dp) :: teped
  !! electron temperature of pedestal (keV) (`ipedestal==1`)

  real(dp) :: tesep
  !! electron temperature at separatrix (keV) (`ipedestal==1`) calculated if reinke
  !! criterion is used (`icc=78`)

  integer :: iprofile
  !! switch for current profile consistency:
  !!
  !! - =0 use input values for alphaj, ind_plasma_internal_norm, beta_norm_max
  !! - =1 make these consistent with input q95, q_0 values (recommend `i_plasma_current=4` with this option)
  !! - =2 use input values for alphaj, ind_plasma_internal_norm. Scale beta_norm_max with aspect ratio (original scaling)
  !! - =3 use input values for alphaj, ind_plasma_internal_norm. Scale beta_norm_max with aspect ratio (Menard scaling)
  !! - =4 use input values for alphaj, beta_norm_max. Set ind_plasma_internal_norm from elongation (Menard scaling)
  !! - =5 use input value for alphaj.  Set ind_plasma_internal_norm and beta_norm_max from Menard scaling
  !! - =6 use input values for alphaj, c_beta.  Set ind_plasma_internal_norm from Menard and beta_norm_max from Tholerus

  integer :: i_rad_loss
  !! switch for radiation loss term usage in power balance (see User Guide):
  !!
  !! - =0 total power lost is scaling power plus radiation
  !! - =1 total power lost is scaling power plus core radiation only
  !! - =2 total power lost is scaling power only, with no additional
  !!   allowance for radiation. This is not recommended for power plant models.

  integer :: i_confinement_time
  !! switch for energy confinement time scaling law (see description in `labels_confinement_scalings`)

  !! labels_confinement_scalings(n_confinement_scalings) : labels describing energy confinement scaling laws
  character*34, parameter, dimension(n_confinement_scalings) :: labels_confinement_scalings = (/  &
    'User input electron confinement   ', &
    'Neo-Alcator                (Ohmic)', &
    'Mirnov                         (H)', &
    'Merezkhin-Muhkovatov    (Ohmic)(L)', &
    'Shimomura                      (H)', &
    'Kaye-Goldston                  (L)', &
    'ITER 89-P                      (L)', &
    'ITER 89-O                      (L)', &
    'Rebut-Lallia                   (L)', &
    'Goldston                       (L)', &
    'T10                            (L)', &
    'JAERI / Odajima-Shimomura      (L)', &
    'Kaye-Big Complex               (L)', &
    'ITER H90-P                     (H)', &
    'ITER 89-P & 89-O min           (L)', &
    'Riedel                         (L)', &
    'Christiansen                   (L)', &
    'Lackner-Gottardi               (L)', &
    'Neo-Kaye                       (L)', &
    'Riedel                         (H)', &
    'ITER H90-P amended             (H)', &
    'LHD                        (Stell)', &
    'Gyro-reduced Bohm          (Stell)', &
    'Lackner-Gottardi           (Stell)', &
    'ITER-93H  ELM-free             (H)', &
    'TITAN RFP OBSOLETE                ', &
    'ITER H-97P ELM-free            (H)', &
    'ITER H-97P ELMy                (H)', &
    'ITER-96P (ITER-97L)            (L)', &
    'Valovic modified ELMy          (H)', &
    'Kaye 98 modified               (L)', &
    'ITERH-PB98P(y)                 (H)', &
    'IPB98(y)                       (H)', &
    'IPB98(y,1)                     (H)', &
    'IPB98(y,2)                     (H)', &
    'IPB98(y,3)                     (H)', &
    'IPB98(y,4)                     (H)', &
    'ISS95                      (Stell)', &
    'ISS04                      (Stell)', &
    'DS03 beta-independent          (H)', &
    'Murari "Non-power law"         (H)', &
    'Petty 2008                 (ST)(H)', &
    'Lang high density              (H)', &
    'Hubbard 2017 - nominal         (I)', &
    'Hubbard 2017 - lower           (I)', &
    'Hubbard 2017 - upper           (I)', &
    'Menard NSTX                (ST)(H)', &
    'Menard NSTX-Petty08 hybrid (ST)(H)', &
    'Buxton NSTX gyro-Bohm      (ST)(H)', &
    'ITPA20                         (H)', &
    'ITPA20-IL                      (H)' /)

  integer :: i_plasma_wall_gap
  !! Switch for plasma-first wall clearances at the mid-plane:
  !!
  !! - =0 use 10% of plasma minor radius
  !! - =1 use input (`dr_fw_plasma_gap_inboard` and `dr_fw_plasma_gap_outboard`)

  integer :: i_plasma_geometry
  !! switch for plasma elongation and triangularity calculations:
  !!
  !! - =0 use input kappa, triang to calculate 95% values
  !! - =1 scale q95_min, kappa, triang with aspect ratio (ST)
  !! - =2 set kappa to the natural elongation value (Zohm ITER scaling), triang input
  !! - =3 set kappa to the natural elongation value (Zohm ITER scaling), triang95 input
  !! - =4 use input kappa95, triang95 to calculate separatrix values
  !! - =5 use input kappa95, triang95 to calculate separatrix values based on MAST scaling (ST)
  !! - =6 use input kappa, triang to calculate 95% values based on MAST scaling (ST)
  !! - =7 use input kappa95, triang95 to calculate separatrix values based on fit to FIESTA (ST)
  !! - =8 use input kappa, triang to calculate 95% values based on fit to FIESTA (ST)
  !! - =9 set kappa to the natural elongation value, triang input
  !! - =10 set kappa to maximum stable value at a given aspect ratio (2.6<A<3.6)), triang input (#1399)
  !! - =11 set kappa Menard 2016 aspect-ratio-dependent scaling, triang input (#1439)

  integer :: i_plasma_shape
  !! switch for plasma boundary shape:
  !!
  !! - =0 use original PROCESS 2-arcs model
  !! - =1 use the Sauter model

  integer :: itart
  !! switch for spherical tokamak (ST) models:
  !!
  !! - =0 use conventional aspect ratio models
  !! - =1 use spherical tokamak models

  integer :: itartpf
  !! switch for Spherical Tokamak PF models:
  !!
  !! - =0 use Peng and Strickler (1986) model
  !! - =1 use conventional aspect ratio model

  integer :: iwalld
  !! switch for neutron wall load calculation:
  !!
  !! - =1 use scaled plasma surface area
  !! - =2 use first wall area directly

  real(dp) :: plasma_square
  !! plasma squareness used by Sauter plasma shape

  real(dp) :: kappa
  !! plasma separatrix elongation (calculated if `i_plasma_geometry = 1-5, 7 or 9-10`)

  real(dp) :: kappa95
  !! plasma elongation at 95% surface (calculated if `i_plasma_geometry = 0-3, 6, or 8-10`)


  real(dp) :: kappa_ipb
  !! Separatrix elongation calculated for IPB scalings

  real(dp) :: ne0
  !! central electron density (/m3)

  real(dp) :: ni0
  !! central ion density (/m3)

  real(dp) :: m_s_limit
  !! margin to vertical stability

  real(dp) :: p0
  !! central total plasma pressure (Pa)

  real(dp) :: j_plasma_0
  !! Central plasma current density (A/m2)

  real(dp) :: vol_avg_pressure
  !! Volume averaged plasma pressure (Pa)

  real(dp) :: f_dd_branching_trit
  !! branching ratio for DD -> T

  real(dp) :: alpha_power_density_plasma
  !! Alpha power per volume just from plasma [MW/m3]

  real(dp) :: alpha_power_density_total
  !! Alpha power per volume from plasma and beams [MW/m3]

  real(dp) :: alpha_power_electron_density
  !! Alpha power per volume to electrons [MW/m3]

  real(dp) :: p_fw_alpha_mw
  !! alpha power escaping plasma and reaching first wall (MW)

  real(dp) :: alpha_power_ions_density
  !! alpha power per volume to ions (MW/m3)

  real(dp) :: alpha_power_plasma
  !! Alpha power from only the plasma (MW)

  real(dp) :: alpha_power_total
  !! Total alpha power from plasma and beams (MW)

  real(dp) :: alpha_power_beams
  !! alpha power from hot neutral beam ions (MW)

  real(dp) :: non_alpha_charged_power
  !! non-alpha charged particle fusion power (MW)

  real(dp) :: charged_particle_power
  !! Total charged particle fusion power [MW]

  real(dp) :: charged_power_density
  !! Non-alpha charged particle fusion power per volume [MW/m3]

  real(dp) :: pcoef
  !! profile factor (= n-weighted T / average T)

  real(dp) :: p_plasma_inner_rad_mw
  !! radiation power from inner zone (MW)

  real(dp) :: pden_plasma_core_rad_mw
  !! total core radiation power per volume (MW/m3)

  real(dp) :: dd_power
  !! deuterium-deuterium fusion power (MW)

  real(dp) :: dhe3_power
  !! deuterium-helium3 fusion power (MW)

  real(dp) :: pdivt
  !! power to conducted to the divertor region (MW)

  real(dp) :: pdivl
  !! power conducted to the lower divertor region (calculated if `i_single_null = 0`) (MW)

  real(dp) :: pdivu
  !! power conducted to the upper divertor region (calculated if `i_single_null = 0`) (MW)

  real(dp) :: pdivmax
  !! power conducted to the divertor with most load (calculated if `i_single_null = 0`) (MW)

  real(dp) :: dt_power_total
  !!  Total deuterium-tritium fusion power, from plasma and beams [MW]

  real(dp) :: dt_power_plasma
  !!  Deuterium-tritium fusion power, just from plasma [MW]

  real(dp) :: p_plasma_outer_rad_mw
  !! radiation power from outer zone (MW)

  real(dp) :: pden_plasma_outer_rad_mw
  !! edge radiation power per volume (MW/m3)

  real(dp) :: vs_plasma_internal
  !! internal plasma V-s

  real(dp) :: pflux_fw_rad_mw
  !! Nominal mean radiation load on inside surface of reactor (MW/m2)

  real(dp) :: piepv
  !! ion/electron equilibration power per volume (MW/m3)

  real(dp) :: plasma_current
  !! plasma current (A)

  real(dp) :: neutron_power_plasma
  !! Neutron fusion power from just the plasma [MW]

  real(dp) :: neutron_power_total
  !! Total neutron fusion power from plasma and beams [MW]

  real(dp) :: neutron_power_density_total
  !! neutron fusion power per volume from beams and plasma (MW/m3)

  real(dp) :: neutron_power_density_plasma
  !! neutron fusion power per volume just from plasma (MW/m3)

  real(dp) :: p_plasma_ohmic_mw
  !! ohmic heating power (MW)

  real(dp) :: pden_plasma_ohmic_mw
  !! ohmic heating power per volume (MW/m3)

  real(dp) :: p_plasma_loss_mw
  !! heating power (= transport loss power) (MW) used in confinement time calculation

  real(dp) :: fusion_power
  !! fusion power (MW)

  real(dp) :: len_plasma_poloidal
  !! plasma poloidal perimeter (m)

  real(dp) :: p_plasma_rad_mw
  !! total radiation power from inside LCFS (MW)

  real(dp) :: pden_plasma_rad_mw
  !! total radiation power per volume (MW/m3)

  real(dp) :: pradsolmw
  !! radiation power from SoL (MW)

  real(dp) :: proton_rate_density
  !! Proton production rate [particles/m3/sec]

  real(dp) :: psolradmw
  !! SOL radiation power (MW) (`stellarator only`)

  real(dp) :: pden_plasma_sync_mw
  !! synchrotron radiation power per volume (MW/m3)

  real(dp) :: p_plasma_sync_mw
  !! Total synchrotron radiation power from plasma (MW)

  integer :: i_l_h_threshold
  !! switch for L-H mode power threshold scaling to use (see l_h_threshold_powers for list)

  real(dp) :: p_l_h_threshold_mw
  !! L-H mode power threshold (MW) (chosen via i_l_h_threshold, and enforced if
  !! constraint equation 15 is on)

  real(dp), dimension(21) :: l_h_threshold_powers
  !! L-H power threshold for various scalings (MW)
  !!
  !! - =1 ITER 1996 scaling: nominal
  !! - =2 ITER 1996 scaling: upper bound
  !! - =3 ITER 1996 scaling: lower bound
  !! - =4 ITER 1997 scaling: excluding elongation
  !! - =5 ITER 1997 scaling: including elongation
  !! - =6 Martin 2008 scaling: nominal
  !! - =7 Martin 2008 scaling: 95% upper bound
  !! - =8 Martin 2008 scaling: 95% lower bound
  !! - =9 Snipes 2000 scaling: nominal
  !! - =10 Snipes 2000 scaling: upper bound
  !! - =11 Snipes 2000 scaling: lower bound
  !! - =12 Snipes 2000 scaling (closed divertor): nominal
  !! - =13 Snipes 2000 scaling (closed divertor): upper bound
  !! - =14 Snipes 2000 scaling (closed divertor): lower bound
  !! - =15 Hubbard et al. 2012 L-I threshold scaling: nominal
  !! - =16 Hubbard et al. 2012 L-I threshold scaling: lower bound
  !! - =17 Hubbard et al. 2012 L-I threshold scaling: upper bound
  !! - =18 Hubbard et al. 2017 L-I threshold scaling
  !! - =19 Martin 2008 aspect ratio corrected scaling: nominal
  !! - =20 Martin 2008 aspect ratio corrected scaling: 95% upper bound
  !! - =21 Martin 2008 aspect ratio corrected scaling: 95% lower bound

  real(dp) :: p_electron_transport_loss_mw
  !! electron transport power (MW)

  real(dp) :: pden_electron_transport_loss_mw
  !! electron transport power per volume (MW/m3)

  real(dp) :: p_ion_transport_loss_mw
  !! ion transport power (MW)

  real(dp) :: pscalingmw
  !! Total transport power from scaling law (MW)

  real(dp) :: pden_ion_transport_loss_mw
  !! ion transport power per volume (MW/m3)

  real(dp) :: q0
  !! Safety factor on axis

  real(dp) :: q95
  !! Safety factor at 95% flux surface (iteration variable 18) (unless icurr=2 (ST current scaling),
  !! in which case q95 = mean edge safety factor qbar)

  real(dp) :: qfuel
  !! plasma fuelling rate (nucleus-pairs/s)

  real(dp) :: tauratio
  !! tauratio /1.0/ : ratio of He and pellet particle confinement times

  real(dp) :: q95_min
  !! lower limit for edge safety factor

  real(dp) :: qstar
  !! cylindrical safety factor

  real(dp) :: rad_fraction_sol
  !! SoL radiation fraction

  real(dp) :: rad_fraction_total
  !! Radiation fraction total = SoL + LCFS radiation / total power deposited in plasma

  real(dp) :: f_nd_alpha_electron
  !! thermal alpha density/electron density (`iteration variable 109`)

  real(dp) :: f_nd_protium_electrons
  !! Seeded f_nd_protium_electrons density / electron density.

  real(dp) :: ind_plasma_internal_norm
  !! Plasma normalised internal inductance (calculated from alphaj if `iprofile=1`)

  real(dp) :: ind_plasma
  !! plasma inductance (H)

  real(dp) :: rmajor
  !! plasma major radius (m) (`iteration variable 3`)

  real(dp) :: rminor
  !! plasma minor radius (m)

  real(dp) :: f_nd_beam_electron
  !! hot beam density / n_e (`iteration variable 7`)

  real(dp) :: rncne
  !! n_carbon / n_e

  real(dp) :: rndfuel
  !! fuel burnup rate (reactions/second)

  real(dp) :: rnfene
  !! n_highZ / n_e

  real(dp) :: rnone
  !! n_oxygen / n_e

  real(dp) :: f_res_plasma_neo
  !! neo-classical correction factor to res_plasma

  real(dp) :: res_plasma
  !! plasma resistance (ohm)

  real(dp) :: t_plasma_res_diffusion
  !! plasma current resistive diffusion time (s)

  real(dp) :: a_plasma_surface
  !! plasma surface area

  real(dp) :: a_plasma_surface_outboard
  !! outboard plasma surface area

  integer :: i_single_null
  !! switch for single null / double null plasma:
  !!
  !! - =0 for double null
  !! - =1 for single null (diverted side down)

  real(dp) :: f_sync_reflect
  !! synchrotron wall reflectivity factor

  real(dp) :: t_electron_energy_confinement
  !! electron energy confinement time (sec)

  real(dp) :: tauee_in
  !! Input electron energy confinement time (sec) (`i_confinement_time=48 only`)

  real(dp) :: t_energy_confinement
  !! global thermal energy confinement time (sec)

  real(dp) :: t_ion_energy_confinement
  !! ion energy confinement time (sec)

  real(dp) :: t_alpha_confinement
  !! alpha particle confinement time (sec)

  real(dp) :: f_alpha_energy_confinement
  !! alpha particle to energy confinement time ratio

  real(dp) :: te
  !! volume averaged electron temperature (keV) (`iteration variable 4`)

  real(dp) :: te0
  !! central electron temperature (keV)

  real(dp) :: ten
  !! density weighted average electron temperature (keV)

  real(dp) :: ti
  !! volume averaged ion temperature (keV). N.B. calculated from te if `tratio > 0.0`

  real(dp) :: ti0
  !! central ion temperature (keV)

  real(dp) :: tin
  !! density weighted average ion temperature (keV)

  real(dp) :: tratio
  !! ion temperature / electron temperature(used to calculate ti if `tratio > 0.0`

  real(dp) :: triang
  !! plasma separatrix triangularity (calculated if `i_plasma_geometry = 1, 3-5 or 7`)

  real(dp) :: triang95
  !! plasma triangularity at 95% surface (calculated if `i_plasma_geometry = 0-2, 6, 8 or 9`)

  real(dp) :: vol_plasma
  !! plasma volume (m3)

  real(dp) :: vs_plasma_burn_required
  !! V-s needed during flat-top (heat + burn times) (Wb)

  real(dp) :: v_plasma_loop_burn
  !! Plasma loop voltage during flat-top (V)

  real(dp) :: vshift
  !! plasma/device midplane vertical shift - single null

  real(dp) :: vs_plasma_ind_ramp
  !! Total plasma inductive flux consumption for plasma current ramp-up (Vs)(Wb)

  real(dp) :: vs_plasma_res_ramp
  !! Plasma resistive flux consumption for plasma current ramp-up (Vs)(Wb)

  real(dp) :: vs_plasma_total_required
  !! total V-s needed (Wb)

  real(dp) :: pflux_fw_neutron_mw
  !! average neutron wall load (MW/m2)

  real(dp) :: wtgpd
  !! mass of fuel used per day (g)

  real(dp) :: a_plasma_poloidal
  !! plasma poloidal cross-sectional area [m^2]

  real(dp) :: zeff
  !! plasma effective charge

  real(dp) :: zeffai
  !! mass weighted plasma effective charge

  contains

  subroutine init_physics_variables
    !! Initialise module variables
    implicit none

    m_beam_amu = 0.0D0
    m_fuel_amu = 0.0D0
    m_ions_total_amu = 0.0D0
    alphaj = 1.0D0
    alphan = 0.25D0
    alphap = 0.0D0
    alpha_rate_density_total = 0.0D0
    alpha_rate_density_plasma = 0.0D0
    alphat = 0.5D0
    aspect = 2.907D0
    beamfus0 = 1.0D0
    beta = 0.042D0
    beta_fast_alpha = 0.0D0
    beta_max = 0.0D0
    beta_min = 0.0D0
    beta_beam = 0.0D0
    beta_poloidal = 0.0D0
    beta_poloidal_eps = 0.0D0
    beta_toroidal = 0.0D0
    beta_thermal = 0.0D0
    beta_thermal_poloidal = 0.0D0
    beta_thermal_toroidal = 0.0D0
    beta_norm_total = 0.0D0
    beta_norm_thermal = 0.0D0
    beta_norm_poloidal = 0.0D0
    e_plasma_beta_thermal = 0.0D0
    beta_norm_toroidal = 0.0D0
    betbm0 = 1.5D0
    bp = 0.0D0
    bt = 5.68D0
    btot = 0.0D0
    burnup = 0.0D0
    burnup_in = 0.0D0
    bvert = 0.0D0
    c_beta = 0.5D0
    csawth = 1.0D0
    f_vol_plasma = 1.0D0
    f_r_conducting_wall = 1.35D0
    dene = 9.8D19
    nd_fuel_ions = 0.0D0
    dlamee = 0.0D0
    dlamie = 0.0D0
    dlimit = 0.0D0
    nd_alphas = 0.0D0
    nd_beam_ions = 0.0D0
    beam_density_out = 0.0D0
    beta_norm_max = 3.5D0
    dnelimt = 0.0D0
    nd_ions_total = 0.0D0
    dnla = 0.0D0
    nd_protons = 0.0D0
    ntau = 0.0D0
    nTtau = 0.0D0
    nd_impurities = 0.0D0
    beta_poloidal_eps_max = 1.38D0
    eps = 0.34399724802D0
    aux_current_fraction = 0.0D0
    inductive_current_fraction = 0.0D0
    f_alpha_electron = 0.0D0
    f_alpha_plasma = 0.95D0
    f_alpha_ion = 0.0D0
    f_deuterium = 0.5D0
    ftar = 1.0D0
    ffwal = 0.92D0
    fgwped = 0.85D0
    fgwsep = 0.50D0
    f_helium3 = 0.0D0
    figmer = 0.0D0
    fkzohm = 1.0D0
    fplhsep = 1.0D0
    fpdivlim = 1.0D0
    fne0 = 1.0D0
    f_tritium = 0.5D0
    fusion_rate_density_total = 0.0D0
    fusion_rate_density_plasma = 0.0D0
    fvsbrnni = 1.0D0
    ejima_coeff = 0.4D0
    f_beta_alpha_beam_thermal = 0.0D0
    hfac = 0.0D0
    hfact = 1.0D0
    taumax = 10.0D0
    i_bootstrap_current = 3
    i_beta_component = 0
    i_plasma_current = 4
    i_diamagnetic_current = 0
    i_density_limit = 8
    idivrt = 2
    i_beta_fast_alpha = 1
    ignite = 0
    ipedestal = 1
    i_pfirsch_schluter_current = 0
    neped = 4.0D19
    nesep = 3.0D19
    alpha_crit = 0.0D0
    nesep_crit = 0.0D0
    plasma_res_factor = 1.0D0
    rhopedn = 1.0D0
    rhopedt = 1.0D0
    tbeta = 2.0D0
    teped = 1.0D0
    tesep = 0.1D0
    iprofile = 1
    i_rad_loss = 1
    i_confinement_time = 34
    i_plasma_wall_gap = 1
    i_plasma_geometry = 0
    i_plasma_shape = 0
    itart = 0
    itartpf = 0
    iwalld = 1
    plasma_square = 0.0D0
    kappa = 1.792D0
    kappa95 = 1.6D0

    kappa_ipb = 0.d0
    ne0 = 0.0D0
    ni0 = 0.0D0
    m_s_limit = 0.3D0
    p0 = 0.0D0
    j_plasma_0 = 0.0D0
    f_dd_branching_trit = 0.0D0
    alpha_power_density_plasma = 0.0D0
    alpha_power_density_total = 0.0D0
    alpha_power_electron_density = 0.0D0
    p_fw_alpha_mw = 0.0D0
    alpha_power_ions_density = 0.0D0
    alpha_power_total = 0.0D0
    alpha_power_plasma = 0.0D0
    alpha_power_beams = 0.0D0
    non_alpha_charged_power = 0.0D0
    charged_power_density = 0.0D0
    pcoef = 0.0D0
    p_plasma_inner_rad_mw = 0.0D0
    pden_plasma_core_rad_mw = 0.0D0
    dd_power = 0.0D0
    dhe3_power = 0.0D0
    pdivt = 0.0D0
    pdivl = 0.0D0
    pdivu = 0.0D0
    pdivmax = 0.0D0
    dt_power_total = 0.0D0
    dt_power_plasma = 0.0D0
    p_plasma_outer_rad_mw = 0.0D0
    pden_plasma_outer_rad_mw = 0.0D0
    charged_particle_power = 0.0D0
    vs_plasma_internal = 0.0D0
    pflux_fw_rad_mw = 0.0D0
    piepv = 0.0D0
    plasma_current = 0.0D0
    neutron_power_plasma = 0.0D0
    neutron_power_total = 0.0D0
    neutron_power_density_total = 0.0D0
    neutron_power_density_plasma = 0.0D0
    p_plasma_ohmic_mw = 0.0D0
    pden_plasma_ohmic_mw = 0.0D0
    p_plasma_loss_mw = 0.0D0
    fusion_power = 0.0D0
    len_plasma_poloidal = 0.0D0
    p_plasma_rad_mw = 0.0D0
    pden_plasma_rad_mw = 0.0D0
    pradsolmw = 0.0D0
    proton_rate_density = 0.0D0
    psolradmw = 0.0D0
    pden_plasma_sync_mw = 0.0D0
    p_plasma_sync_mw = 0.0D0
    i_l_h_threshold = 19
    p_l_h_threshold_mw = 0.0D0
    l_h_threshold_powers = 0.0D0
    p_electron_transport_loss_mw = 0.0D0
    pden_electron_transport_loss_mw = 0.0D0
    p_ion_transport_loss_mw = 0.0D0
    pscalingmw = 0.0D0
    pden_ion_transport_loss_mw = 0.0D0
    q0 = 1.0D0
    q95 = 0.0D0
    qfuel = 0.0D0
    tauratio = 1.0D0
    q95_min = 0.0D0
    qstar = 0.0D0
    rad_fraction_sol = 0.8D0
    rad_fraction_total = 0.0D0
    f_nd_alpha_electron = 0.10D0
    f_nd_protium_electrons = 0.0D0
    ind_plasma_internal_norm = 0.9D0
    ind_plasma = 0.0D0
    rmajor = 8.14D0
    rminor = 0.0D0
    f_nd_beam_electron = 0.005D0
    rncne = 0.0D0
    rndfuel = 0.0D0
    rnfene = 0.0D0
    rnone = 0.0D0
    f_res_plasma_neo = 0.0D0
    res_plasma = 0.0D0
    t_plasma_res_diffusion = 0.0D0
    a_plasma_surface = 0.0D0
    a_plasma_surface_outboard = 0.0D0
    i_single_null = 1
    f_sync_reflect = 0.6D0
    t_electron_energy_confinement = 0.0D0
    tauee_in = 0.0D0
    t_energy_confinement = 0.0D0
    t_ion_energy_confinement = 0.0D0
    t_alpha_confinement = 0.0D0
    f_alpha_energy_confinement = 0.0D0
    te = 12.9D0
    te0 = 0.0D0
    ten = 0.0D0
    ti = 12.9D0
    ti0 = 0.0D0
    tin = 0.0D0
    tratio = 1.0D0
    triang = 0.36D0
    triang95 = 0.24D0
    vol_plasma = 0.0D0
    vs_plasma_burn_required = 0.0D0
    v_plasma_loop_burn = 0.0D0
    vshift = 0.0D0
    vs_plasma_ind_ramp = 0.0D0
    vs_plasma_res_ramp = 0.0D0
    vs_plasma_total_required = 0.0D0
    pflux_fw_neutron_mw = 0.0D0
    wtgpd = 0.0D0
    a_plasma_poloidal = 0.0D0
    zeff = 0.0D0
    zeffai = 0.0D0
  end subroutine init_physics_variables
end module physics_variables
