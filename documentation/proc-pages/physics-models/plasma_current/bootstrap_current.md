## Overview

Bootstrap current in tokamaks originates from the pressure gradients within the plasma and the resulting collisions between particles. As the plasma pressure varies radially, it creates a differential in particle velocities, leading to a net drift of charged particles. This drift generates a toroidal current, known as the bootstrap current, which flows parallel to the magnetic field lines. The phenomenon is a consequence of the neoclassical transport theory, where the collisional processes in a magnetically confined plasma lead to a self-sustaining current. This current is particularly advantageous as it reduces the need for external current drive systems, thereby enhancing the efficiency and stability of the tokamak operation. The bootstrap current is proportional to the pressure gradient and the collisionality of the plasma, making it a critical factor in the design and operation of advanced tokamak reactors aiming for steady-state fusion.

Some more info can be found [here](https://wiki.fusion.ciemat.es/wiki/Bootstrap_current)
## Selection

The fraction of the plasma current provided by the bootstrap effect
can be either input into the code directly, or calculated using one of five
methods, as summarised here. Note that methods `i_bootstrap_current = 1-3 & 5` do not take into account the 
existence of pedestals, whereas the Sauter et al. scaling 
(`i_bootstrap_current = 4`) allows general profiles to be used. 

--------------

### ITER IPDG89 Scaling | `bootstrap_fraction_iter89()`
 
Original empirical ITER bootstrap scaling from the 1989 Physics Design Guidelines[^0]
Is selected by setting `i_bootstrap_current = 1`

Empirical fit for the bootstrap current fraction as:

$$
\frac{I_{\text{BS}}}{I} = C_{\text{BS}}\left(\epsilon^{0.5}\beta_{\text{pa}}\right)^{1.3}
$$

where:

$$
C_{\text{BS}} = 1.32 - 0.235\left(\frac{q_{95}}{q_0}\right) + 0.0185\left(\frac{q_{95}}{q_0}\right)^2
$$

$$
\beta_{\text{pa}} = \frac{\langle p \rangle}{\frac{B_{\text{pa}^2}}{2\mu_0}} = \beta_{\text{tot}}\left(\frac{B_0}{B_{\text{pa}}}\right)^2
$$

$$
B_{\text{pa}} = \frac{I}{5\langle a \rangle}
$$

$$
\langle a \rangle = \left(\frac{V}{2\pi^2 R_0}\right)^{0.5}
$$

Here, $\beta_{\text{tot}}$ is the average total plasma (toroidal) beta. $I$ is given in $\text{MA}$ and $B_0$ is the on-axis toroidal field in Tesla.

------------

### Nevins Scaling | `bootstrap_fraction_nevins()`

The general Nevins scaling is normally cited from a ITER specialists meeting in 1989[^1] which is not publicly accessible.
However it can be found in the appendix [here](https://doi.org/10.1016/j.fusengdes.2014.07.009)[^2].

Is selected by setting `i_bootstrap_current = 2`

$$
f_{\text{BS}} = \frac{2.5\beta_{e0}R_pB_{T}q_{95}}{I_{\text{P}}}\int^1_0 B_{\text{int}}\ dy
$$

$$
\beta_e = \frac{1.6022 \times 10^{-16}n_{\text{e}}T_{\text{e}}}{B_{T}^2 / 2\mu_0}
$$

$$
\beta_{e0} = \frac{1.6022 \times 10^{-16}n_{\text{e0}}T_{\text{e0}}}{B_{T}^2 / 2\mu_0}
$$

$$
\Delta = \epsilon y^{0.5}
$$

$$
x = \frac{1.46 \sqrt{\Delta}+2.4\Delta}{\left(1-\Delta\right)^{1.5}}
$$

$$
d = \sqrt{2}Z_{\text{eff}} + Z_{\text{eff}}^2 + x\left(0.75+2.657Z_{\text{eff}}+2Z_{\text{eff}}^2\right) \\
+x^2\left(0.348+1.243Z_{\text{eff}}+Z_{\text{eff}}^2\right)
$$

$$
A_1 = \left(\alpha_{\text{n}}+\alpha_{\text{T}}\right)\left(1-y\right)^{\alpha_{\text{n}}+\alpha_{\text{T}}-1}
$$

$$
A_2 = \alpha_{\text{T}}\left(1-y\right)^{\alpha_{\text{n}}+\alpha_{\text{T}}-1}
$$

$$
A_{l1} = \frac{x}{d}\left(0.754+2.21Z_{\text{eff}}+Z_{\text{eff}}^2+x\left(0.348+1.243Z_{\text{eff}}+Z_{\text{eff}}^2\right)\right)
$$

$$
A_{l2}=-x\frac{0.884+2.0742Z_{\text{eff}}}{d}
$$

$$
\alpha_i = -\frac{1.172}{1+0.462x}
$$

$$
q = q_0+\left(q_{95}-q_0\right)\frac{y+y^2+y^3}{3}
$$

$$
\beta_{\text{tot}} = \beta_{\text{T}}\frac{B_{\text{tot}}^2}{B_{\text{T}}^2} = \frac{\beta_{T}\left(B_{T}^2+\beta_{\text{P}}\right)}{B_{\text{T}}^2}
$$

$$
P_{\text{ratio}} = \frac{\beta_{\text{T}}-\beta_e}{\beta_e}
$$

$$
B_{\text{int}} = \frac{q}{q_{95}}\left[A_{l1}\left(A_1+P_{\text{ratio}}\left(A_1+\alpha_i A_2\right)\right)+A_{l2}A_2\right]
$$



----------------


### Wilson Scaling | `bootstrap_fraction_wilson()`

Wilson gives an empirical formula[^3] [^7] as a function of the pressure, 
temperature and total current profiles as well as the poloidal beta and aspect ratio of a tokamak. This empirical formula was compared with an expression obtained by the ITER group; also compared with an analytical result (valid at large aspect ratio). It is found that the determined empirical result agreed well with the large aspect ratio result, but not so well with the empirical formula of the ITER group[^0]

Is selected by setting `i_bootstrap_current = 3`

Data is fitted to 3000 equilibria evenly distributed across the parameter range below:

| Variable | $R$ | $A$ | $B_{\text{T}}$ | $\delta$ | $\kappa$ | $\alpha_{\text{P}}$ | $\alpha_{\text{T}}$ | $\alpha_{\text{J}}$ | $Z$ |
|----------|-----|-----|----------------|-----------|-----------|-------------------|-------------------|-------------------|-----|
| Value    | 5.31 | 1.1-5 | 6.2 | 0.2 | 2.0 | 1-3 | 0.1-$\alpha_{\text{P}}$ | 0.5-2.0 | 1-3 |


Using the relation:

$$
\frac{I_{\text{b}}}{I_{\text{P}}} = \beta_{\text{P}}\epsilon_0^{\frac{1}{2}} \sum_{i=1}^{12}a_i(\alpha_{\text{J}}, Z)b_i
$$

In the paper definition $\epsilon_0$ is not the standard inverse aspect ration but is defined as:

$$
\epsilon_0 = \frac{R_2-R_1}{R_2+R_1}
$$

Where $R_2$ and $R_1$ are the maximum and minimum radii of the plasma.

The poloidal beta term is defined as:

$$
\beta_{\text{P}} = \frac{2\mu_0\langle p \rangle}{\langle \langle B_{\text{P}} \rangle \rangle^2}
$$

Where $\langle p \rangle$ is the volume averaged pressure and $\langle \langle B_{\text{P}} \rangle \rangle$  is the plasma surface average of the polidal field.

The Wilson method extrapolates on the analytically derived expression for the bootstrap current fraction in the limit of large aspect ratio with circular cross-section. This large aspect expression assumed coincidence of a constant $r$ surface with a constant flux surface. The allowed the treatment of the temperature, pressure and current density as functions of $r$ only with the simple parabolic profile form. 
For the Wilson method this is expanded to the arbitrary aspect ratio case with D-shaped plasmas with a given triangularity and elongation. The profiles for pressure and temperature are now of the form:

$$
P=P_0\psi^{\alpha_{\text{P}}}
$$

Where $\psi$ is the flux function. For the current density we cannot model it in the form above as it varies across a flux surface. Instead it is convenient to consider a flux surface average of the total parallel current density which is taken to vary as:

$$
\frac{\langle j \cdot B \rangle}{\langle B^2 \rangle^{\frac{1}{2}}} = J_0 \psi^{\alpha_{\text{J}}}
$$

Describing this flux surface average by using a standard **parabolic profile** for the safety factor $(q)$.

$$
q(r) = q_0 +\left(q_{95}-q_0\right)\left(\frac{r}{a}\right)^2
$$

To relate the temperature, pressure and current density profiles across a flux surface average of the flux function we re-arrange their standard parabolic forms into one that can be integrated into the $(q)$ profile function. Using the temperature profile as an example:

$$
T(r) = T_0 \left(1-\left(\frac{r}{a}\right)^2\right)^{\alpha_{\text{T}}}
$$

$$
\frac{T(r)}{T_0} = \left(1-\left(\frac{r}{a}\right)^2\right)^{\alpha_{\text{T}}}
$$

$$
\left(\frac{r}{a}\right)^2 = 1-\left(\frac{T(r)}{T_0}\right)^{\frac{1}{\alpha_{\text{T}}}}
$$

Substituting the above into the $q$ profile and taking a median profile value of half the core value we get:

$$
q_0+(q_{95}-q_0)\times (1-0.5^{\frac{1}{\alpha_{\text{T}}}})
$$

To find the new flux surface average where the $q$ profile shares points where the temperature, pressure and current density are half of their core values, we relate the function above of the calculated $q$ value to the core $q_0$ value. Taking the natural log of the $q$ profile point at 50% of the core value for the temperature, pressure and current density we relate these two ratios compared to $q_{95}$. Dividing $\left(\ln 0.5\right)$ by this result provides new values of the profile exponents $\alpha_{\text{T}}, \alpha_{\text{P}} ,\alpha_{\text{J}}$ which are now averaged across a shared flux function. 


$$
\alpha = \frac{\ln{0.5}}{\ln{\frac{\ln \left(\frac{q_0+(q_{95}-q_0)\times (1-0.5^{\frac{1}{\alpha}})}{q95}\right)}{\ln{\frac{q_0}{q_{95}}}}}}
$$

These new calculated profile exponents $(\alpha)$ are what is used in the matrix values below and **not** the standard parabolic profile indexes.

$$
b_1 = 1 \ \ b_2 = \alpha_{\text{P}} \ \ b_3 = \alpha_{\text{T}} \ \ b_4 = \alpha_{\text{P}}\alpha_{\text{T}} \\
b_5 = \epsilon_0^{\frac{1}{2}} \ \ b_6 = \alpha_{\text{P}}\epsilon_0^{\frac{1}{2}}  \ \ b_7 = \alpha_{\text{T}}\epsilon_0^{\frac{1}{2}} \ \ b_8 = \alpha_{\text{P}}\alpha_{\text{T}}\epsilon_0^{\frac{1}{2}} \\
b_9 = \epsilon_0 \ \ b_{10} = \alpha_{\text{P}}\epsilon_0 \ \ b_{11} = \alpha_{\text{T}}\epsilon_0 \ \ b_{12} = \alpha_{\text{P}}\alpha_{\text{T}}\epsilon_0
$$

$$
a_1 = 1.41\left(1.0 - 0.28 \alpha_{\text{J}}^{\frac{1}{2}}\right)\left(1.0 + \frac{0.12}{Z}\right) \\
a_2 = 0.36  \left(1.0 - 0.59 \alpha_{\text{J}}^{\frac{1}{2}}\right) \left(1.0 + \frac{0.8}{Z}\right) \\
a_3 = -0.27  \left(1.0 - 0.47 \alpha_{\text{J}}^{\frac{1}{2}}\right)  \left(1.0 + \frac{3.0}{Z}\right) \\
a_4 = 0.0053  \left(1.0 + \frac{5.0}{Z}\right) \\
a_5 = -0.93  \left(1.0 - 0.34 \alpha_{\text{J}}^{\frac{1}{2}}\right)  \left(1.0 + \frac{0.15}{Z}\right) \\
a_6 = -0.26  \left(1.0 - 0.57 \alpha_{\text{J}}^{\frac{1}{2}}\right)  \left(1.0 - 0.27 Z\right) \\
a_7 = 0.064  \left(1.0 - 0.6 \alpha_{\text{J}} + 0.15 \alpha_{\text{J}}^2\right)  \left(1.0 + \frac{7.6}{Z}\right) \\
a_8 = -0.0011  \left(1.0 + \frac{9.0}{Z}\right) \\
a_9 = -0.33  \left(1.0 - \alpha_{\text{J}} + 0.33 \alpha_{\text{J}}^2\right) \\
a_{10} = -0.26  \left(1.0 - \frac{0.87}{\alpha_{\text{J}}^{\frac{1}{2}}} - 0.16 \alpha_{\text{J}}\right) \\
a_{11} = -0.14  \left(1.0 - \frac{1.14}{\alpha_{\text{J}}^{\frac{1}{2}}} - 0.45 \alpha_{\text{J}}^{\frac{1}{2}}\right) \\
a_{12} = -0.0069 \\
$$

Coefficients are obtained by a least squares fit to the 3000 equilibria numerical solutions. Error distribution shows an average error of 3.6% and a maximum error of 20%. These larger errors appear to come from the cases where the temperature profile is approximately equal to the pressure profile. For these cases the density profile is very flat over much of the plasma radius and (as it is forced to be zero at the plasma edge) this 
means that it falls off sharply at the plasma edge.

!!! quote "Excerpt from Wilson[^3]"

    *"The ITER group gives an expression for the bootstrap current[^0] which differs significantly from that 
    presented here. Their relatively simple expression,
    shows no variation with density and temperature profiles. It also does not reproduce the large aspect ratio 
    scaling with $\beta_{\text{P}}$, and $\epsilon$ in this limit."*   

!!! warning "Flux average profile indexes"

    The implementation of calculating the new profile indexes to be used in the $a$ & $b$ coefficients is not explicitly given in  Wilson[^3]. It is an adaption to make the method more aligned with what `PROCESS` calculates.    


--------------------

### Sauter Scaling | `bootstrap_fraction_sauter()`

Sauter et al.[^4] [^5] provides a formula using the exact Fokker–Planck operator and
without any approximation on the plasma geometry or collisionality. In this way we have been able to accurately determine the neoclassical resistivity and the coefficients for the
bootstrap current which allows one to calculate the bootstrap fraction.

Is selected by setting `i_bootstrap_current = 4`

$$
\left\langle j_{\|} B\right\rangle= \sigma_{\text {neo }}\left\langle E_{\|} B\right\rangle-I(\psi) p(\psi)\left[\mathcal{L}_{31} \frac{\partial \ln n_e}{\partial \psi} \\
+R_{p e}\left(\mathcal{L}_{31}+\mathcal{L}_{32}\right) \frac{\partial \ln T_e}{\partial \psi}+\left(1-R_{p e}\right) \times\left(1+\frac{\mathcal{L}_{34}}{\mathcal{L}_{31}} \alpha\right) \mathcal{L}_{31} \frac{\partial \ln T_i}{\partial \psi}\right]
$$

Note that the above $\left\langle j_{\|} B\right\rangle$ given by Sauter et.al.[^4]   gives the component of the current density in the direction of the field – not the toroidal component of the current.  The error is second order in the pitch angle, but can be important, especially in the outboard region of a low aspect ratio tokamak, where the pitch angle is large.  Moreover this error will always have the effect of overestimating the current. This is accounted for via poloidal correction function implemented as: [`beta_poloidal_sauter()`](#calculate-electron-only-poloidal-beta-correction-beta_poloidal_sauter) and[`beta_poloidal_total_sauter()`](#calculate-ion-and-electron-poloidal-beta-correction-beta_poloidal_total_sauter)

The correction is of the form:

$$
I_{\phi}^{\text{bs}} = 2\pi \int d\psi \frac{q(\psi)}{\langle B^{2}\rangle} \left\langle j_{\|} B\right\rangle
$$

where $q(\psi)$ is the safety factor.

The reconstructed implementation in PROCESS looks as such:

$$
\frac{I_{\text{b}}}{I_{\text{P}}} = \sum_2^{\rho_{\text{max}}} \left(2\pi\left[\rho\right]_{-1} \times \left(\left[\rho\right] -\left[\rho\right]_{-1}\right)  \right) \times  \\
\left(0.5 \times \left[\mathcal{L}_{31} \frac{\partial \ln n_e}{\partial \psi} 
+\left(\mathcal{L}_{31} + \mathcal{L}_{32}\right) \frac{\partial \ln T_e}{\partial \psi} 
+ \left(1 + \frac{\mathcal{L}_{34}}{\mathcal{L}_{31}} \alpha\right) \mathcal{L}_{31} \frac{\partial \ln T_i}{\partial \psi}\right] \times \\
 1 \times 10^6 \times \frac{-B_{0}\left[\rho\right]_{-1}\left[\frac{1}{q}\right]_{-1}}{0.2\pi R_0}\right)
$$

In this case square brackets denote array variables equal in length to $\rho_{\text{max}}$ representing the normalized radius elements across the profile. The $-1$ subscript denotes the previous array element in the summation.

It is not known fully if the $\left(\sigma_{\text {neo }}\left\langle E_{\|} B\right\rangle-I(\psi) p(\psi)\right)$ term is properly implemented into the `PROCESS` version. The $R_{pe}$ value is assumingly taken to be 0.5 as it is stated to approximately to be in Sauter et.al[^4].

!!! warning "Validity of the Sauter Bootstrap Scaling"

    In its current state the several base functions called by the Sauter scaling have no reference and cannot be verified. The ad-hoc adaption of the Sauter scaling for use in `PROCESS` is done knowing that `PROCESS` does not calculate flux surfaces across the plasma.

-----------

#### Calculate the trapped particle fraction | `_trapped_particle_fraction_sauter()`

This function calculates the trapped particle fraction $\left(f_t\right)$ used within other key internal Sauter scaling functions.

$$
f_t = \frac{1.0 - (1.0-\epsilon_{-1})\sqrt{1.0-\epsilon_{-1}}}{\left(1.0+1.46 \sqrt{\epsilon_{-1}}\right)}
$$

$\epsilon$ in this case is the local aspect ratio at that normalised radial point in the profile given by $\epsilon = \rho \left(\frac{a}{R}\right)$. The value of $\rho$ varies from 0 to 1 across the profile.

The $-1$ subscript in this case refers to the value of the variable in the previous array index value.

-------------

#### Calculate electron density coefficient | `_calculate_l31_coefficient()`

This function calculates and returns the $\mathcal{L}_{31}$ coefficient value for $\frac{\partial \ln n_e}{\partial \psi}$

$$
\mathcal{L}_{31}= F_{31}\left(X=f_{\text {teff }}^{31}\right) \equiv\left(1+\frac{1.4}{Z+1}\right) X-\frac{1.9}{Z+1} X^2+\frac{0.3}{Z+1} X^3 +\frac{0.2}{Z+1} X^4 \\
f_{\text {teff }}^{31}\left(\nu_{e *}\right)= \frac{f_t}{1+\left(1-0.1 f_t\right) \sqrt{\nu_{e *}}+0.5\left(1-f_t\right) \nu_{e *} / Z}
$$

The returned value is $\mathcal{L}_{31} \times$ [`_beta_poloidal_total_sauter()`](#calculate-ion-and-electron-poloidal-beta-correction-beta_poloidal_total_sauter)  

---------------

#### Calculate electron temperature coefficient | `_calculate_l31_32_coefficient()`

This function calculates and returns the $\left(\mathcal{L}_{31}+\mathcal{L}_{32}\right)$ coefficient value for $\frac{\partial \ln T_{\text{e}}}{\partial \psi}$.

$$
\begin{align*}
\mathcal{L}_{32} &= F_{32 \_e e}\left(X=f_{\text {teff }}^{32 \_e e}\right)+F_{32 \_e i}\left(Y=f_{\text {teff }}^{32 \_e i}\right) \\ \\
F_{32 \_e e}(X) &= \frac{0.05+0.62 Z}{Z(1+0.44 Z)}\left(X-X^4\right)+\frac{1}{1+0.22 Z}\left[X^2-X^4\right. \\
& \quad \left.-1.2\left(X^3-X^4\right)\right]+\frac{1.2}{1+0.5 Z} X^4 \\
f_{\text {teff }}^{32_{-e e}}\left(\nu_{e *}\right) &= \frac{f_t}{1+0.26\left(1-f_t\right) \sqrt{\nu_{e *}}+0.18\left(1-0.37 f_t\right) \frac{\nu_{e *}}{\sqrt{Z}}}
\end{align*}
$$

$$
F_{32 \_e i}(Y) = -\frac{0.56+1.93 Z}{Z(1+0.44 Z)}\left(Y-Y^4\right)+\frac{4.95}{1+2.48 Z}\left[Y^2-Y^4\right] \\
-0.55\left(Y^3-Y^4\right)-\frac{1.2}{1+0.5 Z} Y^4 \\
f_{\text {teff }}^{32 \_e i}\left(\nu_{e *}\right) = \frac{f_t}{1+\left(1+0.6 f_t\right) \sqrt{\nu_{e *}}+0.85\left(1-0.37 f_t\right) \nu_{e *}(1+Z)}
$$

The above is added to a call of [`_calculate_l31_coefficient()`](#calculate-electron-density-coefficient-calculate_l31_coefficient). This is then multiplied by [`_beta_poloidal_sauter()`](#calculate-electron-only-poloidal-beta-correction-beta_poloidal_sauter). 

This product above is then multiplied by ([`_beta_poloidal_sauter()`](#calculate-electron-only-poloidal-beta-correction-beta_poloidal_sauter) divided by [`_beta_poloidal_total_sauter()`](#calculate-ion-and-electron-poloidal-beta-correction-beta_poloidal_total_sauter))

---------------

#### Calculate ion temperature coefficient | `_calculate_l34_alpha_31_coefficient()`

This function calculates and returns the $\left(1+\frac{\mathcal{L}_{34}}{\mathcal{L}_{31}}\alpha\right)\mathcal{L}_{31}$ coefficient value for $\frac{\partial \ln T_{\text{i}}}{\partial \psi}$.

$$
\mathcal{L}_{34}=F_{31}\left(X=f_{\text {teff }}^{34}\right) 
$$

$$
f_{\text {teff }}^{34}\left(\nu_{e *}\right)=\frac{f_t}{1+\left(1-0.1 f_t\right) \sqrt{\nu_{e *}}+0.5\left(1-0.5 f_t\right) \nu_{e *} / Z} 
$$

$$
\alpha_0=-\frac{1.17\left(1-f_t\right)}{1-0.22 f_t-0.19 f_t^2} 
$$

$$
\alpha\left(\nu_{i *}\right)=\left[\frac{\alpha_0+0.25\left(1-f_t^2\right) \sqrt{\nu_{i *}}}{1+0.5 \sqrt{\nu_{i *}}} \\
+0.315 \nu_{i *}^2 f_t^6\right] \frac{1}{1+0.15 \nu_{i *}^2 f_t^6}
$$

The definition of $\alpha\left(\nu_{i *}\right)$ is that found in the erratum paper which changes the value of $-0.315\nu_{i *}^2 f_t^6$ to positive.[^5]

The return sequence is ([`_beta_poloidal_total_sauter()`](#calculate-ion-and-electron-poloidal-beta-correction-beta_poloidal_total_sauter) - [`_beta_poloidal_sauter()`](#calculate-electron-only-poloidal-beta-correction-beta_poloidal_sauter)) $\times (\mathcal{L}_{34} + \alpha)$ + [`_calculate_l31_coefficient()`](#calculate-electron-density-coefficient-calculate_l31_coefficient) $\times$ (1.0 -  [`_beta_poloidal_sauter()`](#calculate-electron-only-poloidal-beta-correction-beta_poloidal_sauter) divided by [`_beta_poloidal_total_sauter()`](#calculate-ion-and-electron-poloidal-beta-correction-beta_poloidal_total_sauter))



-------------

#### Calculate the Coulomb logarithm | `_coulomb_logarithm_sauter()`

$$
\ln \Lambda = 15.9 -0.5 \times \ln{n_{\text{e}}}+\ln{T_{\text{e}}}
$$

-----------

#### Calculate frequency of electron collisions | `_electron_collisions_sauter()`

Using the Coulomb logarithm ($\ln \Lambda$) calculated from [`_coulomb_logarithm_sauter()`](#calculate-the-coulomb-logarithm--_coulomb_logarithm_sauter) we get:

$$
\nu_{\text{e}} = 670 \times \frac{\ln \Lambda \times n_{\text{e}}}{T_{\text{e}}^{3/2}}
$$

------------

#### Calculate electron collisionality | `_electron_collisionality_sauter()`

The origins of the coefficients values are not known, but thought to be derived from a condition of the [Bohm diffusion coefficient](https://en.wikipedia.org/wiki/Bohm_diffusion)

Using the electron collision frequency ($\nu_{\text{e}}$) calculated from [`_electron_collisions_sauter()`](#calculate-frequency-of-electron-collisions--_electron_collisions_sauter) we get:
$$
\nu_{\text{e*}} = \frac{1.4 \ R \ \nu_{\text{e}}  \ Z_{\text{eff}}}{\left|\frac{1}{q}\epsilon^{3/2}\sqrt{T_{\text{e}}}\times 1.875\times10^7\right|}
$$

-------------

#### Calculate frequency of ion collisions | `_ion_collisions_sauter()`


$$
\nu_{\text{i}} = 320 \times \frac{Z_{\text{eff}}^4n_{\text{i}}}{T_{\text{i}}^{3/2}\sqrt{a_{\text{i}}}}
$$

-----

#### Calculate ion collisionality | `_ion_collisionality_sauter()`

The origins of the coefficients values are not known, but thought to be derived from a condition of the [Bohm diffusion coefficient](https://en.wikipedia.org/wiki/Bohm_diffusion)

Using the ion collision frequency ($\nu_{\text{i}}$) calculated from [`_ion_collisions_sauter()`](#calculate-frequency-of-ion-collisions--_ion_collisions_sauter) we get:

$$
\nu_{\text{e*}} = \frac{3.2\times10^{-6} \nu_{\text{i}} R}{\left|\left(\frac{1}{q}+0.0001\right)\epsilon^{3/2} \sqrt{\frac{T_{\text{i}}}{a_{\text{i}}}} \right|}
$$

----------------

#### Calculate electron only poloidal beta correction | `_beta_poloidal_sauter()`

This function returns an electron only local poloidal beta correction dependant on the array index of the profile.

If the current index is not equal to the size value (or end of the array) then the following is returned:

$$
\frac{1.6\times 10^{-4}\pi R \left(n_{\text{e}}+n_{\text{e-1}}\right)\times \left(T_{\text{e}}+T_{\text{e-1}}\right)}{\left(B_{\text{T}}\rho_{-1}\left|\left(\frac{1}{q}\right)_{-1}+1\times 10^{-4}\right|\right)^2}
$$

Otherwise the following is returned:

$$
\frac{6.4\times 10^{-4}\pi R \left(n_{\text{e-1}}T_{\text{e-1}}\right)}{\left(B_{\text{T}}\rho_{-1}\left|\left(\frac{1}{q}\right)_{-1}+1\times 10^{-4}\right|\right)^2}
$$

The $-1$ subscript in this case refers to the value of the variable in the previous array index value



---------------

#### Calculate ion and electron poloidal beta correction | `_beta_poloidal_total_sauter()`

This function returns the local poloidal beta correction with both electron and ion pressure dependant on the array index of the profile.

If the current index is not equal to the size value (or end of the array) then the following is returned:

$$
\frac{1.6\times 10^{-4}\pi R \left[\left(\left(n_{\text{e}}+n_{\text{e-1}}\right)\times \left(T_{\text{e}}+T_{\text{e-1}}\right)\right)+\left(\left(n_{\text{i}}+n_{\text{i-1}}\right)\times \left(T_{\text{i}}+T_{\text{i-1}}\right)\right)\right]}{\left(B_{\text{T}}\rho_{-1}\left|\left(\frac{1}{q}\right)_{-1}+1\times 10^{-4}\right|\right)^2}
$$

Otherwise the following is returned:

$$
\frac{6.4\times 10^{-4}\pi R \left[\left(n_{\text{e-1}}T_{\text{e-1}}\right)+\left(n_{\text{i-1}}T_{\text{i-1}}\right)\right]}{\left(B_{\text{T}}\rho_{-1}\left|\left(\frac{1}{q}\right)_{-1}+1\times 10^{-4}\right|\right)^2}
$$

The $-1$ subscript in this case refers to the value of the variable in the previous array index value


------------------

### Sakai Scaling | `bootstrap_fraction_sakai()`

Is selected by setting `i_bootstrap_current = 5`[^5]

$$
f_{\text{BS}} = 10^{0.951 \epsilon - 0.948} \cdot \beta_p^{1.226 \epsilon + 1.584} \cdot l_i^{-0.184\epsilon - 0.282} \cdot \left(\frac{q_{95}}{q_0}\right)^{-0.042 \epsilon - 0.02} \\
\cdot \alpha_n^{0.13 \epsilon + 0.05} \cdot \alpha_t^{0.502 \epsilon - 0.273}
$$

The model includes the toroidal diamagnetic current in the calculation due to the dataset, so `i_diamagnetic_current = 0` can only be used with it

---------------------

## Setting of maximum desirable bootstrap current fraction

The variable `bootstrap_current_fraction_max` can be set to the value of maximum desirable bootstrap current fraction for a specific design. When optimising if the value of the calculated `bootstrap_current_fraction` for the model selected with `i_bootstrap_current` exceeds this value, then `bootstrap_current_fraction` is set to the value of `bootstrap_current_fraction_max`.

An error is also raised to the user in the terminal output at the end of the run saying "Bootstrap fraction upper limit enforced".

## Fixing the bootstrap current fraction

If the user wants to set the value of the bootstrap current fraction directly then the value can be set by assigning the negative of the desired value to `bootstrap_current_fraction_max`.


```txt
>>> IN.DAT

# Setting a fixed bootstrap current fraction of 80%

bootstrap_current_fraction_max = -0.8
```


[^0]: N.A. Uckan and ITER Physics Group, 'ITER Physics Design Guidelines: 1989',
[^1]: Nevins, W. M. "Summary report: ITER specialists’ meeting on heating and current drive." ITER-TN-PH-8-4, June 1988. 1988. 
[^2]: Keii Gi, Makoto Nakamura, Kenji Tobita, Yasushi Ono, Bootstrap current fraction scaling for a tokamak reactor design study,
Fusion Engineering and Design, Volume 89, Issue 11, 2014, Pages 2709-2715, ISSN 0920-3796, https://doi.org/10.1016/j.fusengdes.2014.07.009.
[^3]: Wilson, H.R. (1992). Bootstrap current scaling in tokamaks. Nuclear Fusion, 32(2), pp.257–263. doi:https://doi.org/10.1088/0029-5515/32/2/i05.
[^4]: O. Sauter, C. Angioni, Y. R. Lin-Liu; Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime. Phys. Plasmas 1 July 1999; 6 (7): 2834–2839. https://doi.org/10.1063/1.873240 
[^5]: O. Sauter, C. Angioni, Y. R. Lin-Liu; Erratum: “Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime” [Phys. Plasmas 6, 2834 (1999)]. Phys. Plasmas 1 December 2002; 9 (12): 5140. https://doi.org/10.1063/1.1517052  
[^6]: Ryosuke Sakai, Takaaki Fujita, Atsushi Okamoto, Derivation of bootstrap current fraction scaling formula for 0-D system code analysis, Fusion Engineering and Design, Volume 149, 2019, 111322, ISSN 0920-3796, https://doi.org/10.1016/j.fusengdes.2019.111322.
[^7]: T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992 

