"""Closed-form results of the Chazalviel / Sand picture for a binary electrolyte.

Geometry: two parallel metal electrodes (copper or silver) a distance L apart, constant current
density J (A/m^2).  Copper dissolves at the anode and plates at the cathode, so
the total amount of salt is conserved and the salt concentration c(x, t) obeys a
plain diffusion equation with the flux boundary condition

    D dc/dx = J t_a / (z F)        at both electrodes

(t_a = anion transference number: migration carries the fraction t_c of the
metal-ion flux, diffusion has to supply the rest).
"""
import numpy as np
from scipy.optimize import brentq

from .properties import F, Z, G0, Electrolyte


# --- Sand's time --------------------------------------------------------------
def sand_time(J, c0, D, t_a, z=Z):
    """Sand's time (s) for a semi-infinite electrolyte: tau = pi D (z c0 F / (2 J t_a))^2."""
    return np.pi * D * (z * c0 * F / (2.0 * np.asarray(J, dtype=float) * t_a)) ** 2


def current_density_for_sand_time(tau, c0, D, t_a, z=Z):
    """Inverse of sand_time: the J that puts Sand's time at tau."""
    return z * c0 * F / (2.0 * t_a) * np.sqrt(np.pi * D / tau)


def diffusion_limited_current(c0, D, t_a, L, z=Z):
    """Chazalviel's J*: the largest J a closed cell of gap L can carry forever by
    diffusion + migration alone.  Sand's time only exists for J > J*."""
    return 2.0 * z * c0 * F * D / (t_a * L)


def depletion_length(D, tau):
    """Thickness of the depleted layer when Sand's time is reached, sqrt(pi D tau)."""
    return np.sqrt(np.pi * D * tau)


# --- exact solution in a finite cell -----------------------------------------
def _series(theta, n_terms=400):
    """S(theta) = (8/pi^2) * sum_{n odd} exp(-n^2 pi^2 theta) / n^2;  S(0) = 1."""
    n = np.arange(1, 2 * n_terms, 2, dtype=float)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return (8.0 / np.pi ** 2) * np.sum(np.exp(-np.outer(theta, n ** 2) * np.pi ** 2) / n ** 2, axis=1)


def cathode_concentration(t, J, c0, D, t_a, L, z=Z):
    """Concentration at the cathode surface in a closed cell of gap L (constant D)."""
    G = J * t_a / (z * F * D)          # concentration gradient at the electrodes
    return c0 - 0.5 * G * L * (1.0 - _series(D * np.asarray(t, dtype=float) / L ** 2))


def concentration_profile(x, t, J, c0, D, t_a, L, z=Z, n_terms=400):
    """c(x, t) in the closed cell; x = 0 is the cathode, x = L the anode."""
    G = J * t_a / (z * F * D)
    x = np.asarray(x, dtype=float)
    n = np.arange(1, 2 * n_terms, 2, dtype=float)
    terms = (4.0 * G * L / (n ** 2 * np.pi ** 2)) * np.cos(np.outer(x, n) * np.pi / L) \
        * np.exp(-n ** 2 * np.pi ** 2 * D * t / L ** 2)
    return c0 + G * (x - L / 2.0) + terms.sum(axis=1)


def sand_time_finite_cell(J, c0, D, t_a, L, z=Z):
    """Time for the cathode concentration to reach zero in a closed cell of gap L.

    Returns np.inf when J <= J* (the cell settles into a steady state instead).
    For L >> sqrt(D tau) this equals sand_time().
    """
    if J <= diffusion_limited_current(c0, D, t_a, L, z) * (1.0 + 1e-9):
        return np.inf
    tau_inf = float(sand_time(J, c0, D, t_a, z))
    f = lambda t: float(cathode_concentration(t, J, c0, D, t_a, L, z)[0])
    hi = tau_inf
    while f(hi) > 0.0:
        hi *= 2.0
    return brentq(f, 0.0, hi, xtol=1e-9 * hi)


# --- buoyant convection (what gravity adds) ----------------------------------
def convective_layer_thickness(g, c0, height, T_C=25.0, orientation="vertical",
                               electrolyte=None, delta_c=None):
    """Thickness (m) of the diffusion layer that steady natural convection maintains.

    Uses mass-transfer correlations measured for copper deposition from CuSO4:
      vertical electrodes      Sh = 0.673 (Sc Gr)^(1/4)   (Wilke, Eisenberg & Tobias 1953)
      cathode facing up        Sh = 0.19  (Sc Gr)^(1/3)   (Fenech & Tobias 1960)
      cathode facing down      stably stratified - no convection, returns inf
    with Gr = g (d rho/rho) h^3 / nu^2 and delta = h / Sh.  Below Ra = Sc Gr ~ 1e3
    buoyancy is too weak to organise a boundary layer; inf is returned.

    orientation: 'vertical', 'cathode_below' (unstable) or 'cathode_above' (stable).
    """
    el = electrolyte or Electrolyte()
    if g <= 0.0 or orientation == "cathode_above":
        return np.inf
    dc = c0 if delta_c is None else delta_c
    nu = float(el.viscosity(c0, T_C) / el.density(c0))
    D = float(el.D(c0, T_C))
    rayleigh = g * el.densification() * dc * height ** 3 / (nu * D)     # = Sc * Gr
    if rayleigh < 1e3:
        return np.inf
    if orientation == "vertical":
        sherwood = 0.673 * rayleigh ** 0.25
    elif orientation == "cathode_below":
        sherwood = 0.19 * rayleigh ** (1.0 / 3.0)
    else:
        raise ValueError("orientation must be 'vertical', 'cathode_below' or 'cathode_above'")
    return height / (sherwood * el.convection_scale)


def convective_limiting_current(g, c0, height, T_C=25.0, orientation="vertical", electrolyte=None):
    """Largest J that steady natural convection can supply (A/m^2).  Below this, a
    cell in gravity never reaches Sand's time; 0 if there is no convection."""
    el = electrolyte or Electrolyte()
    delta = convective_layer_thickness(g, c0, height, T_C, orientation, el)
    if not np.isfinite(delta):
        return 0.0
    c = np.linspace(0.0, c0, 400)
    D_integral = float(np.sum(0.5 * (el.D(c[1:], T_C) + el.D(c[:-1], T_C)) * np.diff(c)))   # integral of D dc
    return el.z * F * D_integral / (el.t_anion * delta)


def steady_film_thickness(J, g, c0, height, T_C=25.0, orientation="vertical", electrolyte=None):
    """Self-consistent convective film for a cell running BELOW its limiting current.

    convective_layer_thickness() sizes the buoyancy with the full depletion c0 (right
    for asking "does Sand's time happen?").  A cell far below the limit only
    depletes by dc = J t_a delta / (z F D) at the wall, which drives weaker
    convection and a thicker film.  Returns (delta, dc); delta = inf if no convection.
    """
    el = electrolyte or Electrolyte()
    D = float(el.D(c0, T_C))
    dc = c0
    delta = np.inf
    for _ in range(200):
        delta = convective_layer_thickness(g, c0, height, T_C, orientation, el, delta_c=dc)
        if not np.isfinite(delta):
            return np.inf, np.nan
        dc_new = min(J * el.t_anion * delta / (el.z * F * D), c0)
        if abs(dc_new - dc) < 1e-9 * c0:
            break
        dc = dc_new
    return delta, dc


def critical_gravity(J, c0, L, height, T_C=25.0, orientation="vertical", electrolyte=None):
    """Gravity (m/s^2) below which a cell at current density J reaches Sand's time.

    Above it, convection carries the current and the cell never depletes.  Returns
    inf if convection can't carry J even at 1000 g (always depletes), 0 if J < J* (never depletes).
    """
    el = electrolyte or Electrolyte()
    D = float(el.D(c0, T_C))
    if J <= diffusion_limited_current(c0, D, el.t_anion, L, el.z):
        return 0.0
    lim = lambda g: convective_limiting_current(g, c0, height, T_C, orientation, el)
    if orientation == "cathode_above" or lim(1000 * G0) < J:
        return np.inf
    lo, hi = 1e-12 * G0, 1000 * G0
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if lim(mid) < J else (lo, mid)
    return hi


def effective_gap(L, delta):
    """A stirred core between two stagnant films of thickness delta behaves like a
    closed cell of gap 2*delta; pure diffusion applies when the films overlap."""
    return min(L, 2.0 * delta)


# --- after Sand's time --------------------------------------------------------
def dendrite_front_velocity(J, c0, t_a, z=Z):
    """Chazalviel: the ramified deposit advances at the anion drift velocity,
    v = mu_a E = J t_a / (z F c0)   (m/s)."""
    return np.asarray(J, dtype=float) * t_a / (z * F * c0)


def compact_growth_rate(J, electrolyte=None):
    """Thickness growth of a dense metal layer before Sand's time, m/s (Faraday)."""
    el = electrolyte or Electrolyte()
    return np.asarray(J, dtype=float) * el.metal_molar_mass / (el.z * F * el.metal_density)


def relaxation_time(L, D):
    """Slowest decay time of a concentration disturbance in a closed cell, L^2/(pi^2 D).
    Wait several of these between trials that reuse the same electrolyte."""
    return L ** 2 / (np.pi ** 2 * D)


def gravity_summary(J, c0, L, height, T_C=25.0, g=G0, orientation="vertical", electrolyte=None):
    """One-stop comparison of a cell in microgravity and at gravity g (constant-D estimates)."""
    el = electrolyte or Electrolyte()
    D = float(el.D(c0, T_C))
    ta, z = el.t_anion, el.z
    delta = convective_layer_thickness(g, c0, height, T_C, orientation, el)
    L_g = effective_gap(L, delta)
    return {
        "D": D,
        "tau_sand_ideal": float(sand_time(J, c0, D, ta, z)),
        "tau_microgravity": sand_time_finite_cell(J, c0, D, ta, L, z),
        "tau_gravity": sand_time_finite_cell(J, c0, D, ta, L_g, z),
        "J_limit_microgravity": diffusion_limited_current(c0, D, ta, L, z),
        "J_limit_gravity": max(convective_limiting_current(g, c0, height, T_C, orientation, el),
                               diffusion_limited_current(c0, D, ta, L, z)),
        "convective_layer": delta,
        "dendrite_velocity": float(dendrite_front_velocity(J, c0, ta, z)),
    }
