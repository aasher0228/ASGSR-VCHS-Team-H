"""1-D time-dependent simulation of a metal | salt | metal cell (Cu|CuSO4|Cu or
Ag|AgNO3|Ag) at constant current.

Solves   dc/dt = d/dx ( D(c, T) dc/dx )   between cathode (x = 0) and anode (x = L)
with the salt flux J t_a / (z F) leaving at the cathode and entering at the anode,
and converts the concentration field into the cell voltage the flight hardware
would log:

    V = ohmic drop + concentration (Nernst + diffusion) potential + activation
        + I * series_resistxance   (anything outside the electrolyte: films, contacts)

Beyond the textbook formula this captures (a) the concentration dependence of D,
(b) the finite gap and (c) gravity, through a stagnant-film model: natural
convection keeps the core of the cell mixed and leaves a diffusion layer of
thickness delta(g) on each electrode.  g = 0 is pure diffusion.

The model stops being valid at Sand's time: once the cathode concentration hits
zero the deposit goes dendritic and the voltage is set by the power supply.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from . import analytic
from .properties import F, R_GAS, Electrolyte


@dataclass
class Cell:
    """One electrochemical cell.  SI units throughout."""
    c0: float = 100.0              # bulk salt concentration, mol/m^3 (100 = 0.1 M)
    gap: float = 5e-3              # electrode separation, m
    cathode_area: float = 1e-4     # wetted cathode area, m^2 (1e-4 = 1 cm^2)
    height: float = 10e-3          # vertical extent of the electrodes, m (sets convection)
    current: float = 10e-3         # applied current, A
    T_C: float = 25.0              # electrolyte temperature, deg C
    electrolyte: Electrolyte = field(default_factory=Electrolyte)
    series_resistance: float = 0.0 # ohm outside the electrolyte model (fit to a measured start voltage)

    @property
    def J(self):
        """Mean cathode current density, A/m^2."""
        return self.current / self.cathode_area

    @property
    def D0(self):
        return float(self.electrolyte.D(self.c0, self.T_C))

    def sand_time_ideal(self):
        """Textbook Sand's time using D at the bulk concentration."""
        return float(analytic.sand_time(self.J, self.c0, self.D0, self.electrolyte.t_anion, self.electrolyte.z))


@dataclass
class Result:
    t: np.ndarray               # s
    c_cathode: np.ndarray       # mol/m^3
    c_anode: np.ndarray
    V_ohmic: np.ndarray         # V
    V_concentration: np.ndarray
    V_activation: np.ndarray
    V: np.ndarray               # total, clipped at the supply compliance
    x: np.ndarray               # cell-centre positions, m
    c: np.ndarray               # concentration profiles, shape (len(t), len(x))
    tau: float                  # time the cathode concentration reaches zero (inf if never)
    delta: float                # convective layer thickness used (inf = pure diffusion)
    cell: Cell = None
    g: float = 0.0

    @property
    def reached_sand_time(self):
        return np.isfinite(self.tau)

    def voltage_transition_time(self, rise=0.5):
        """What the flight software can measure: first time the voltage has risen
        `rise` volts above its starting value.  inf if it never does."""
        above = np.nonzero(self.V >= self.V[0] + rise)[0]
        if len(above) == 0:
            return np.inf
        k = above[0]
        if k == 0:
            return float(self.t[0])
        v0, v1 = self.V[k - 1], self.V[k]
        frac = (self.V[0] + rise - v0) / (v1 - v0) if v1 > v0 else 1.0
        return float(self.t[k - 1] + frac * (self.t[k] - self.t[k - 1]))


def _grid(L, h0, growth=1.06):
    """Cell faces on [0, L], refined geometrically towards both electrodes."""
    h_max = L / 150.0
    widths, h, x = [], min(h0, h_max), 0.0
    while x + h < L / 2.0:
        widths.append(h)
        x += h
        h = min(h * growth, h_max)
    widths.append(L / 2.0 - x)
    half = np.array(widths)
    return np.concatenate([[0.0], np.cumsum(np.concatenate([half, half[::-1]]))])


def _activation_overpotential(j, i0, alpha_a, alpha_c, f):
    """Solve Butler-Volmer  j = i0 [exp(alpha_a f eta) - exp(-alpha_c f eta)]  for eta."""
    def resid(eta):
        return i0 * (np.exp(alpha_a * f * eta) - np.exp(-alpha_c * f * eta)) - j
    lo, hi = -5.0, 5.0
    return brentq(resid, lo, hi, xtol=1e-9)


def simulate(cell, g=0.0, orientation="vertical", v_compliance=3.3, t_end=None,
             n_steps=1500, mixing_factor=1e4, delta=None):
    """Run the constant-current transient.

    g: gravitational acceleration, m/s^2 (0 = ideal microgravity, 9.81 = ground unit).
    orientation: 'vertical', 'cathode_below' or 'cathode_above' (only matters if g > 0).
    v_compliance: the highest voltage the current source can put across the cell.
    delta: override the convective film thickness (m); e.g. from
        analytic.steady_film_thickness() for a cell far below its limiting current.
    """
    el, c0, L, J, T = cell.electrolyte, cell.c0, cell.gap, cell.J, cell.T_C
    ta, Z = el.t_anion, el.z
    flux = J * ta / (Z * F)                      # salt flux at the electrodes, mol/(m^2 s)
    D0 = cell.D0

    if delta is None:
        delta = analytic.convective_layer_thickness(g, c0, cell.height, T, orientation, el)
    L_eff = analytic.effective_gap(L, delta)
    tau_est = analytic.sand_time_finite_cell(J, c0, D0, ta, L_eff, Z)

    # grid: resolve whichever is thinnest - the depletion layer or the convective film
    ell = analytic.depletion_length(D0, tau_est) if np.isfinite(tau_est) else L_eff
    faces = _grid(L, min(ell, L_eff, L) / 200.0)
    w = np.diff(faces)
    x = 0.5 * (faces[1:] + faces[:-1])
    d = 0.5 * (w[1:] + w[:-1])                   # centre-to-centre distances
    n = len(x)

    # stagnant-film model: strongly enhanced mixing further than delta from either electrode
    wall_dist = np.minimum(x, L - x)
    mix = np.ones(n)
    if np.isfinite(delta) and 2.0 * delta < L:
        mix = np.where(wall_dist > delta, mixing_factor, 1.0)

    if t_end is None:
        t_relax = L_eff ** 2 / (np.pi ** 2 * D0)
        t_end = 3.0 * tau_est if np.isfinite(tau_est) else 8.0 * t_relax
    dt_max = t_end / n_steps
    dt = dt_max * 1e-4

    c = np.full(n, float(c0))
    f = F / (R_GAS * (T + 273.15))
    t_now = 0.0
    out = {k: [] for k in ("t", "cc", "ca", "vo", "vc", "va", "c")}

    def surface(conc):
        cc = conc[0] - flux * 0.5 * w[0] / float(el.D(conc[0], T))
        ca = conc[-1] + flux * 0.5 * w[-1] / float(el.D(conc[-1], T))
        return cc, ca

    def record(conc, t):
        cc, ca = surface(conc)
        cc_v = max(cc, 1e-6 * c0)
        v_ohm = J * np.sum(w / el.kappa(np.clip(conc, 1e-6 * c0, None), T))
        v_conc = 2.0 * ta / (Z * f) * np.log(ca / cc_v)
        eta_a = _activation_overpotential(J, float(el.i0(ca)), el.alpha_a, el.alpha_c, f)
        eta_c = _activation_overpotential(-J, float(el.i0(cc_v)), el.alpha_a, el.alpha_c, f)
        for k, v in zip(("t", "cc", "ca", "vo", "vc", "va", "c"),
                        (t, cc, ca, v_ohm, v_conc, eta_a - eta_c, conc.copy())):
            out[k].append(v)
        return cc

    record(c, 0.0)
    tau = np.inf
    while t_now < t_end:
        c_old, c_guess = c, c
        for _ in range(2):                       # Picard iteration on D(c)
            Dc = el.D(c_guess, T) * mix
            Df = 2.0 * Dc[1:] * Dc[:-1] / (Dc[1:] + Dc[:-1]) / d      # face conductances
            ab = np.zeros((3, n))
            ab[1] = w / dt
            ab[1, :-1] += Df
            ab[1, 1:] += Df
            ab[0, 1:] = -Df
            ab[2, :-1] = -Df
            rhs = w / dt * c_old
            rhs[0] -= flux
            rhs[-1] += flux
            c_guess = solve_banded((1, 1), ab, rhs)
        c = c_guess
        t_now += dt
        cc_prev = out["cc"][-1]
        cc = record(c, t_now)
        if cc <= 0.0:                            # Sand's time: interpolate the zero crossing
            tau = t_now - dt * (0.0 - cc) / (cc_prev - cc)
            break
        dt = min(dt * 1.08, dt_max)

    t = np.array(out["t"])
    v_o, v_c, v_a = (np.array(out[k]) for k in ("vo", "vc", "va"))
    v_total = np.minimum(v_o + v_c + v_a + cell.current * cell.series_resistance, v_compliance)
    if np.isfinite(tau):
        v_total[-1] = v_compliance               # depleted: the supply rails
    return Result(t=t, c_cathode=np.array(out["cc"]), c_anode=np.array(out["ca"]),
                  V_ohmic=v_o, V_concentration=v_c, V_activation=v_a, V=v_total,
                  x=x, c=np.array(out["c"]), tau=tau, delta=delta, cell=cell, g=g)


def sand_time_uncertainty(cell, D_rel=0.08, t_cation_abs=0.04, T_abs=0.5, area_rel=0.03,
                          current_rel=0.01, c0_rel=0.01, n=4000, seed=0):
    """Monte-Carlo spread of the predicted Sand's time from input uncertainties.

    Defaults: D known to 8%, t_cation to +/-0.04, temperature to 0.5 C, cathode area
    to 3%, current and concentration to 1% (all one standard deviation).
    Returns (median, 16th percentile, 84th percentile) in seconds.
    """
    rng = np.random.default_rng(seed)
    el = cell.electrolyte
    D = cell.D0 * (1 + D_rel * rng.standard_normal(n))
    T = cell.T_C + T_abs * rng.standard_normal(n)
    D = D * el.D(cell.c0, T) / cell.D0
    ta = 1.0 - (el.t_cation + t_cation_abs * rng.standard_normal(n))
    J = cell.J * (1 + current_rel * rng.standard_normal(n)) / (1 + area_rel * rng.standard_normal(n))
    c0 = cell.c0 * (1 + c0_rel * rng.standard_normal(n))
    tau = analytic.sand_time(J, c0, D, ta, el.z)
    return tuple(np.percentile(tau, [50, 16, 84]))
