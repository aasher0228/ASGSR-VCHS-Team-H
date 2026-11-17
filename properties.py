"""Physical properties of binary electrolytes: copper sulfate (CuSO4, 2:2) and
silver nitrate (AgNO3, 1:1).

All quantities are SI: concentration in mol/m^3 (1 M = 1000 mol/m^3), temperature
in deg C at the function boundary, diffusivity in m^2/s, conductivity in S/m.

Use the factories:
    copper_sulfate()   - Team H's planned cell (also what Electrolyte() gives)
    silver_nitrate(c0) - Team C's flown cell; the transference number is read off
                         the team's property table at the bulk concentration c0
Any field can be overridden to test sensitivity, e.g. silver_nitrate(1000, D_scale=1.1).

Property tables are the largest source of uncertainty in a Sand's-time prediction
(tau scales as D / t_a^2).  Outside a table's range values are extrapolated
linearly in sqrt(c) - treat those as estimates (Electrolyte.in_table tells you).
"""
from dataclasses import dataclass
import numpy as np

F = 96485.332          # Faraday constant, C/mol
R_GAS = 8.314462       # J/(mol K)
Z = 2                  # charge number of Cu2+ (kept for older code; use Electrolyte.z)
M_CU = 63.546e-3       # kg/mol
RHO_CU = 8960.0        # kg/m^3
G0 = 9.80665           # m/s^2
BARD_CONVECTION_SCALE = 0.76   # measured / predicted convection threshold, Bard (1961) (0.75 vertical, 0.77 facing up)

# --- copper sulfate, 25 C ------------------------------------------------------
# Mutual (salt) diffusion coefficient; c = 0 is the Nernst limit from ionic
# conductances.  Molar conductivity from equivalent-conductance tables (x2).
CUSO4_D = ((0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1400.0),
           tuple(v * 1e-10 for v in (8.55, 7.0, 6.2, 5.8, 5.2, 4.7, 4.2, 3.8)))
CUSO4_LAMBDA = ((0.0, 0.25, 0.5, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1400.0),
                tuple(v * 2e-4 for v in (133.6, 121.6, 115.3, 94.1, 83.1, 72.2, 59.1, 50.6, 42.5, 34.0, 28.0, 22.3, 19.0)))

# --- silver nitrate, 25 C ------------------------------------------------------
# 0.1-1.0 M rows: Team H "Silver Nitrate Property Data Table" (diffusion: doi
# 10.1021/j100657a012; conductance and transference: Campbell et al., concentrated
# AgNO3/AgClO4 at 25 C; MacInnes 1932).  c = 0 row: limiting ionic conductances
# lambda(Ag+) = 61.9, lambda(NO3-) = 71.42 S cm^2/mol (CRC Handbook), which give
# D = 2 D+ D- / (D+ + D-) = 1.766e-9 m^2/s and t+ = 0.464.
AGNO3_C = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)
AGNO3_D = (AGNO3_C, tuple(v * 1e-9 for v in (1.766, 1.598, 1.547, 1.508, 1.476, 1.448, 1.422, 1.399, 1.377, 1.356, 1.336)))
AGNO3_LAMBDA = (AGNO3_C, tuple(v * 1e-4 for v in (133.3, 109.2, 101.9, 96.8, 92.8, 89.5, 86.6, 84.1, 81.9, 79.9, 78.1)))
AGNO3_T_CATION = (AGNO3_C, (0.464, 0.4682, 0.4718, 0.4753, 0.4788, 0.4822, 0.4856, 0.489, 0.4923, 0.4956, 0.4989))
M_AGNO3 = 169.87e-3    # kg/mol
M_AG = 107.868e-3      # kg/mol
RHO_AG = 10490.0       # kg/m^3


def water_viscosity(T_C):
    """Dynamic viscosity of water, Pa s (Vogel equation, good to ~1% for 0-60 C)."""
    T_K = np.asarray(T_C, dtype=float) + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))


def _temperature_factor(T_C):
    """Stokes-Einstein scaling of ion mobility relative to 25 C: (T/T25) * eta25/eta(T)."""
    return (np.asarray(T_C, dtype=float) + 273.15) / 298.15 * water_viscosity(25.0) / water_viscosity(T_C)


def _interp_sqrt(c, table, floor_frac=0.2):
    """Interpolate a table linearly in sqrt(c); extrapolate beyond the last row along
    the last segment (never below floor_frac of the last value)."""
    xs, ys = np.sqrt(np.asarray(table[0])), np.asarray(table[1])
    s = np.sqrt(np.clip(np.asarray(c, dtype=float), 0.0, None))
    out = np.interp(s, xs, ys)
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    beyond = s > xs[-1]
    return np.where(beyond, np.maximum(ys[-1] + slope * (s - xs[-1]), floor_frac * ys[-1]), out)


@dataclass
class Electrolyte:
    """Binary z:z salt solution.  Defaults are copper sulfate.

    t_cation: metal-ion transference number (share of the current carried by the
        metal ion).  Published values for 0.1-1 M CuSO4 are about 0.30-0.40.
    D_scale / kappa_scale: multiply the tabulated curves (use for calibration).
    D_constant: if set, use this single diffusivity (m^2/s at 25 C) instead of the
        concentration-dependent table - the assumption behind the textbook formula.
    density_slope: d(rho)/dc, kg/m^3 per mol/m^3 - drives buoyant convection.
    convection_scale: multiplies the Sherwood number of the convection correlations.
        1.0 = as published; ~0.76 reproduces Bard (1961)'s measured threshold
        (see validate_bard1961.py and BARD_CONVECTION_SCALE).
    """
    name: str = "CuSO4"
    z: int = 2
    t_cation: float = 0.36
    D_scale: float = 1.0
    kappa_scale: float = 1.0
    D_constant: float = None
    D_table: tuple = CUSO4_D
    lambda_table: tuple = CUSO4_LAMBDA
    density_slope: float = 0.156
    viscosity_coeffs: tuple = (0.53, 0.30)     # eta/eta_water = 1 + a cM + b cM^2
    convection_scale: float = 1.0
    metal_molar_mass: float = M_CU
    metal_density: float = RHO_CU
    # Butler-Volmer kinetics (Cu2+ + 2e- <-> Cu by default, two-step mechanism)
    i0_ref: float = 10.0        # exchange current density at c_ref, A/m^2
    c_ref: float = 100.0        # mol/m^3
    i0_exponent: float = 0.5
    alpha_a: float = 1.5
    alpha_c: float = 0.5

    @property
    def t_anion(self):
        return 1.0 - self.t_cation

    def in_table(self, c):
        """True if c lies inside the measured range of both property tables."""
        return float(c) <= min(self.D_table[0][-1], self.lambda_table[0][-1])

    def D(self, c, T_C=25.0):
        """Salt diffusion coefficient, m^2/s."""
        c = np.clip(np.asarray(c, dtype=float), 0.0, None)
        if self.D_constant is not None:
            base = np.full_like(c, self.D_constant)
        else:
            base = _interp_sqrt(c, self.D_table)
        return base * self.D_scale * _temperature_factor(T_C)

    def kappa(self, c, T_C=25.0):
        """Ionic conductivity, S/m.  Goes to zero linearly as c -> 0."""
        c = np.clip(np.asarray(c, dtype=float), 0.0, None)
        return c * _interp_sqrt(c, self.lambda_table) * self.kappa_scale * _temperature_factor(T_C)

    def i0(self, c):
        """Exchange current density, A/m^2."""
        c = np.clip(np.asarray(c, dtype=float), 1e-9, None)
        return self.i0_ref * (c / self.c_ref) ** self.i0_exponent

    def density(self, c):
        """Solution density, kg/m^3 (linear in concentration)."""
        return 997.0 + self.density_slope * np.asarray(c, dtype=float)

    def densification(self):
        """(1/rho) d rho / d c, m^3/mol - drives buoyant convection."""
        return self.density_slope / 997.0

    def viscosity(self, c, T_C=25.0):
        """Dynamic viscosity of the solution, Pa s."""
        cM = np.asarray(c, dtype=float) / 1000.0
        a, b = self.viscosity_coeffs
        return water_viscosity(T_C) * (1.0 + a * cM + b * cM ** 2)


def copper_sulfate(**overrides):
    """CuSO4 (Team H's planned cell)."""
    return Electrolyte(**overrides)


def silver_nitrate(c0=1000.0, **overrides):
    """AgNO3 (Team C's flown cell).  t_cation defaults to the table value at c0.

    Ag+ + e- <-> Ag is a fast one-electron reaction: i0 of order 1e3 A/m^2 at 1 M,
    so activation overpotential is negligible at Team C's few A/m^2.
    density_slope 0.142 kg/m^3 per mol/m^3 matches CRC densities up to ~1 M (0.64 M
    at 10 wt% has rho = 1.089 g/mL); it falls to ~0.133 near 5 M.
    Relative viscosity 1 + 0.06 cM is an estimate (AgNO3 barely thickens water).
    """
    kw = dict(name="AgNO3", z=1, t_cation=float(_interp_sqrt(c0, AGNO3_T_CATION)),
              D_table=AGNO3_D, lambda_table=AGNO3_LAMBDA, density_slope=0.142,
              viscosity_coeffs=(0.06, 0.0), metal_molar_mass=M_AG, metal_density=RHO_AG,
              i0_ref=1000.0, c_ref=1000.0, i0_exponent=0.5, alpha_a=0.5, alpha_c=0.5)
    kw.update(overrides)
    return Electrolyte(**kw)


def molarity_from_recipe(salt_g, water_g, molar_mass=M_AGNO3, density=None):
    """mol/m^3 of a solution made from salt_g of salt in water_g of water.

    density (kg/m^3) of the final solution; if None it is found self-consistently
    from the silver-nitrate density_slope (an overestimate above ~3 M).
    """
    w = salt_g / (salt_g + water_g)
    if density is not None:
        return w * density / molar_mass
    c = 0.0
    for _ in range(50):
        c = w * (997.0 + 0.142 * c) / molar_mass
    return c
