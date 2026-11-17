"""2-D current distribution in the chamber cross-section, for choosing electrode shapes.

Sand's time depends on the *local* current density squared, so an electrode
arrangement that concentrates current at edges depletes there first and smears
the voltage transition the experiment is trying to time.  This module solves
Laplace's equation for the potential in a top-down slice of the chamber
(electrodes assumed to run the full liquid height) and reports

  * the cell resistance, hence the voltage the current source must supply,
  * how uniform the current density is over the cathode,
  * the resulting spread of local Sand's times.

Electrode kinetics are included as a linearised surface resistance (a "secondary"
current distribution), which removes the unphysical infinite current density that
pure Laplace gives at sharp electrode edges.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .properties import F, R_GAS, Electrolyte

FLUID, ANODE, CATHODE, INSULATOR = 0, 1, 2, 3


@dataclass
class Chamber:
    """Interior of the liquid chamber seen from above.  x = width, y = depth."""
    width: float = 20e-3
    depth: float = 20e-3
    height: float = 20e-3          # liquid height = immersed electrode length
    h: float = 0.1e-3              # grid spacing

    def grid(self):
        nx, ny = int(round(self.width / self.h)), int(round(self.depth / self.h))
        x = (np.arange(nx) + 0.5) * self.h
        y = (np.arange(ny) + 0.5) * self.h
        return np.meshgrid(x, y, indexing="ij")


# --- electrode layouts ---------------------------------------------------------
# Each builder returns an integer map of the cross-section (FLUID/ANODE/CATHODE/INSULATOR).
# The cathode sits near y = 0 and the anode towards y = depth.

def _rod(X, Y, cx, cy, dia):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= (dia / 2.0) ** 2


def rod_and_free_strip(ch, gap=10e-3, cathode_width=13.8e-3, thickness=0.4e-3, rod_dia=9.53e-3):
    """Team C's flight layout: round rod anode, thin strip cathode standing free in
    the liquid so both faces and both edges are wetted."""
    X, Y = ch.grid()
    m = np.zeros(X.shape, dtype=int)
    y_c = (ch.depth - gap - rod_dia - thickness) / 2.0 + thickness / 2.0
    m[(np.abs(X - ch.width / 2) <= cathode_width / 2) & (np.abs(Y - y_c) <= thickness / 2)] = CATHODE
    m[_rod(X, Y, ch.width / 2, y_c + thickness / 2 + gap + rod_dia / 2, rod_dia)] = ANODE
    return m


def rod_and_wall_strip(ch, gap=10e-3, cathode_width=13.8e-3, rod_dia=9.53e-3):
    """Rod anode; strip cathode flat against the chamber wall (back face insulated)."""
    X, Y = ch.grid()
    m = np.zeros(X.shape, dtype=int)
    m[(np.abs(X - ch.width / 2) <= cathode_width / 2) & (Y < ch.h)] = CATHODE
    m[_rod(X, Y, ch.width / 2, ch.h + gap + rod_dia / 2, rod_dia)] = ANODE
    return m


def parallel_plates_partial(ch, gap=10e-3, cathode_width=10e-3, anode_width=10e-3):
    """Two facing plates narrower than the chamber, backs insulated, edges exposed."""
    X, Y = ch.grid()
    m = np.zeros(X.shape, dtype=int)
    m[Y > gap + ch.h] = INSULATOR
    m[(np.abs(X - ch.width / 2) <= cathode_width / 2) & (Y < ch.h)] = CATHODE
    m[(np.abs(X - ch.width / 2) <= anode_width / 2) & (Y > gap + ch.h) & (Y < gap + 2 * ch.h)] = ANODE
    return m


def channel_cell(ch, gap=5e-3, channel_width=10e-3):
    """Recommended: both electrodes close off the ends of a straight insulating
    channel, so every current line is parallel and J is the same everywhere."""
    X, Y = ch.grid()
    m = np.full(X.shape, INSULATOR, dtype=int)
    inside = np.abs(X - ch.width / 2) <= channel_width / 2
    m[inside & (Y > ch.h) & (Y < gap + ch.h)] = FLUID
    m[inside & (Y < ch.h)] = CATHODE
    m[inside & (Y > gap + ch.h) & (Y < gap + 2 * ch.h)] = ANODE
    return m


def recessed_cathode_rod_anode(ch, recess=5e-3, cathode_width=10e-3, gap=10e-3, rod_dia=9.53e-3):
    """Cathode at the bottom of an insulating slot, rod anode in the open chamber
    beyond it.  The slot straightens the current lines before they reach the cathode."""
    X, Y = ch.grid()
    m = np.zeros(X.shape, dtype=int)
    in_slot = np.abs(X - ch.width / 2) <= cathode_width / 2
    m[(Y < recess + ch.h) & ~in_slot] = INSULATOR
    m[in_slot & (Y < ch.h)] = CATHODE
    m[_rod(X, Y, ch.width / 2, ch.h + gap + rod_dia / 2, rod_dia)] = ANODE
    return m


GEOMETRIES = {
    "Rod anode + free-standing strip (Team C layout)": rod_and_free_strip,
    "Rod anode + strip on wall": rod_and_wall_strip,
    "Facing plates, edges exposed": parallel_plates_partial,
    "Rod anode + recessed cathode": recessed_cathode_rod_anode,
    "Channel cell (plates fill the channel)": channel_cell,
}


# --- solver --------------------------------------------------------------------
@dataclass
class CurrentDistribution:
    layout: np.ndarray            # the electrode map that was solved
    phi: np.ndarray               # potential in the liquid, V (cathode = 0)
    voltage: float                # anode-to-cathode voltage at the requested current, V
    resistance_solution: float    # ohmic resistance of the liquid alone, ohm
    j_cathode: np.ndarray         # local current density on each cathode face, A/m^2
    face_xy: np.ndarray           # (n, 2) positions of those faces, m
    cathode_area: float           # wetted cathode area, m^2
    chamber: Chamber = None

    @property
    def j_mean(self):
        return float(self.j_cathode.mean())

    @property
    def j_relative(self):
        return self.j_cathode / self.j_mean

    def uniformity(self):
        """Summary statistics of local current density relative to the mean."""
        r = self.j_relative
        return {"min": float(r.min()), "p5": float(np.percentile(r, 5)),
                "p95": float(np.percentile(r, 95)), "max": float(r.max()),
                "within_5pct": float(np.mean(np.abs(r - 1.0) <= 0.05))}

    def sand_time_spread(self):
        """Local Sand's time / Sand's time at the mean current density, at the
        instants 5 %, 50 % and 95 % of the cathode area has depleted.  (Treats every
        patch of cathode independently - a guide to how smeared the transition is.)"""
        ratio = np.sort(1.0 / self.j_relative ** 2)
        return {"first_5pct": float(np.percentile(ratio, 5)), "median": float(np.percentile(ratio, 50)),
                "last_5pct": float(np.percentile(ratio, 95))}


def _solve(layout, kappa, h, r_anode, r_cathode):
    """Solve for phi with the anode at 1 V and the cathode at 0 V.  Returns phi and
    the per-face cathode currents (A per metre of height)."""
    nx, ny = layout.shape
    fluid = layout == FLUID
    index = -np.ones(layout.shape, dtype=int)
    index[fluid] = np.arange(fluid.sum())
    padded = np.pad(layout, 1, constant_values=INSULATOR)
    g_a = h / (h / (2.0 * kappa) + r_anode)       # liquid half-cell + surface resistance
    g_c = h / (h / (2.0 * kappa) + r_cathode)

    rows, cols, vals = [], [], []
    diag = np.zeros(fluid.sum())
    rhs = np.zeros(fluid.sum())
    faces = []
    I, J = np.nonzero(fluid)
    me = index[I, J]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = padded[I + 1 + di, J + 1 + dj]
        f = nb == FLUID
        rows.append(me[f]); cols.append(index[I[f] + di, J[f] + dj]); vals.append(np.full(f.sum(), -kappa))
        np.add.at(diag, me[f], kappa)
        a = nb == ANODE
        np.add.at(diag, me[a], g_a)
        np.add.at(rhs, me[a], g_a * 1.0)
        c = nb == CATHODE
        np.add.at(diag, me[c], g_c)
        faces.append((me[c], I[c] + 0.5 * di, J[c] + 0.5 * dj))
    rows.append(np.arange(len(diag))); cols.append(np.arange(len(diag))); vals.append(diag)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)))).tocsr()
    sol = spsolve(A, rhs)
    phi = np.full(layout.shape, np.nan)
    phi[fluid] = sol
    face_cell = np.concatenate([f[0] for f in faces])
    face_ij = np.column_stack([np.concatenate([f[1] for f in faces]), np.concatenate([f[2] for f in faces])])
    return phi, g_c * sol[face_cell], face_ij


def current_distribution(chamber, layout, current, c0, T_C=25.0, electrolyte=None):
    """Current distribution for a given electrode layout carrying `current` amps."""
    el = electrolyte or Electrolyte()
    h = chamber.h
    kappa = float(el.kappa(c0, T_C))
    thermal = R_GAS * (T_C + 273.15) / F
    i0 = float(el.i0(c0))

    # primary distribution first: gives the solution resistance and the wetted areas
    _, face_i, face_ij = _solve(layout, kappa, h, 0.0, 0.0)
    cathode_area = len(face_i) * h * chamber.height
    r_solution = 1.0 / (face_i.sum() * chamber.height)

    # linearised kinetics about the operating point (Tafel slope at high J, linear at low J)
    j_c = current / cathode_area
    n_anode_faces = _count_faces(layout, ANODE)
    j_a = current / (n_anode_faces * h * chamber.height)
    r_c = thermal / ((el.alpha_a + el.alpha_c) * i0 + el.alpha_c * j_c)
    r_a = thermal / ((el.alpha_a + el.alpha_c) * i0 + el.alpha_a * j_a)
    phi, face_i, face_ij = _solve(layout, kappa, h, r_a, r_c)

    volts = current / (face_i.sum() * chamber.height)        # scale the 1 V solution
    return CurrentDistribution(layout=layout, phi=phi * volts, voltage=volts,
                               resistance_solution=r_solution,
                               j_cathode=face_i * volts / h, face_xy=(face_ij + 0.5) * h,
                               cathode_area=cathode_area, chamber=chamber)


def _count_faces(layout, kind):
    padded = np.pad(layout, 1, constant_values=INSULATOR)
    fluid = padded == FLUID
    n = 0
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n += np.sum(fluid[1:-1, 1:-1] & (np.roll(np.roll(padded, -di, 0), -dj, 1)[1:-1, 1:-1] == kind))
    return int(n)
