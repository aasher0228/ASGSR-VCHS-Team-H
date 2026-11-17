"""Run the model for the Team H copper experiment and write figures 1-4, 6 + tables to results/.

Team C's silver flight data: team_c_silver.py (figure 5).  Published-data check:
validate_bard1961.py (figure 7).

    python make_report.py

Edit the BASELINE block to try a different design; everything else follows from it.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import binary_dilation

from sands_model import (Cell, Chamber, Electrolyte, GEOMETRIES, G0, analytic,
                         current_distribution, sand_time_uncertainty, simulate)
from sands_model.geometry import ANODE, CATHODE, INSULATOR

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- baseline
EL = Electrolyte()
BASELINE = dict(c0=250.0,             # 0.25 M CuSO4
                gap=5e-3,             # 5 mm between electrodes
                cathode_area=0.5e-4,  # 5 mm x 10 mm cathode
                height=10e-3,         # electrodes are 10 mm tall
                current=10e-3,        # 10 mA
                T_C=25.0)
CURRENT_BUDGET = 20e-3     # A available for the cell (MicroLab total is 100 mA; Team C's unflown copper circuit was 18.7 mA)
V_COMPLIANCE = 3.3         # V the constant-current source can reach (5 V rail minus regulator headroom)
V_START_MAX = 1.5          # keep the starting voltage below this so the spike is unmistakable
TAU_WINDOW = (30.0, 120.0)  # s, target range for Sand's time

# ----------------------------------------------------------------------------- style
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blues", BLUES)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 160,
})


TEAM_C_CREDIT = "Flight and ground data: ISS Team C 2024-25, Valley Christian High School AMSE ISS Research Lab"
BARD_CREDIT = "Data: A. J. Bard, Anal. Chem. 33, 11-15 (1961), Table I"
MODEL_CREDIT = "Model: ISS Team H 2026-27"


def save(fig, name, credit=None):
    if credit:
        fig.text(0.99, -0.075, credit, ha="right", va="top", fontsize=8, color=INK2, style="italic")
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def fmt_time(s):
    if not np.isfinite(s):
        return "never"
    return f"{s:.0f} s" if s < 600 else (f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h")


def variable_D_factor(c0, T_C=25.0):
    """Ratio of the simulated Sand's time (D varies with c) to the textbook formula
    evaluated with D at the bulk concentration.  Independent of J for a wide gap."""
    cell = Cell(c0=c0, gap=20e-3, cathode_area=1e-4, current=0.02 * c0 / 100.0, T_C=T_C, electrolyte=EL)
    return simulate(cell).tau / cell.sand_time_ideal()


# ----------------------------------------------------------------------------- 1. baseline transient
def figure_baseline():
    cell = Cell(electrolyte=EL, **BASELINE)
    micro = simulate(cell, v_compliance=V_COMPLIANCE)
    t_show = 1.35 * micro.tau
    ground = simulate(cell, g=G0, orientation="vertical", v_compliance=V_COMPLIANCE, t_end=t_show)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})
    t_m = np.append(micro.t, t_show)
    v_m = np.append(micro.V, V_COMPLIANCE)
    ax1.plot(t_m, v_m, color=BLUE, label="Microgravity (ISS)")
    ax1.plot(ground.t, ground.V, color=ORANGE, label="Earth, electrodes vertical")
    ax1.axvline(micro.tau, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax1.text(micro.tau - 0.02 * t_show, 0.55 * V_COMPLIANCE, f"Sand's time\n{micro.tau:.0f} s", ha="right", color=INK)
    ax1.text(t_show, V_COMPLIANCE + 0.06, "supply limit", ha="right", va="bottom", color=INK2, fontsize=9)
    ax1.text(t_show, ground.V[-1] - 0.1, "Earth: convection keeps\nfeeding the cathode", ha="right", va="top", color=INK2, fontsize=9)
    ax1.set(xlabel="Time since current switched on (s)", ylabel="Cell voltage (V)", xlim=(0, t_show),
            ylim=(0, V_COMPLIANCE + 0.45), title="Predicted voltage trace")
    ax1.legend(loc="upper left")

    fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    x_mm = micro.x * 1e3
    for frac, col in zip(fracs, BLUES[2:]):
        k = np.searchsorted(micro.t, frac * micro.tau)
        k = min(k, len(micro.t) - 1)
        ax2.plot(x_mm, micro.c[k] / 1000.0, color=col, label=f"{frac * micro.tau:.0f} s")
    ax2.plot(ground.x * 1e3, ground.c[-1] / 1000.0, color=ORANGE, label="Earth, steady state")
    ax2.legend(loc="lower right", title="Microgravity, time since start", alignment="left")
    ax2.set(xlabel="Distance from cathode (mm)", ylabel="CuSO$_4$ concentration (M)", xlim=(0, 1.0),
            ylim=(0, 1.12 * cell.c0 / 1000.0), title="Copper depletion next to the cathode")
    fig.suptitle(f"Baseline cell: {cell.c0 / 1000:g} M CuSO$_4$, {cell.current * 1e3:g} mA on a "
                 f"{cell.cathode_area * 1e4:g} cm$^2$ cathode ({cell.J:.0f} A/m$^2$), {cell.gap * 1e3:g} mm gap",
                 x=0.125, ha="left", fontsize=11, color=INK2, y=1.0)
    save(fig, "1_baseline_voltage_and_depletion.png")
    return cell, micro, ground


# ----------------------------------------------------------------------------- 2. tau vs J
def figure_tau_vs_current(factors):
    concs = [100.0, 250.0, 500.0, 1000.0]
    J = np.logspace(1, 3.3, 140)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, c0 in zip(axes, concs):
        D_eff = float(EL.D(c0)) * np.interp(c0, factors[0], factors[1])
        ta = EL.t_anion
        delta = analytic.convective_layer_thickness(G0, c0, BASELINE["height"], electrolyte=EL)
        tau0 = np.array([analytic.sand_time_finite_cell(j, c0, D_eff, ta, BASELINE["gap"]) for j in J])
        tau1 = np.array([analytic.sand_time_finite_cell(j, c0, D_eff, ta, analytic.effective_gap(BASELINE["gap"], delta)) for j in J])
        ax.axhspan(*TAU_WINDOW, color=GRID, alpha=0.7, lw=0)
        ax.plot(J, tau0, color=BLUE, label="Microgravity")
        ax.plot(J[np.isfinite(tau1)], tau1[np.isfinite(tau1)], color=ORANGE, label="Earth, electrodes vertical")
        j_lim = analytic.diffusion_limited_current(c0, D_eff, ta, 2 * delta)
        ax.axvline(j_lim, color=ORANGE, lw=0.8, ls=(0, (4, 3)))
        ax.text(j_lim * 0.93, 2.2e4, f"Earth: no dendrites\nbelow {j_lim:.0f} A/m$^2$", color=INK2, fontsize=8, ha="right", va="top")
        ax.set(xscale="log", yscale="log", xlim=(J[0], J[-1]), ylim=(1, 3e4), title=f"{c0 / 1000:g} M CuSO$_4$")
        ax.set_xlabel("Current density (A/m$^2$)")
    axes[0].set_ylabel("Time to deplete the cathode (s)")
    axes[0].text(11, TAU_WINDOW[0] * 1.15, "target window", color=INK2, fontsize=8, va="bottom")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Sand's time against current density, {BASELINE['gap'] * 1e3:g} mm gap, {BASELINE['height'] * 1e3:g} mm tall electrodes",
                 x=0.125, ha="left", fontsize=11, color=INK2, y=1.04)
    save(fig, "2_sand_time_vs_current_density.png")


# ----------------------------------------------------------------------------- 3. design map
def start_voltage(c0, J, gap, T_C=25.0):
    """Voltage the instant current is applied: ohmic drop + activation at bulk concentration."""
    from sands_model.transport import _activation_overpotential
    from sands_model.properties import F, R_GAS
    f = F / (R_GAS * (T_C + 273.15))
    i0 = float(EL.i0(c0))
    eta = _activation_overpotential(J, i0, EL.alpha_a, EL.alpha_c, f) - _activation_overpotential(-J, i0, EL.alpha_a, EL.alpha_c, f)
    return J * gap / float(EL.kappa(c0, T_C)) + eta


def figure_design_map(factors):
    I = BASELINE["current"]
    area = np.logspace(np.log10(0.1e-4), np.log10(3e-4), 90)
    conc = np.logspace(np.log10(50.0), np.log10(1000.0), 80)
    A, C = np.meshgrid(area, conc)
    J = I / A
    D_eff = EL.D(C) * np.interp(C, factors[0], factors[1])
    tau = analytic.sand_time(J, C, D_eff, EL.t_anion)
    v0 = np.vectorize(lambda c, j: start_voltage(c, j, BASELINE["gap"]))(C, J)
    j_lim = np.vectorize(lambda c: analytic.convective_limiting_current(G0, c, BASELINE["height"], electrolyte=EL))(C)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    mesh = ax.pcolormesh(A * 1e4, C / 1000.0, tau, norm=LogNorm(1, 1e5), cmap=SEQ, shading="auto", rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Sand's time in microgravity (s)")
    cb.outline.set_visible(False)
    ax.contour(A * 1e4, C / 1000.0, tau, levels=list(TAU_WINDOW), colors=INK, linewidths=1.4)
    ax.contour(A * 1e4, C / 1000.0, v0, levels=[V_START_MAX], colors=INK2, linewidths=1.4, linestyles=[(0, (5, 3))])
    ax.contour(A * 1e4, C / 1000.0, J / j_lim, levels=[1.0], colors=ORANGE, linewidths=2.0)

    def tag(text, field, level, at_conc):
        # put a boxed label on the contour `field == level` at concentration `at_conc`
        row = np.argmin(np.abs(conc - at_conc))
        col = np.argmin(np.abs(np.log(field[row] / level)))
        ax.text(area[col] * 1e4, conc[row] / 1000.0, text, fontsize=9, color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID, lw=0.8))
    tag("Sand's time 30 s", tau, TAU_WINDOW[0], 600.0)
    tag("Sand's time 120 s", tau, TAU_WINDOW[1], 600.0)
    tag(f"starts at {V_START_MAX:g} V", v0, V_START_MAX, 80.0)
    tag("Earth unit depletes too", J / j_lim, 1.0, 130.0)
    ax.plot(BASELINE["cathode_area"] * 1e4, BASELINE["c0"] / 1000.0, "o", ms=9, mfc="white", mec=INK, mew=1.6)
    ax.annotate("baseline", (BASELINE["cathode_area"] * 1e4, BASELINE["c0"] / 1000.0), xytext=(9, -4),
                textcoords="offset points", color=INK, fontsize=10, fontweight="bold")
    ax.set(xscale="log", yscale="log", xlabel="Cathode area (cm$^2$)", ylabel="CuSO$_4$ concentration (M)",
           title=f"Choosing cathode area and concentration at {I * 1e3:g} mA")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "3"])
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1]); ax.set_yticklabels(["0.05", "0.1", "0.25", "0.5", "1"])
    ax.grid(False)
    fig.text(0.125, -0.02, "Aim between the two black lines (30-120 s).  Left of the orange line the ground unit would deplete too;\n"
             "left of the dashed line the cell starts above 1.5 V and leaves little room to see the spike.",
             color=INK2, fontsize=9, va="top")
    save(fig, "3_design_map.png")


# ----------------------------------------------------------------------------- 4. electrode shapes
def figure_geometry():
    ch = Chamber(width=20e-3, depth=20e-3, height=BASELINE["height"], h=0.1e-3)
    w = 5e-3
    layouts = {
        "Rod + free strip\n(Team C layout)": GEOMETRIES["Rod anode + free-standing strip (Team C layout)"](ch, gap=5e-3, cathode_width=w),
        "Rod anode +\nstrip on wall": GEOMETRIES["Rod anode + strip on wall"](ch, gap=5e-3, cathode_width=w),
        "Facing plates,\nedges exposed": GEOMETRIES["Facing plates, edges exposed"](ch, gap=5e-3, cathode_width=w, anode_width=w),
        "Rod anode +\nrecessed cathode": GEOMETRIES["Rod anode + recessed cathode"](ch, recess=5e-3, cathode_width=w, gap=7e-3),
        "Channel cell": GEOMETRIES["Channel cell (plates fill the channel)"](ch, gap=5e-3, channel_width=w),
    }
    rows = []
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.12)
    ax_bar = fig.add_subplot(gs[1, :])
    for k, (name, layout) in enumerate(layouts.items()):
        dist = current_distribution(ch, layout, BASELINE["current"], BASELINE["c0"], BASELINE["T_C"], EL)
        u, s = dist.uniformity(), dist.sand_time_spread()
        rows.append((name.replace("\n", " "), dist, u, s))
        ax = fig.add_subplot(gs[0, k])
        ext = (0, ch.width * 1e3, 0, ch.depth * 1e3)
        shade = np.where(layout == INSULATOR, 1.0, np.nan)
        ax.imshow(shade.T, origin="lower", extent=ext, cmap=LinearSegmentedColormap.from_list("g", [GRID, GRID]), vmin=0, vmax=1)
        ax.contour(dist.phi.T, levels=np.linspace(0, dist.voltage, 13)[1:-1], extent=ext, colors=BLUES[3], linewidths=0.7)
        for kind, col in ((ANODE, ORANGE), (CATHODE, INK)):
            m = np.where(binary_dilation(layout == kind, iterations=2), 1.0, np.nan)   # thickened so plates show
            ax.imshow(m.T, origin="lower", extent=ext, cmap=LinearSegmentedColormap.from_list("e", [col, col]), vmin=0, vmax=1,
                      interpolation="nearest")
        ax.set(xticks=[], yticks=[], title=name)
        ax.title.set_fontsize(9.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
        ax.text(0.5, -0.04, f"{dist.cathode_area * 1e4:.2f} cm$^2$ wetted, {dist.voltage:.2f} V", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=INK2)
        # range chart: local Sand's time relative to the uniform-current value
        y = len(layouts) - 1 - k
        ax_bar.plot([s["first_5pct"], s["last_5pct"]], [y, y], color=BLUE, lw=5, solid_capstyle="round")
        ax_bar.plot(s["median"], y, "o", ms=8, mfc=SURFACE, mec=BLUE, mew=2)
        ax_bar.text(max(s["last_5pct"], 1.0) * 1.07, y, f"{100 * u['within_5pct']:.0f}% of cathode within 5% of mean current",
                    va="center", fontsize=9, color=INK2)
    ax_bar.axvline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax_bar.set(yticks=range(len(layouts)), yticklabels=[n.replace("\n", " ") for n in layouts][::-1], xscale="log",
               xlim=(0.08, 40), xlabel="Local Sand's time / Sand's time at the mean current density  (bar: when 5% to 95% of the cathode has depleted; circle: half)",
               title="How much each layout smears the transition")
    ax_bar.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10]); ax_bar.set_xticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x"])
    ax_bar.grid(axis="y", visible=False)
    fig.suptitle(f"Electrode layouts seen from above (cathode black, anode orange, insulator grey, lines = equipotentials), "
                 f"{BASELINE['current'] * 1e3:g} mA in {BASELINE['c0'] / 1000:g} M", x=0.125, ha="left", fontsize=11, color=INK2, y=0.975)
    save(fig, "4_electrode_layouts.png")
    return rows


# ----------------------------------------------------------------------------- 6. uncertainty budget
def figure_uncertainty(cell):
    base = dict(D_rel=0.0, t_cation_abs=0.0, T_abs=0.0, area_rel=0.0, current_rel=0.0, c0_rel=0.0)
    items = [("Cu$^{2+}$ transference number (0.36 $\\pm$ 0.04)", dict(t_cation_abs=0.04)),
             ("Diffusion coefficient ($\\pm$ 8%)", dict(D_rel=0.08)),
             ("Wetted cathode area ($\\pm$ 3%)", dict(area_rel=0.03)),
             ("Concentration ($\\pm$ 1%)", dict(c0_rel=0.01)),
             ("Current ($\\pm$ 1%)", dict(current_rel=0.01)),
             ("Temperature ($\\pm$ 0.5 $^\\circ$C)", dict(T_abs=0.5))]
    spread = []
    for label, kw in items:
        med, lo, hi = sand_time_uncertainty(cell, **{**base, **kw})
        spread.append(100.0 * (hi - lo) / (2.0 * med))
    med, lo, hi = sand_time_uncertainty(cell)
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    y = np.arange(len(items))[::-1]
    ax.barh(y, spread, height=0.55, color=BLUE)
    for yi, s in zip(y, spread):
        ax.text(s + 0.25, yi, f"{s:.1f}%", va="center", color=INK, fontsize=9)
    ax.set(yticks=y, yticklabels=[i[0] for i in items], xlabel="Uncertainty it adds to the predicted Sand's time (one standard deviation, %)",
           xlim=(0, max(spread) * 1.18), title=f"What limits the prediction: all together $\\pm${100 * (hi - lo) / (2 * med):.0f}%")
    ax.grid(axis="y", visible=False)
    save(fig, "6_prediction_uncertainty.png")
    return med, lo, hi, dict(zip([i[0] for i in items], spread))


# ----------------------------------------------------------------------------- tables
def table_operating_points(factors):
    """Candidate operating points inside the current budget, written to CSV."""
    rows = []
    for c0 in (100.0, 250.0, 500.0):
        for area in (0.25e-4, 0.5e-4, 1.0e-4):
            for I in (5e-3, 10e-3, 20e-3):
                if I > CURRENT_BUDGET:
                    continue
                J = I / area
                D_eff = float(EL.D(c0)) * np.interp(c0, factors[0], factors[1])
                tau = analytic.sand_time_finite_cell(J, c0, D_eff, EL.t_anion, BASELINE["gap"])
                j_lim = analytic.convective_limiting_current(G0, c0, BASELINE["height"], electrolyte=EL)
                delta = analytic.convective_layer_thickness(G0, c0, BASELINE["height"], electrolyte=EL)
                tau_g = analytic.sand_time_finite_cell(J, c0, D_eff, EL.t_anion, analytic.effective_gap(BASELINE["gap"], delta))
                v0 = start_voltage(c0, J, BASELINE["gap"])
                ok = TAU_WINDOW[0] <= tau <= TAU_WINDOW[1] and v0 <= V_START_MAX and J < j_lim
                rows.append(dict(conc_M=c0 / 1000, cathode_cm2=area * 1e4, current_mA=I * 1e3, J_A_m2=round(J),
                                 sand_time_microgravity_s=round(tau, 1),
                                 sand_time_earth_vertical_s="never" if not np.isfinite(tau_g) else round(tau_g, 1),
                                 start_voltage_V=round(v0, 2),
                                 depleted_layer_mm=round(float(analytic.depletion_length(D_eff, tau)) * 1e3, 2),
                                 dendrite_speed_um_per_s=round(float(analytic.dendrite_front_velocity(J, c0, EL.t_anion)) * 1e6, 2),
                                 minutes_for_dendrites_to_cross_gap=round(BASELINE["gap"] / float(analytic.dendrite_front_velocity(J, c0, EL.t_anion)) / 60, 1),
                                 meets_all_targets="yes" if ok else "no"))
    with open(os.path.join(OUT, "operating_points.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader(); wri.writerows(rows)
    print("wrote", os.path.join(OUT, "operating_points.csv"))
    return rows


def main():
    conc_grid = np.array([50.0, 100.0, 250.0, 500.0, 1000.0])
    factors = (conc_grid, np.array([variable_D_factor(c) for c in conc_grid]))
    cell, micro, ground = figure_baseline()
    figure_tau_vs_current(factors)
    figure_design_map(factors)
    geo = figure_geometry()
    unc = figure_uncertainty(cell)
    rows = table_operating_points(factors)

    lines = []
    add = lines.append
    add("BASELINE CELL")
    add(f"  {cell.c0 / 1000:g} M CuSO4, {cell.current * 1e3:g} mA, cathode {cell.cathode_area * 1e4:g} cm^2 (J = {cell.J:.0f} A/m^2), gap {cell.gap * 1e3:g} mm, {cell.T_C:g} C")
    add(f"  Textbook Sand's time (D at bulk concentration) : {cell.sand_time_ideal():.1f} s")
    add(f"  Simulated Sand's time, microgravity            : {micro.tau:.1f} s")
    add(f"  Time the voltage has risen 0.5 V               : {micro.voltage_transition_time():.1f} s")
    k = micro.tau / cell.sand_time_ideal()
    add(f"  68% range from input uncertainties             : {k * unc[1]:.0f}-{k * unc[2]:.0f} s")
    add(f"  Starting voltage / just before the spike       : {micro.V[0]:.2f} V / {micro.V[int(0.9 * len(micro.V))]:.2f} V")
    for orient in ("vertical", "cathode_below", "cathode_above"):
        r = simulate(cell, g=G0, orientation=orient, t_end=3 * micro.tau)
        d = analytic.convective_layer_thickness(G0, cell.c0, cell.height, cell.T_C, orient, EL)
        jl = analytic.convective_limiting_current(G0, cell.c0, cell.height, cell.T_C, orient, EL)
        add(f"  Earth, {orient:14s}: Sand's time {fmt_time(r.tau):>8s}; convective layer {d * 1e3:.3f} mm; convection can carry {jl:.0f} A/m^2")
    add(f"  Depleted-layer thickness at Sand's time        : {analytic.depletion_length(cell.D0, micro.tau) * 1e3:.2f} mm")
    v = float(analytic.dendrite_front_velocity(cell.J, cell.c0, EL.t_anion))
    add(f"  Dendrite front speed after Sand's time         : {v * 1e6:.1f} um/s -> crosses the gap in {cell.gap / v / 60:.0f} min")
    add(f"  Wait between runs in the same liquid           : {analytic.relaxation_time(cell.gap, cell.D0) / 3600:.1f} h per e-fold (use ~5x)")
    add(f"  Temperature sensitivity                        : {100 * (float(EL.D(cell.c0, 26.0) / EL.D(cell.c0, 25.0)) - 1):.1f}% per deg C")
    add("")
    add("VARIABLE-D CORRECTION (simulated / textbook Sand's time)")
    for c, fct in zip(*factors):
        add(f"  {c / 1000:5.2f} M : x{fct:.3f}")
    add("")
    add("ELECTRODE LAYOUTS (5 mm wide, 10 mm tall cathode, 10 mA, 0.25 M)")
    for name, dist, u, s in geo:
        add(f"  {name:38s} area {dist.cathode_area * 1e4:.2f} cm^2 | {dist.voltage:.2f} V | local J p5-p95 {u['p5']:.2f}-{u['p95']:.2f} x mean, max {u['max']:.2f} | "
            f"local Sand's time {s['first_5pct']:.2f}-{s['last_5pct']:.2f} x")
    add("")
    add("OPERATING POINTS THAT MEET ALL TARGETS")
    for r in rows:
        if r["meets_all_targets"] == "yes":
            add(f"  {r['conc_M']:g} M, {r['cathode_cm2']:g} cm^2, {r['current_mA']:g} mA -> {r['sand_time_microgravity_s']} s in microgravity, "
                f"{r['sand_time_earth_vertical_s']} on Earth, starts at {r['start_voltage_V']} V")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
