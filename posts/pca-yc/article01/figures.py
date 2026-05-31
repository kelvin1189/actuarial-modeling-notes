"""
Generate all figures for the 3-tenor PCA article.

Outputs (saved in the current working directory):
    raw_scatter_3d.png         -- 3D scatter of the 10 weekly observations
    pc_loadings.png            -- bar chart: each PC's loadings on the 3 tenors
    before_after_pca.png       -- 3D scatter before (raw) and after (PC scores)
    rotation_animation.gif     -- animated viewpoint rotation around the cloud
                                  with eigenvector axes drawn through the mean
    stress_overlay.png         -- 3D scatter of 10 weeks + mean + 3 stress points
    before_after_stress.png    -- 3 panels: cloud translated by each shock

Run from the article's working directory:
    python figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 110,
})

# Colours kept consistent with the 2D toy article
COL_RAW = "#1f4e79"   # raw observations
COL_PC  = "#7b2d1f"   # PCA / PC1 / Level emphasis
COL_S1  = "#7b2d1f"   # Stress A (PC1)
COL_S2  = "#2d6a4f"   # Stress B (PC2)
COL_S3  = "#b8860b"   # Stress C (PC3)
COL_MU  = "#444444"   # mean

# -----------------------------------------------------------------
# Data and PCA -- exact values used in the article
# -----------------------------------------------------------------
raw = np.array([
    [  5,   5,   5],
    [ -4,  -4,  -4],
    [  6,   5,   4],
    [ -5,  -4,  -3],
    [  3,   3,   3],
    [ -3,   0,   2],
    [ -2,   1,   3],
    [  3,   0,  -2],
    [  1,   3,   1],
    [ -1,  -3,  -1],
], dtype=float)
tenors = ["2y", "5y", "10y"]
labels = [f"W{i+1}" for i in range(10)]

mean_vec = raw.mean(axis=0)
centered = raw - mean_vec

cov = np.cov(centered, rowvar=False, ddof=1)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]

# Sign conventions (matching compute_numbers.py and the article tables)
if eigvecs[:, 0].sum() < 0:
    eigvecs[:, 0] *= -1
if eigvecs[2, 1] < 0:
    eigvecs[:, 1] *= -1
if eigvecs[1, 2] > 0:
    eigvecs[:, 2] *= -1

scores = centered @ eigvecs
sd = np.sqrt(eigvals)

shock_A = sd[0] * eigvecs[:, 0]
shock_B = sd[1] * eigvecs[:, 1]
shock_C = sd[2] * eigvecs[:, 2]


# -----------------------------------------------------------------
# Helper for 3D scatters with consistent styling
# -----------------------------------------------------------------
def style_3d(ax, xlabel="Δy(2y) (bps)", ylabel="Δy(5y) (bps)",
             zlabel="Δy(10y) (bps)"):
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.tick_params(axis="both", labelsize=9)
    # Soft grey panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#fafafa")
        axis.pane.set_edgecolor("lightgrey")


# -----------------------------------------------------------------
# Figure 1: raw_scatter_3d.png
# -----------------------------------------------------------------
def make_raw_scatter():
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], s=70, c=COL_RAW,
               edgecolor="white", linewidth=0.8, depthshade=True)
    for i, lab in enumerate(labels):
        ax.text(raw[i, 0], raw[i, 1], raw[i, 2] + 0.4, lab,
                fontsize=9, ha="left", color=COL_RAW)
    # Mean
    ax.scatter([mean_vec[0]], [mean_vec[1]], [mean_vec[2]],
               s=160, marker="o", facecolor="white",
               edgecolor=COL_MU, linewidth=2, depthshade=False,
               label=f"Mean μ = ({mean_vec[0]:.2f}, {mean_vec[1]:.2f}, {mean_vec[2]:.2f})")
    ax.set_title("10 weeks of yield changes on (2y, 5y, 10y)",
                 fontweight="bold", pad=10)
    style_3d(ax)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    plt.savefig("raw_scatter_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote raw_scatter_3d.png")


# -----------------------------------------------------------------
# Figure 2: pc_loadings.png -- the canonical L/S/C loadings bar chart
# -----------------------------------------------------------------
def make_pc_loadings():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    pc_names = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    pc_colors = [COL_S1, COL_S2, COL_S3]
    x = np.arange(3)

    for i, ax in enumerate(axes):
        loadings = eigvecs[:, i]
        bars = ax.bar(x, loadings, color=pc_colors[i],
                      edgecolor="white", linewidth=1.2, alpha=0.92)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(tenors, fontsize=11)
        ax.set_ylim(-1.0, 1.0)
        share = 100 * eigvals[i] / eigvals.sum()
        ax.set_title(f"{pc_names[i]}\n{share:.1f}% of variance",
                     fontweight="bold", fontsize=11)
        # Numeric labels
        for bar, val in zip(bars, loadings):
            y = val + (0.05 if val >= 0 else -0.07)
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:+.3f}", ha="center", fontsize=10,
                    color="black")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("Loading")

    fig.suptitle("PC loadings across the three tenors",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("pc_loadings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote pc_loadings.png")


# -----------------------------------------------------------------
# Figure 3: before_after_pca.png -- 3D scatter before / after
# -----------------------------------------------------------------
def make_before_after_pca():
    fig = plt.figure(figsize=(14, 6.5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(centered[:, 0], centered[:, 1], centered[:, 2], s=70,
                c=COL_RAW, edgecolor="white", linewidth=0.8)
    for i, lab in enumerate(labels):
        ax1.text(centered[i, 0], centered[i, 1], centered[i, 2] + 0.4,
                 lab, fontsize=8, color=COL_RAW)
    ax1.set_title(f"Before PCA: centered (2y, 5y, 10y)\n"
                  f"max pairwise correlation = "
                  f"{np.corrcoef(centered.T)[1,2]:.2f}",
                  fontweight="bold", fontsize=11)
    style_3d(ax1, "Centered Δy(2y)", "Centered Δy(5y)", "Centered Δy(10y)")
    ax1.view_init(elev=22, azim=-58)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(scores[:, 0], scores[:, 1], scores[:, 2], s=70,
                c=COL_PC, edgecolor="white", linewidth=0.8)
    for i, lab in enumerate(labels):
        ax2.text(scores[i, 0], scores[i, 1], scores[i, 2] + 0.15,
                 lab, fontsize=8, color=COL_PC)
    # Correlation across PC axes
    corr_pc = np.corrcoef(scores.T)
    max_off = np.max(np.abs(corr_pc - np.eye(3)))
    ax2.set_title(f"After PCA: (PC1, PC2, PC3)\n"
                  f"max off-diagonal correlation = {max_off:.1e} (≈ 0)",
                  fontweight="bold", fontsize=11)
    style_3d(ax2, "PC1 score (Level)", "PC2 score (Slope)", "PC3 score (Curv.)")
    ax2.view_init(elev=22, azim=-58)

    plt.tight_layout()
    plt.savefig("before_after_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote before_after_pca.png")


# -----------------------------------------------------------------
# Figure 4: rotation_animation.gif -- animated viewpoint orbit with
#           eigenvector axes drawn through the mean
# -----------------------------------------------------------------
def make_rotation_animation():
    fig = plt.figure(figsize=(10, 7.5))
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)
    ax = fig.add_subplot(111, projection="3d")

    sd_scale = 1.0
    # Eigenvector axes extending ±2 SD from the mean
    axis_extents = []
    for i in range(3):
        v = eigvecs[:, i] * sd[i] * 2.0 * sd_scale
        p0 = mean_vec - v
        p1 = mean_vec + v
        axis_extents.append((p0, p1))

    pc_colors = [COL_S1, COL_S2, COL_S3]
    pc_names = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]

    def update(frame):
        ax.clear()
        # Observations
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], s=60,
                   c=COL_RAW, edgecolor="white", linewidth=0.6,
                   depthshade=True)
        # Mean
        ax.scatter([mean_vec[0]], [mean_vec[1]], [mean_vec[2]],
                   s=140, marker="o", facecolor="white",
                   edgecolor=COL_MU, linewidth=2, depthshade=False)
        # Eigenvector axes through the mean
        for (p0, p1), col, nm in zip(axis_extents, pc_colors, pc_names):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=col, lw=2.5, alpha=0.85,
                    label=nm if frame == 0 else None)
            # Small marker at the +SD tip
            tip = mean_vec + eigvecs[:, pc_colors.index(col)] * sd[pc_colors.index(col)]
            ax.scatter([tip[0]], [tip[1]], [tip[2]],
                       s=40, c=col, edgecolor="white", linewidth=0.6,
                       depthshade=False)

        ax.set_xlim(-6, 8)
        ax.set_ylim(-5, 6)
        ax.set_zlim(-5, 6)
        style_3d(ax)
        # Orbit the camera
        azim = -60 + frame * (360 / n_frames)
        ax.view_init(elev=22, azim=azim)
        ax.set_title("PCA axes are the variance-aligned axes of the cloud\n"
                     "(rotating viewpoint; lines = PC1 / PC2 / PC3 through the mean, ±1 SD)",
                     fontweight="bold", fontsize=11)
        # Manual legend (handles disappear on clear)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=c, lw=2.5, label=n)
                   for c, n in zip(pc_colors, pc_names)]
        ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)
        return []

    n_frames = 72  # 360 degrees / 5-degree step
    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=80, blit=False)
    ani.save("rotation_animation.gif", writer=PillowWriter(fps=15))
    plt.close()
    print("  wrote rotation_animation.gif")


# -----------------------------------------------------------------
# Figure 5: stress_overlay.png -- 10 weeks + mean + 3 stresses in 3D
# -----------------------------------------------------------------
def make_stress_overlay():
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Observations
    ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], s=55, c=COL_RAW,
               edgecolor="white", linewidth=0.6, depthshade=True,
               label="10 observed weeks")
    # Mean
    ax.scatter([mean_vec[0]], [mean_vec[1]], [mean_vec[2]],
               s=160, marker="o", facecolor="white",
               edgecolor=COL_MU, linewidth=2, depthshade=False,
               label=f"Mean μ = ({mean_vec[0]:.2f}, {mean_vec[1]:.2f}, {mean_vec[2]:.2f})")

    stresses = [
        (shock_A, COL_S1, "D", "Stress A (+1 SD PC1, Level)"),
        (shock_B, COL_S2, "^", "Stress B (+1 SD PC2, Slope)"),
        (shock_C, COL_S3, "s", "Stress C (+1 SD PC3, Curvature)"),
    ]
    for shock, col, marker, name in stresses:
        p = mean_vec + shock
        ax.scatter([p[0]], [p[1]], [p[2]], s=130, c=col, marker=marker,
                   edgecolor="white", linewidth=0.6, depthshade=False,
                   label=name)
        # Arrow (segment from mean to stress point)
        ax.plot([mean_vec[0], p[0]], [mean_vec[1], p[1]], [mean_vec[2], p[2]],
                color=col, lw=1.5, alpha=0.85)

    ax.set_title("Three stress scenarios overlaid on the 10 weekly observations",
                 fontweight="bold", fontsize=11)
    style_3d(ax)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    plt.savefig("stress_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote stress_overlay.png")


# -----------------------------------------------------------------
# Figure 6: before_after_stress.png -- three panels, one per shock
# -----------------------------------------------------------------
def make_before_after_stress():
    fig = plt.figure(figsize=(16, 6))
    panels = [
        (shock_A, COL_S1, "D", "Level (+1 SD PC1)",
         f"shock = ({shock_A[0]:+.2f}, {shock_A[1]:+.2f}, {shock_A[2]:+.2f})"),
        (shock_B, COL_S2, "^", "Slope (+1 SD PC2)",
         f"shock = ({shock_B[0]:+.2f}, {shock_B[1]:+.2f}, {shock_B[2]:+.2f})"),
        (shock_C, COL_S3, "s", "Curvature (+1 SD PC3)",
         f"shock = ({shock_C[0]:+.2f}, {shock_C[1]:+.2f}, {shock_C[2]:+.2f})"),
    ]

    for idx, (shock, col, marker, name, shock_str) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        stressed = raw + shock

        # Connection segments (drawn first, behind points)
        segs = [[(raw[i, 0], raw[i, 1], raw[i, 2]),
                 (stressed[i, 0], stressed[i, 1], stressed[i, 2])]
                for i in range(len(raw))]
        lc = Line3DCollection(segs, colors="grey", alpha=0.4, linewidths=0.9)
        ax.add_collection3d(lc)

        # Originals
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], s=50, c=COL_RAW,
                   edgecolor="white", linewidth=0.6, depthshade=True,
                   label="Original")
        # Stressed
        ax.scatter(stressed[:, 0], stressed[:, 1], stressed[:, 2], s=55,
                   c=col, marker=marker, edgecolor="white", linewidth=0.6,
                   depthshade=True, label="Stressed")

        ax.set_title(f"{name}\n{shock_str}", fontweight="bold", fontsize=10)
        style_3d(ax)
        ax.view_init(elev=22, azim=-58)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    fig.suptitle("Each of the 10 weeks translated by the corresponding shock vector",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig("before_after_stress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote before_after_stress.png")


# -----------------------------------------------------------------
# Figure 7: curve_shock_mean.png
# -----------------------------------------------------------------
# Visualise the three stresses as yield-curve transformations rather than
# as coordinate translations. Adds an arbitrary baseline level curve
# (2.00 / 2.50 / 3.00 %) and shows baseline + (mean + shock) curves.
# This is Interpretation A from the article's exposition: the PCA is
# still on changes; the level baseline is invented for visualisation only.
# -----------------------------------------------------------------
def make_curve_shock_mean():
    baseline_pct = np.array([2.00, 2.50, 3.00])  # 2y, 5y, 10y in PERCENT
    tenor_x = np.array([2, 5, 10])

    # Convert mean and stress impacts from bps to % for plotting on the
    # same axis as the baseline level curve.
    # 1 bp = 0.01% so divide by 100.
    mean_pct = mean_vec / 100.0
    shocks_pct = {
        "Level (+1 SD PC1)":     (shock_A / 100.0, COL_S1, "D"),
        "Slope (+1 SD PC2)":     (shock_B / 100.0, COL_S2, "^"),
        "Curvature (+1 SD PC3)": (shock_C / 100.0, COL_S3, "s"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- LEFT panel: level curves ----
    ax = axes[0]
    # Baseline
    ax.plot(tenor_x, baseline_pct, color=COL_MU, lw=2.0, marker="o",
            markersize=8, markerfacecolor="white", markeredgewidth=2,
            label=f"Baseline ({baseline_pct[0]:.2f}, {baseline_pct[1]:.2f}, {baseline_pct[2]:.2f}) %",
            zorder=3)
    # Stressed: baseline + (mean change + shock)
    for name, (shock, col, marker) in shocks_pct.items():
        stressed = baseline_pct + mean_pct + shock
        ax.plot(tenor_x, stressed, color=col, lw=1.8, marker=marker,
                markersize=8, alpha=0.92,
                label=f"After {name}", zorder=4)

    ax.set_xticks(tenor_x)
    ax.set_xticklabels(["2y", "5y", "10y"])
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Yield level (%)")
    ax.set_title("Baseline yield curve and the three stressed curves",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # ---- RIGHT panel: shock vectors as curves ----
    ax = axes[1]
    # Plot the shocks themselves (relative to baseline), in bps for clarity
    for name, (shock_pc, col, marker) in shocks_pct.items():
        shock_bps = shock_pc * 100.0  # convert back to bps for the y-axis
        ax.plot(tenor_x, shock_bps, color=col, lw=1.8, marker=marker,
                markersize=8, alpha=0.92, label=name)
    ax.axhline(0, color=COL_MU, lw=1, alpha=0.5, ls="--",
               label="Zero shock (no change)")

    ax.set_xticks(tenor_x)
    ax.set_xticklabels(["2y", "5y", "10y"])
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Shock impact (bps)")
    ax.set_title("The three shocks viewed as curves\n"
                 "(the L/S/C signature)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("Stress scenarios as yield-curve transformations",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("curve_shock_mean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote curve_shock_mean.png")


# -----------------------------------------------------------------
# Figure 8: curves_before_after_stress.png  (LEVEL-SPACE version)
# -----------------------------------------------------------------
# Per-week version showing each week's yield-LEVEL curve before vs after
# each shock, on a tight y-axis per panel so the small (3-4 bps) shock
# remains visible against a 200-300 bps baseline.
#
# Baseline construction: arbitrary illustrative yields of
# (2.00, 2.50, 3.00) % on the (2y, 5y, 10y) tenors. The baseline plays
# no role in the PCA -- it exists only to anchor the level-curve
# visualization. See the article's §9 introduction for the explicit
# caveat.
# -----------------------------------------------------------------
def make_curves_before_after_stress():
    tenor_x = np.array([2, 5, 10])
    baseline_pct = np.array([2.00, 2.50, 3.00])  # 2y, 5y, 10y

    # Each week's level curve = baseline + observed weekly change (bps -> %)
    # raw is in bps; convert to % by dividing by 100.
    baseline_levels = baseline_pct[None, :] + raw / 100.0  # (10, 3)

    panels = [
        (shock_A, COL_S1, "D", "Level (+1 SD PC1)",
         f"shock = ({shock_A[0]:+.2f}, {shock_A[1]:+.2f}, {shock_A[2]:+.2f}) bps"),
        (shock_B, COL_S2, "^", "Slope (+1 SD PC2)",
         f"shock = ({shock_B[0]:+.2f}, {shock_B[1]:+.2f}, {shock_B[2]:+.2f}) bps"),
        (shock_C, COL_S3, "s", "Curvature (+1 SD PC3)",
         f"shock = ({shock_C[0]:+.2f}, {shock_C[1]:+.2f}, {shock_C[2]:+.2f}) bps"),
    ]

    # NOTE: deliberately not sharey -- with such small shocks against a
    # ~2-3% baseline, we need each panel to set its own tight y-range.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, (shock_bps, col, marker, name, shock_str) in zip(axes, panels):
        stressed_levels = baseline_levels + shock_bps / 100.0  # (10, 3) in %

        # Baseline weekly level curves (faint blue)
        for i in range(len(raw)):
            ax.plot(tenor_x, baseline_levels[i], color=COL_RAW,
                    lw=1.0, alpha=0.40, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    zorder=2)
        # Stressed weekly level curves (emphasised in shock colour)
        for i in range(len(raw)):
            ax.plot(tenor_x, stressed_levels[i], color=col,
                    lw=1.3, alpha=0.85, marker=marker, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    zorder=3)

        # Tight per-panel y-range so the shock is visible
        all_vals = np.concatenate([baseline_levels.ravel(),
                                    stressed_levels.ravel()])
        pad = 0.015  # 1.5 percentage-point padding
        ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)

        # Dummy lines for legend
        ax.plot([], [], color=COL_RAW, lw=1.5, marker="o", markersize=5,
                label="10 baseline weekly level curves")
        ax.plot([], [], color=col, lw=1.5, marker=marker, markersize=6,
                label="10 stressed weekly level curves")

        ax.set_xticks(tenor_x)
        ax.set_xticklabels(["2y", "5y", "10y"])
        ax.set_xlabel("Tenor")
        ax.set_ylabel("Yield level (%)")
        ax.set_title(f"{name}\n{shock_str}", fontweight="bold", fontsize=10)
        # Show y-axis values to 3 decimal places so the shock is readable
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Each of the 10 weekly LEVEL curves before and after each shock\n"
                 "(baseline = arbitrary 2.00 / 2.50 / 3.00 %; same shock vector added to every week)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("curves_before_after_stress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote curves_before_after_stress.png (level-space)")


# -----------------------------------------------------------------
# Figure 9: var_995_stress.png  --  99.5% VaR severity stress curves
# -----------------------------------------------------------------
# Single-panel level-curve figure showing the baseline yield curve plus
# six stressed yield curves at the 99.5% one-year VaR multiplier
# (k = 2.576 SD under a Gaussian assumption). Both positive and negative
# directions on each PC are shown, since regulatory stress testing
# considers both: Solvency II asks for the worst case across an upward
# AND a downward shock.
#
# Baseline = arbitrary illustrative (2.00, 2.50, 3.00) % at (2y, 5y, 10y).
# -----------------------------------------------------------------
def make_var_995_stress():
    tenor_x = np.array([2, 5, 10])
    baseline_pct = np.array([2.00, 2.50, 3.00])
    k = 2.576  # 99.5% one-year VaR multiplier under Gaussian

    # Each scenario: baseline + mean change + (sign * k * SD * eigenvector)
    mean_shift_pct = mean_vec / 100.0   # bps -> %
    base_plus_mean = baseline_pct + mean_shift_pct

    scenarios = [
        ("+k SD PC1 (Level up)",        +k, 0,  COL_S1, "D", "-"),
        ("−k SD PC1 (Level down)",      -k, 0,  COL_S1, "v", "--"),
        ("+k SD PC2 (Steepener)",       +k, 1,  COL_S2, "^", "-"),
        ("−k SD PC2 (Flattener)",       -k, 1,  COL_S2, "v", "--"),
        ("+k SD PC3 (Curvature up)",    +k, 2,  COL_S3, "s", "-"),
        ("−k SD PC3 (Curvature down)",  -k, 2,  COL_S3, "P", "--"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Baseline (mean-adjusted) reference curve
    ax.plot(tenor_x, base_plus_mean, color=COL_MU, lw=2.6, marker="o",
            markersize=9, markerfacecolor="white", markeredgewidth=2,
            label=f"Baseline + mean change "
                  f"({base_plus_mean[0]:.3f}, {base_plus_mean[1]:.3f}, "
                  f"{base_plus_mean[2]:.3f}) %",
            zorder=5)

    # Six stressed curves
    for name, sign, pc_idx, col, marker, ls in scenarios:
        shock_pct = sign * k * sd[pc_idx] * eigvecs[:, pc_idx] / 100.0
        stressed = base_plus_mean + shock_pct
        ax.plot(tenor_x, stressed, color=col, lw=1.7, marker=marker,
                markersize=8, ls=ls, alpha=0.92,
                label=name, zorder=4)

    ax.set_xticks(tenor_x)
    ax.set_xticklabels(["2y", "5y", "10y"])
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Yield level (%)")
    ax.set_title("99.5% one-year VaR stress scenarios\n"
                 "(k = 2.576 SD; six scenarios, one in each direction on PC1 / PC2 / PC3)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95,
              bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig("var_995_stress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote var_995_stress.png")


if __name__ == "__main__":
    print("Generating figures for the 3-tenor PCA article:")
    make_raw_scatter()
    make_pc_loadings()
    make_before_after_pca()
    make_rotation_animation()
    make_stress_overlay()
    make_before_after_stress()
    make_curve_shock_mean()
    make_curves_before_after_stress()
    make_var_995_stress()
    print(f"Variance shares: PC1={100*eigvals[0]/eigvals.sum():.2f}%, "
          f"PC2={100*eigvals[1]/eigvals.sum():.2f}%, "
          f"PC3={100*eigvals[2]/eigvals.sum():.2f}%")
    print("Done.")
