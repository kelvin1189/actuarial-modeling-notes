"""
Generate all figures for Article 2 (PCA on Yield Curves: a 5-day worked example).

Outputs (saved alongside this script's working directory):
    before_pca.png            -- raw centered scatter (used in §3)
    rotation_sequence.png     -- six-panel rotation sequence (used in §5)
    rotation_animation.gif    -- animated rotation (used in §5)
    before_after_pca.png      -- side-by-side comparison (used in §6)

Re-run after any data change with:
    python figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------------------------------------------
# Styling -- kept minimal and consistent across all figures
# -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

COL_RAW = "#1f4e79"   # raw / centered data
COL_PC  = "#7b2d1f"   # PCA-rotated data, PC1 emphasis

# -----------------------------------------------------------------
# Data and PCA -- exact values from the article
# -----------------------------------------------------------------
raw = np.array([
    [ 2.0,  1.5],
    [ 5.0,  4.0],
    [-3.0, -2.0],
    [-1.0, -1.0],
    [ 1.0,  0.5],
])
labels = [f"Day {i+1}" for i in range(5)]
short_labels = [f"D{i+1}" for i in range(5)]

mean_vec = raw.mean(axis=0)
centered = raw - mean_vec

cov = np.cov(centered, rowvar=False, ddof=1)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]
# Sign convention: PC1's first loading positive, PC2's second positive
if eigvecs[0, 0] < 0:
    eigvecs[:, 0] *= -1
if eigvecs[1, 1] < 0:
    eigvecs[:, 1] *= -1

scores = centered @ eigvecs
theta_pc = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))


# -----------------------------------------------------------------
# Figure 1: before_pca.png -- centered raw scatter
# -----------------------------------------------------------------
def make_before_pca():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(centered[:, 0], centered[:, 1], s=80, c=COL_RAW,
               edgecolor="white", linewidth=0.8, zorder=3)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (centered[i, 0], centered[i, 1]),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Centered Δy₂ (bps)")
    ax.set_ylabel("Centered Δy₁₀ (bps)")
    ax.set_aspect("equal")
    ax.set_title("Centered yield changes — highly correlated", fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("before_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote before_pca.png")


# -----------------------------------------------------------------
# Figure 2: rotation_sequence.png -- six panels showing the rotation
# -----------------------------------------------------------------
def rotate(data, angle_deg):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    return data @ R.T


def horizontal_variance(data):
    return np.var(data[:, 0], ddof=1)


def make_rotation_sequence():
    n_panels = 6
    angles = np.linspace(0, -theta_pc, n_panels)
    max_var = eigvals[0]
    xlim, ylim = (-6, 6), (-4.5, 4.5)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for i, ang in enumerate(angles):
        ax = axes[i]
        rotated = rotate(centered, ang)
        var_h = horizontal_variance(rotated)
        pct = 100 * var_h / max_var

        ax.axhline(0, color="lightgrey", lw=0.6, zorder=1)
        ax.axvline(0, color="lightgrey", lw=0.6, zorder=1)
        ax.axhline(0, color=COL_PC, lw=1.5, alpha=0.5, zorder=2)
        ax.scatter(rotated[:, 0], rotated[:, 1], s=70, c=COL_RAW,
                   edgecolor="white", linewidth=0.8, zorder=3)
        for j, lab in enumerate(short_labels):
            ax.annotate(lab, (rotated[j, 0], rotated[j, 1]),
                        xytext=(7, 6), textcoords="offset points", fontsize=9)
        is_final = (i == n_panels - 1)
        ax.set_title(
            f"Rotation: {-ang:5.1f}°   |   Horizontal variance: {var_h:5.2f}\n"
            f"({pct:.1f}% of maximum)",
            fontsize=10,
            fontweight="bold" if is_final else "normal",
            color=COL_PC if is_final else "black",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("(horizontal axis)")
        ax.set_ylabel("(vertical axis)")

    fig.suptitle("Spinning the data until horizontal variance is maximised",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig("rotation_sequence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote rotation_sequence.png")


# -----------------------------------------------------------------
# Figure 3: rotation_animation.gif -- animated GIF
# -----------------------------------------------------------------
def make_rotation_animation():
    # NOTE: PillowWriter does NOT honour bbox_inches="tight" the way savefig
    # does for static PNGs. So we cannot put the suptitle at y > 1.0 and rely
    # on a tight bbox to expand the canvas — it will be clipped. Instead we
    # reserve top margin explicitly via subplots_adjust and place the
    # suptitle inside the figure boundary at y < 1.0.
    fig, (ax_main, ax_var) = plt.subplots(
        1, 2, figsize=(12, 6.0),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.97, wspace=0.25)

    sweep_angles = np.linspace(0, -theta_pc * 1.3, 80)
    all_angles_fine = np.linspace(-60, 60, 500)
    all_vars = np.array([horizontal_variance(rotate(centered, a))
                         for a in all_angles_fine])
    max_var = eigvals[0]
    xlim, ylim = (-6, 6), (-4.5, 4.5)

    def update(frame):
        ang = sweep_angles[frame]
        rotated = rotate(centered, ang)
        var_h = horizontal_variance(rotated)

        ax_main.clear()
        ax_main.axhline(0, color="lightgrey", lw=0.6, zorder=1)
        ax_main.axvline(0, color="lightgrey", lw=0.6, zorder=1)
        ax_main.axhline(0, color=COL_PC, lw=1.5, alpha=0.5, zorder=2)
        ax_main.scatter(rotated[:, 0], rotated[:, 1], s=80, c=COL_RAW,
                        edgecolor="white", linewidth=0.8, zorder=3)
        for j, lab in enumerate(short_labels):
            ax_main.annotate(lab, (rotated[j, 0], rotated[j, 1]),
                             xytext=(7, 6), textcoords="offset points", fontsize=10)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_main.set_aspect("equal")
        ax_main.grid(True, alpha=0.2)
        ax_main.set_xlabel("(horizontal axis)")
        ax_main.set_ylabel("(vertical axis)")
        ax_main.set_title(f"Rotation: {-ang:5.1f}°", fontweight="bold")

        ax_var.clear()
        ax_var.plot(-all_angles_fine, all_vars, color="grey", lw=1.2)
        ax_var.axhline(max_var, color=COL_PC, ls="--", lw=1, alpha=0.6,
                       label=f"Max = {max_var:.2f} (PC1 eigenvalue)")
        ax_var.scatter([-ang], [var_h], s=90, c=COL_PC, zorder=5)
        ax_var.axvline(theta_pc, color=COL_PC, ls=":", lw=1, alpha=0.5)
        ax_var.set_xlim(-10, 60)
        ax_var.set_ylim(0, max_var * 1.1)
        ax_var.set_xlabel("Rotation angle (degrees)")
        ax_var.set_ylabel("Variance along horizontal axis")
        ax_var.set_title(f"Current variance: {var_h:.2f}", fontweight="bold")
        ax_var.legend(loc="lower right", fontsize=9)
        ax_var.grid(True, alpha=0.2)

        fig.suptitle("PCA is the rotation that maximises horizontal variance",
                     fontsize=13, fontweight="bold", y=0.95)
        return []

    ani = FuncAnimation(fig, update, frames=len(sweep_angles),
                        interval=80, blit=False)
    ani.save("rotation_animation.gif", writer=PillowWriter(fps=15))
    plt.close()
    print("  wrote rotation_animation.gif")


# -----------------------------------------------------------------
# Figure 4: before_after_pca.png -- side-by-side
# -----------------------------------------------------------------
def make_before_after_pca():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(centered[:, 0], centered[:, 1], s=80, c=COL_RAW,
               edgecolor="white", linewidth=0.8, zorder=3)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (centered[i, 0], centered[i, 1]),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Centered Δy₂ (bps)")
    ax.set_ylabel("Centered Δy₁₀ (bps)")
    ax.set_title("Before PCA: correlated", fontweight="bold")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.scatter(scores[:, 0], scores[:, 1], s=80, c=COL_PC,
               edgecolor="white", linewidth=0.8, zorder=3)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (scores[i, 0], scores[i, 1]),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("PC1 score (Level)")
    ax.set_ylabel("PC2 score (Steepness)")
    ax.set_title("After PCA: uncorrelated", fontweight="bold")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("before_after_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote before_after_pca.png")


# -----------------------------------------------------------------
# Figure 5: stress_overlay.png -- the two stress scenarios overlaid on the raw sample
# -----------------------------------------------------------------
def make_stress_overlay():
    """
    Plot the 5 observed days in raw yield-change space, then overlay:
      - the historical mean
      - Scenario A: mean + 1 SD on PC1 (Level shock)
      - Scenario B: mean + 1 SD on PC2 (Steepness shock)
      - the PC1 / PC2 eigenvector axes drawn through the mean
    The point of the figure is to show that PCA-derived stresses sit
    on the observed cloud, not off it -- and that Stress B is so small
    it is visually almost on top of the mean.
    """
    COL_S1 = "#7b2d1f"   # Stress A (PC1 / Level)
    COL_S2 = "#2d6a4f"   # Stress B (PC2 / Steepness)
    COL_MU = "#444444"   # mean

    # The two stresses in RAW yield-change space (not centered)
    sd_pc1 = float(np.sqrt(eigvals[0]))
    sd_pc2 = float(np.sqrt(eigvals[1]))
    stress_A = mean_vec + sd_pc1 * eigvecs[:, 0]
    stress_B = mean_vec + sd_pc2 * eigvecs[:, 1]

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # PC1 eigenvector line through the mean (full sample extent)
    t = np.linspace(-5, 5, 2)
    pc1_line = mean_vec[:, None] + eigvecs[:, 0:1] * t
    pc2_line = mean_vec[:, None] + eigvecs[:, 1:2] * t * 0.6
    ax.plot(pc1_line[0], pc1_line[1], color=COL_S1, ls="--", lw=1.0,
            alpha=0.45, zorder=1,
            label="PC1 direction (through mean)")
    ax.plot(pc2_line[0], pc2_line[1], color=COL_S2, ls="--", lw=1.0,
            alpha=0.45, zorder=1,
            label="PC2 direction (through mean)")

    # Observed days
    ax.scatter(raw[:, 0], raw[:, 1], s=85, c=COL_RAW,
               edgecolor="white", linewidth=0.8, zorder=3,
               label="Observed days")
    for i, lab in enumerate(labels):
        ax.annotate(lab, (raw[i, 0], raw[i, 1]),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)

    # Historical mean
    ax.scatter([mean_vec[0]], [mean_vec[1]], s=180, marker="o",
               facecolor="white", edgecolor=COL_MU, linewidth=2.2,
               zorder=4, label=f"Mean μ = ({mean_vec[0]:.2f}, {mean_vec[1]:.2f})")

    # Stress A
    ax.scatter([stress_A[0]], [stress_A[1]], s=140, marker="D",
               color=COL_S1, edgecolor="white", linewidth=0.8,
               zorder=5, label=f"+1 SD on PC1 = ({stress_A[0]:.2f}, {stress_A[1]:.2f})")
    ax.annotate("Stress A", (stress_A[0], stress_A[1]),
                xytext=(10, -4), textcoords="offset points",
                fontsize=10, color=COL_S1, fontweight="bold")

    # Stress B
    ax.scatter([stress_B[0]], [stress_B[1]], s=140, marker="^",
               color=COL_S2, edgecolor="white", linewidth=0.8,
               zorder=5, label=f"+1 SD on PC2 = ({stress_B[0]:.2f}, {stress_B[1]:.2f})")
    ax.annotate("Stress B", (stress_B[0], stress_B[1]),
                xytext=(10, 6), textcoords="offset points",
                fontsize=10, color=COL_S2, fontweight="bold")

    # Connector line from mean to Stress A (showing the shock vector)
    ax.annotate("", xy=stress_A, xytext=mean_vec,
                arrowprops=dict(arrowstyle="->", color=COL_S1,
                                lw=1.5, alpha=0.85))

    ax.axhline(0, color="lightgrey", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgrey", lw=0.5, zorder=0)
    ax.set_xlim(-4.5, 6.5)
    ax.set_ylim(-3.5, 5.0)
    ax.set_xlabel("Δy₂ (2-Year change, bps)")
    ax.set_ylabel("Δy₁₀ (10-Year change, bps)")
    ax.set_title(
        "Stress scenarios overlaid on the 5 observed days\n"
        "(raw yield-change space, including the sample mean)",
        fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    plt.savefig("stress_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote stress_overlay.png")


# -----------------------------------------------------------------
# Figure 6: before_after_stress.png -- the 5 days before and after each stress
# -----------------------------------------------------------------
def make_before_after_stress():
    """
    Two panels showing the rigid translation of the entire 5-day cloud
    under each stress. Left: +1 SD on PC1. Right: +1 SD on PC2.
    Each original day is connected to its stressed counterpart by a
    light grey arrow, making the uniform translation visually explicit.
    The PC2 panel will look almost unchanged from the original -- which
    is exactly the point.
    """
    COL_S1 = "#7b2d1f"
    COL_S2 = "#2d6a4f"

    # Shock vectors matching the rounded values used in the article tables
    shock_pc1 = np.array([3.031, 2.324])
    shock_pc2 = np.array([-0.119, 0.155])

    stressed_pc1 = raw + shock_pc1
    stressed_pc2 = raw + shock_pc2

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- Left panel: +PC1 (Level) ----
    ax = axes[0]
    # Connecting arrows from original to stressed (drawn first so points sit on top)
    for i in range(len(raw)):
        ax.annotate("", xy=stressed_pc1[i], xytext=raw[i],
                    arrowprops=dict(arrowstyle="->", color="grey",
                                    alpha=0.45, lw=1.0),
                    zorder=2)
    # Originals
    ax.scatter(raw[:, 0], raw[:, 1], s=80, c=COL_RAW,
               edgecolor="white", linewidth=0.8, zorder=3,
               label="Original")
    for i, lab in enumerate(labels):
        ax.annotate(lab, (raw[i, 0], raw[i, 1]),
                    xytext=(-26, -2), textcoords="offset points",
                    fontsize=9, color=COL_RAW)
    # Stressed
    ax.scatter(stressed_pc1[:, 0], stressed_pc1[:, 1], s=85,
               c=COL_S1, marker="D",
               edgecolor="white", linewidth=0.8, zorder=4,
               label="After +1 SD on PC1")

    ax.axhline(0, color="lightgrey", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgrey", lw=0.5, zorder=0)
    ax.set_xlim(-4.5, 9.5)
    ax.set_ylim(-3, 7.5)
    ax.set_xlabel("Δy₂ (2-Year change, bps)")
    ax.set_ylabel("Δy₁₀ (10-Year change, bps)")
    ax.set_aspect("equal")
    ax.set_title("Each day translated by +1 SD on PC1\n"
                 "shock vector = (+3.03, +2.32) bps",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # ---- Right panel: +PC2 (Steepness) ----
    ax = axes[1]
    for i in range(len(raw)):
        ax.annotate("", xy=stressed_pc2[i], xytext=raw[i],
                    arrowprops=dict(arrowstyle="->", color="grey",
                                    alpha=0.45, lw=1.0),
                    zorder=2)
    ax.scatter(raw[:, 0], raw[:, 1], s=80, c=COL_RAW,
               edgecolor="white", linewidth=0.8, zorder=3,
               label="Original")
    for i, lab in enumerate(labels):
        ax.annotate(lab, (raw[i, 0], raw[i, 1]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=COL_RAW)
    ax.scatter(stressed_pc2[:, 0], stressed_pc2[:, 1], s=85,
               c=COL_S2, marker="^",
               edgecolor="white", linewidth=0.8, zorder=4,
               label="After +1 SD on PC2")

    ax.axhline(0, color="lightgrey", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgrey", lw=0.5, zorder=0)
    ax.set_xlim(-4.5, 6.5)
    ax.set_ylim(-3, 5)
    ax.set_xlabel("Δy₂ (2-Year change, bps)")
    ax.set_ylabel("Δy₁₀ (10-Year change, bps)")
    ax.set_aspect("equal")
    ax.set_title("Each day translated by +1 SD on PC2\n"
                 "shock vector = (−0.12, +0.16) bps",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("The 5 days before and after each stress shock",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig("before_after_stress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote before_after_stress.png")


if __name__ == "__main__":
    print("Generating figures for Article 2 (PCA toy):")
    make_before_pca()
    make_rotation_sequence()
    make_rotation_animation()
    make_before_after_pca()
    make_stress_overlay()
    make_before_after_stress()
    print()
    print(f"PC1 eigenvalue: {eigvals[0]:.4f}")
    print(f"PC2 eigenvalue: {eigvals[1]:.4f}")
    print(f"PC1 eigenvector: [{eigvecs[0,0]:.4f}, {eigvecs[1,0]:.4f}]")
    print(f"PC2 eigenvector: [{eigvecs[0,1]:.4f}, {eigvecs[1,1]:.4f}]")
    print(f"PC1 angle from x-axis: {theta_pc:.2f}°")
    print("Done.")
