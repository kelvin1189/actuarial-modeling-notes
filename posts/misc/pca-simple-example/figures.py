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

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
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
    fig, (ax_main, ax_var) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

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
                     fontsize=13, fontweight="bold", y=1.01)
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
    plt.savefig("before_after_pca.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  wrote before_after_pca.png")


if __name__ == "__main__":
    print("Generating figures for Article 2 (PCA toy):")
    make_before_pca()
    make_rotation_sequence()
    make_rotation_animation()
    make_before_after_pca()
    print()
    print(f"PC1 eigenvalue: {eigvals[0]:.4f}")
    print(f"PC2 eigenvalue: {eigvals[1]:.4f}")
    print(f"PC1 eigenvector: [{eigvecs[0,0]:.4f}, {eigvecs[1,0]:.4f}]")
    print(f"PC2 eigenvector: [{eigvecs[0,1]:.4f}, {eigvecs[1,1]:.4f}]")
    print(f"PC1 angle from x-axis: {theta_pc:.2f}°")
    print("Done.")
