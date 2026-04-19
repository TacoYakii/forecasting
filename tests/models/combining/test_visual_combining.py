"""Visual verification of horizontal, vertical, and angular combining.

Reproduces Figures 1-5 from Taylor & Meng (2025) using two Gaussian CDFs
with means -0.15 and 0.15, both with std=0.1, equal weights.

Generates CDF and PDF plots for θ = 0° (horizontal), 45° (angular),
and 90° (vertical) to visually confirm the sampling-based implementation
matches the theoretical behavior:
  - Vertical (90°): bimodal PDF, widest distribution
  - Horizontal (0°): unimodal PDF, narrowest distribution
  - Angular (45°): intermediate shape

Usage:
    uv run python tests/models/combining/test_visual_combining.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from src.models.combining.vertical import _sampling_combine

OUTPUT_DIR = Path(__file__).parent / "visual_output"

# ── Setup: Two Gaussian CDFs (Taylor & Meng 2025, Figures 1-5) ──
MU_1, MU_2, SIGMA = -0.15, 0.15, 0.1
N_QUANTILES = 999
TAU = np.linspace(0, 1, N_QUANTILES + 2)[1:-1]
WEIGHTS = np.array([0.5, 0.5])

# Quantile arrays for the two Gaussians: shape (1, Q) — single timestep
Q1 = norm.ppf(TAU, loc=MU_1, scale=SIGMA).reshape(1, -1)
Q2 = norm.ppf(TAU, loc=MU_2, scale=SIGMA).reshape(1, -1)
QUANTILE_ARRAYS = [Q1, Q2]

N_SAMPLES = 50000  # high for smooth visual plots
DEGREES = [0.0, 45.0, 90.0]
LABELS = {
    0.0: "Horizontal (θ = 0°)",
    45.0: "Angular (θ = 45°)",
    90.0: "Vertical (θ = 90°)",
}
COLORS = {0.0: "tab:blue", 45.0: "tab:red", 90.0: "tab:green"}


def _combine_at_degree(degree: float) -> np.ndarray:
    """Return combined quantile values (Q,) for a single timestep."""
    result = _sampling_combine(
        WEIGHTS, TAU, QUANTILE_ARRAYS,
        degree_deg=degree, n_samples=N_SAMPLES, rng_seed=42,
    )
    return result[0]  # single timestep → (Q,)


def plot_cdf_pdf():
    """Generate CDF and PDF comparison plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_cdf, ax_pdf = axes

    # ── Individual CDFs (gray) ──
    x_grid = np.linspace(-0.5, 0.5, 500)
    for mu, label in [(MU_1, "F₁"), (MU_2, "F₂")]:
        cdf_vals = norm.cdf(x_grid, loc=mu, scale=SIGMA)
        ax_cdf.plot(x_grid, cdf_vals, color="gray", alpha=0.5,
                    linewidth=1.5, label=label)
        pdf_vals = norm.pdf(x_grid, loc=mu, scale=SIGMA)
        ax_pdf.plot(x_grid, pdf_vals, color="gray", alpha=0.5,
                    linewidth=1.5, label=label)

    # ── Combined CDFs for each degree ──
    for deg in DEGREES:
        q_combined = _combine_at_degree(deg)
        color = COLORS[deg]
        label = LABELS[deg]

        # CDF: plot quantile levels vs quantile values
        ax_cdf.plot(q_combined, TAU, color=color, linewidth=2, label=label)

        # PDF: KDE from quantile-derived samples (smooth, no finite-diff noise)
        samples = np.interp(
            np.random.default_rng(0).uniform(0, 1, 50000), TAU, q_combined
        )
        kde = gaussian_kde(samples)
        ax_pdf.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)

    # ── Formatting ──
    ax_cdf.set_xlabel("x")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("(a) CDF: Horizontal, Angular, and Vertical Combining")
    ax_cdf.legend(loc="upper left", fontsize=9)
    ax_cdf.set_xlim(-0.5, 0.5)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(True, alpha=0.3)

    ax_pdf.set_xlabel("x")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.set_title("(b) PDF: Horizontal, Angular, and Vertical Combining")
    ax_pdf.legend(loc="upper left", fontsize=9)
    ax_pdf.set_xlim(-0.5, 0.5)
    ax_pdf.set_ylim(0, None)
    ax_pdf.grid(True, alpha=0.3)

    fig.suptitle(
        "Combining Two Gaussians: N(−0.15, 0.1²) and N(0.15, 0.1²), "
        "equal weights\n(cf. Taylor & Meng 2025, Figures 1-5)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "combining_cdf_pdf.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_multi_angle():
    """Generate CDF/PDF for multiple angles (0, 30, 45, 70, 85, 90)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    degrees = [0.0, 30.0, 45.0, 70.0, 85.0, 90.0]
    cmap = plt.cm.coolwarm
    norm_deg = plt.Normalize(0, 90)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_cdf, ax_pdf = axes

    x_grid = np.linspace(-0.5, 0.5, 500)
    for mu in [MU_1, MU_2]:
        ax_cdf.plot(x_grid, norm.cdf(x_grid, loc=mu, scale=SIGMA),
                    color="gray", alpha=0.3, linewidth=1)
        ax_pdf.plot(x_grid, norm.pdf(x_grid, loc=mu, scale=SIGMA),
                    color="gray", alpha=0.3, linewidth=1)

    for deg in degrees:
        q_combined = _combine_at_degree(deg)
        color = cmap(norm_deg(deg))

        ax_cdf.plot(q_combined, TAU, color=color, linewidth=1.5,
                    label=f"θ = {deg:.0f}°")

        samples = np.interp(
            np.random.default_rng(0).uniform(0, 1, 50000), TAU, q_combined
        )
        kde = gaussian_kde(samples)
        ax_pdf.plot(x_grid, kde(x_grid), color=color, linewidth=1.5,
                    label=f"θ = {deg:.0f}°")

    ax_cdf.set_xlabel("x")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("(a) CDF at various angles")
    ax_cdf.legend(fontsize=8)
    ax_cdf.set_xlim(-0.5, 0.5)
    ax_cdf.grid(True, alpha=0.3)

    ax_pdf.set_xlabel("x")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.set_title("(b) PDF at various angles")
    ax_pdf.legend(fontsize=8)
    ax_pdf.set_xlim(-0.5, 0.5)
    ax_pdf.set_ylim(0, None)
    ax_pdf.grid(True, alpha=0.3)

    fig.suptitle(
        "Angular Combining at Multiple Angles "
        "(cf. Taylor & Meng 2025, Figure 5)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "combining_multi_angle.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    plot_cdf_pdf()
    plot_multi_angle()
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
