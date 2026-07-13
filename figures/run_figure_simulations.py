import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from covtest.simulation.generate_covariances import (
    generate_spiked_covariance,
    generate_toeplitz_cov,
)
from covtest.methods.hypothesis_identity import (
    ledoit_wolf_identity,
    chen_2010_identity,
    srivastava_2014_identity,
)
from covtest.methods.hypothesis_spherical import (
    bartlett_sphericity_test,
    srivastava_2005_sphericity,
    czz_sphericity_test,
)
from covtest.methods.hypothesis_proportionality import (
    flury_proportionality_test,
    proportionality_test_LZ,
    proportional_cov_test_tsukuda,
)
from covtest.methods.hypothesis_two_sample import (
    boxm_test,
    srivastava_two_sample_2007,
    schott2007,
)

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Global parameters
RNG_SEED = 42
N_SIM = 500       # replications for null calibration
N_SIM_PWR = 300   # replications for power curve
ALPHA = 0.05
P = 20            # dimension (fixed throughout)
N_NULL = 200      # sample size for null calibration panels
N_VALUES = [50, 75, 100, 150, 200, 300, 400]

# Pastel palette, avoiding yellow and peach.
COLORS = ["#5B8DB8", "#78B7A5", "#9C8AC7"]
MARKERS = ["o", "s", "D"]

# Figure-wide styling. Keep this centralized so the plotting style is easy
# to reuse across other simulation figures.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.4,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Method lists
identity_methods = [
    ("Ledoit-Wolf (2002)", ledoit_wolf_identity),
    ("Chen et al. (2010)", chen_2010_identity),
    ("Srivastava et al. (2014)", srivastava_2014_identity),
]

sphericity_methods = [
    ("Bartlett (1937)", bartlett_sphericity_test),
    ("Srivastava (2005)", srivastava_2005_sphericity),
    ("Chen-Zhang-Zhong (2010)", czz_sphericity_test),
]

proportionality_methods = [
    ("Liu et al. (2014)", proportionality_test_LZ),
    ("Tsukuda & Matsuura (2019)", proportional_cov_test_tsukuda),
]

two_sample_methods = [
    ("Box M (1953)", boxm_test),
    ("Srivastava (2007)", srivastava_two_sample_2007),
    ("Schott (2007)", schott2007),
]

all_columns = [
    ("Identity", identity_methods),
    ("Sphericity", sphericity_methods),
    ("Proportionality", proportionality_methods),
    ("Two-Sample", two_sample_methods),
]

# Covariance matrices
Sigma_base = generate_toeplitz_cov(P, rho=0.3)
L_base = np.linalg.cholesky(Sigma_base)

# Identity alternative: spiked model
Sigma_alt_identity = generate_spiked_covariance(
    P,
    spike_eigenvalue=1.8,
    num_spikes=2,
    base_variance=1.0,
    rng=np.random.default_rng(RNG_SEED),
)
L_alt_identity = np.linalg.cholesky(Sigma_alt_identity)

# Sphericity alternative: Toeplitz rho=0.15
Sigma_alt_sphericity = generate_toeplitz_cov(P, rho=0.15)
L_alt_sphericity = np.linalg.cholesky(Sigma_alt_sphericity)

# Proportionality alternative: Y is Toeplitz rho=0.12
Sigma_alt_prop = generate_toeplitz_cov(P, rho=0.12)
L_alt_prop = np.linalg.cholesky(Sigma_alt_prop)

# Two-sample alternative: Y is Toeplitz rho=0.45
Sigma_alt_two_sample = generate_toeplitz_cov(P, rho=0.45)
L_alt_two_sample = np.linalg.cholesky(Sigma_alt_two_sample)


def style_axis(ax):
    """Apply the user's preferred publication-style axis formatting."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.spines["left"].set_linewidth(1.6)
    ax.tick_params(axis="both", which="major", direction="out", length=4.5, width=1.2)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=1.0)


def generate_sim_data(col_idx, n, null_status, rng):
    """Generate data for each covariance test class."""
    if col_idx == 0:  # Identity
        if null_status:
            X = rng.normal(size=(n, P))
        else:
            X = rng.normal(size=(n, P)) @ L_alt_identity.T
        return (X,)
    if col_idx == 1:  # Sphericity
        if null_status:
            X = rng.normal(size=(n, P)) * np.sqrt(3.0)
        else:
            X = rng.normal(size=(n, P)) @ L_alt_sphericity.T
        return (X,)
    if col_idx == 2:  # Proportionality
        if null_status:
            X = rng.normal(size=(n, P)) @ L_base.T
            Y = rng.normal(size=(n, P)) @ (np.sqrt(2.0) * L_base).T
        else:
            X = rng.normal(size=(n, P)) @ L_base.T
            Y = rng.normal(size=(n, P)) @ L_alt_prop.T
        return X, Y
    if col_idx == 3:  # Two-sample
        if null_status:
            X = rng.normal(size=(n, P)) @ L_base.T
            Y = rng.normal(size=(n, P)) @ L_base.T
        else:
            X = rng.normal(size=(n, P)) @ L_base.T
            Y = rng.normal(size=(n, P)) @ L_alt_two_sample.T
        return X, Y
    raise ValueError("Invalid column index")


fig, axes = plt.subplots(
    2,
    4,
    figsize=(17.5, 8.8),
    constrained_layout=True,
)

fig.set_constrained_layout_pads(
    hspace=0.1,   # increase vertical space between top and bottom rows
    wspace=0.05
)

#fig.suptitle(
#    "Null calibration and empirical power for covariance test classes",
#    fontsize=16,
#    fontweight="semibold",
#    y=1.03,
#)

# Run simulations and fill plots.
for col_idx, (col_label, methods) in enumerate(all_columns):
    ax_null = axes[0, col_idx]
    ax_pwr = axes[1, col_idx]
    band_drawn = False

    # Reference line for the nominal level in the power panels.
    ax_pwr.axhline(
        ALPHA,
        linestyle=(0, (4, 3)),
        color="#6F6F6F",
        linewidth=1.2,
        alpha=0.85,
        zorder=1,
    )
    ax_pwr.text(
        0.985,
        ALPHA + 0.025,
        r"$\alpha = 0.05$",
        transform=ax_pwr.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#5A5A5A",
    )

    for m_idx, (m_label, fn) in enumerate(methods):
        color = COLORS[m_idx]
        marker = MARKERS[m_idx]

        # Row 0: null calibration.
        null_pvals = []
        for sim_idx in range(N_SIM):
            rng = np.random.default_rng(RNG_SEED + sim_idx)
            args = generate_sim_data(col_idx, N_NULL, null_status=True, rng=rng)

            if fn.__name__ == "proportionality_test_LZ" and N_NULL <= P + 1:
                pval = np.nan
            elif fn.__name__ == "boxm_test" and N_NULL <= P:
                pval = np.nan
            else:
                try:
                    res = fn(*args)
                    pval = float(res["p_value"])
                    if not (0.0 <= pval <= 1.0):
                        pval = np.nan
                except Exception:
                    pval = np.nan
            null_pvals.append(pval)

        valid_null = [pv for pv in null_pvals if np.isfinite(pv)]
        if len(valid_null) >= 50:
            obs = np.sort(valid_null)
            expected = np.arange(1, len(obs) + 1) / (len(obs) + 1)

            if not band_drawn:
                eps = 1.36 / np.sqrt(len(obs))
                lo = np.clip(expected - eps, 0, 1)
                hi = np.clip(expected + eps, 0, 1)
                ax_null.fill_between(
                    expected,
                    lo,
                    hi,
                    color="#D9DEE8",
                    alpha=0.85,
                    linewidth=0,
                    label="KS band",
                    zorder=0,
                )
                band_drawn = True

            ax_null.plot(
                expected,
                obs,
                color=color,
                label=m_label,
                linewidth=2.2,
                alpha=0.96,
                zorder=2,
            )

        # Row 1: power versus sample size.
        power_values = []
        for n_val in N_VALUES:
            if fn.__name__ == "proportionality_test_LZ" and n_val <= P + 1:
                power_values.append(np.nan)
                continue
            if fn.__name__ == "boxm_test" and n_val <= P:
                power_values.append(np.nan)
                continue

            pvals_pwr = []
            for sim_idx in range(N_SIM_PWR):
                rng = np.random.default_rng(RNG_SEED + sim_idx)
                args = generate_sim_data(col_idx, n_val, null_status=False, rng=rng)
                try:
                    res = fn(*args)
                    pval = float(res["p_value"])
                    if not (0.0 <= pval <= 1.0):
                        pval = np.nan
                except Exception:
                    pval = np.nan
                pvals_pwr.append(pval)

            valid_pwr = [pv for pv in pvals_pwr if np.isfinite(pv)]
            if valid_pwr:
                power_values.append(np.mean(np.array(valid_pwr) < ALPHA))
            else:
                power_values.append(np.nan)

        power_values = np.array(power_values)
        valid_mask = np.isfinite(power_values)
        ax_pwr.plot(
            np.array(N_VALUES)[valid_mask],
            power_values[valid_mask],
            color=color,
            label=m_label,
            marker=marker,
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=2.2,
            alpha=0.96,
            zorder=2,
        )

    # Formatting row 0: null calibration.
    ax_null.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 3)),
        color="#6F6F6F",
        linewidth=1.1,
        alpha=0.85,
        zorder=1,
    )
    ax_null.set_xlim(0, 1)
    ax_null.set_ylim(0, 1)
    ax_null.set_aspect("equal", adjustable="box")
    ax_null.set_xticks(np.linspace(0, 1, 5))
    ax_null.set_yticks(np.linspace(0, 1, 5))
    ax_null.set_xlabel("Expected p-value",fontsize=20)
    if col_idx == 0:
        ax_null.set_ylabel("Observed p-value",fontsize=20)
    else:
        ax_null.set_ylabel("")
    ax_null.set_title(col_label, pad=12, fontsize=24)
    ax_null.legend(
        loc="upper left",
        frameon=False,
        borderaxespad=0.3,
        handlelength=1.8,
        handletextpad=0.55,
    )
    style_axis(ax_null)

    # Formatting row 1: power.
    ax_pwr.set_xlim(min(N_VALUES), max(N_VALUES))
    ax_pwr.set_ylim(0, 1.05)
    ax_pwr.set_xticks(N_VALUES)
    ax_pwr.set_yticks(np.linspace(0, 1, 5))
    ax_pwr.set_xlabel("Sample size, n",fontsize=20)
    if col_idx == 0:
        ax_pwr.set_ylabel("Empirical power",fontsize=20)
    else:
        ax_pwr.set_ylabel("")
    style_axis(ax_pwr)

fig.savefig("figures/figure_simulations_nicer.pdf")
fig.savefig("figures/figure_simulations_nicer.png")
print("Saved figures/figure_simulations_nicer.pdf and figures/figure_simulations_nicer.png successfully!")
