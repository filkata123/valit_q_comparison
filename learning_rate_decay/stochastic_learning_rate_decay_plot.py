import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# =========================
# User parameters
# =========================
CSV_PATH = "stochastic_learning_rate_decay_results_10_samples.csv"
CONVERGENCE_COL = "Avg Iterations"   # or Avg Time / Avg action count
JITTER_Y = 0.02

# =========================
# Parsing helpers
# =========================
def extract_success_rate(s):
    m = re.search(r"\((0\.[0-9]+)\s+success\)", s)
    return float(m.group(1)) if m else None


def extract_decay_function(s):
    m = re.search(r"decay_function\s*=\s*([a-zA-Z_]+)", s)
    return m.group(1) if m else "static"


def extract_param_signature(s):
    params = re.findall(r"(alpha_zero|omega|decay_rate)\s*=\s*([0-9.]+)", s)
    return "|".join(f"{k}={v}" for k, v in params) if params else "default"


# =========================
# Load & prepare
# =========================
df = pd.read_csv(CSV_PATH)

df["success_prob"] = df["Algorithm"].apply(extract_success_rate)
df["decay_function"] = df["Algorithm"].apply(extract_decay_function)
df["param_sig"] = df["Algorithm"].apply(extract_param_signature)

# Only successful runs matter for convergence
df = df[df["Goal reached"] == True]
df = df.dropna(subset=[CONVERGENCE_COL, "success_prob"])

# Aggregate per configuration
agg = (
    df
    .groupby(["decay_function", "param_sig", "success_prob"])
    .agg(avg_conv=(CONVERGENCE_COL, "mean"))
    .reset_index()
)

# =========================
# Plot per stochasticity level
# =========================
success_levels = sorted(agg["success_prob"].unique())
markers = ["o", "s", "^", "D", "v", "P", "X"]

for success in success_levels:
    subset = agg[agg["success_prob"] == success]

    plt.figure(figsize=(9, 6))

    for i, decay in enumerate(subset["decay_function"].unique()):
        d = subset[subset["decay_function"] == decay]

        # Visual separation only
        y = np.full(len(d), i) + np.random.uniform(-JITTER_Y, JITTER_Y, size=len(d))

        plt.scatter(
            d["avg_conv"],
            y,
            s=70,
            marker=markers[i % len(markers)],
            alpha=0.7,
            label=decay
        )

        # Highlight fastest configuration for this decay
        best = d.loc[d["avg_conv"].idxmin()]
        plt.scatter(
            best["avg_conv"],
            i,
            s=200,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

    plt.yticks(range(len(subset["decay_function"].unique())),
               subset["decay_function"].unique())

    plt.xlabel(f"Average convergence ({CONVERGENCE_COL}) ↓")
    plt.title(f"Fastest successful convergence (success probability = {success})")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()
