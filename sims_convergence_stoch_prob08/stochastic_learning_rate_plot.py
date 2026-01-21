import pandas as pd
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# =========================
# User parameters
# =========================
CSV_PATH = "5kep_stochastic_learning_rate_results_100_samples.csv"   # path to your CSV file

# =========================
# Helper functions
# =========================
def extract_alpha_and_success(algorithm_str):
    """
    Extract learning rate (alpha) and Probability of Successful Transition
    from the Algorithm column.
    """
    alpha_match = re.search(r"alpha\s*=\s*([0-9.]+)", algorithm_str)
    success_match = re.search(r"\((0\.[0-9]+)\s+success\)", algorithm_str)

    alpha = float(alpha_match.group(1)) if alpha_match else None
    success = float(success_match.group(1)) if success_match else None

    return alpha, success


def plot_3d(df, z_col, z_label, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Separate converged vs not converged
    reached = df[df["Goal reached"] == True]
    not_reached = df[df["Goal reached"] == False]

    # Create grid
    alphas = np.sort(reached["alpha_idx"].unique())
    successes = np.sort(reached["success_idx"].unique())

    A, S = np.meshgrid(alphas, successes)

    Z = np.full(A.shape, np.nan)

    for _, row in reached.iterrows():
        i = np.where(successes == row["success_idx"])[0][0]
        j = np.where(alphas == row["alpha_idx"])[0][0]
        Z[i, j] = row[z_col]

    # Surface plot (only successful runs)
    ax.plot_surface(
        A, S, Z,
        alpha=1,
        edgecolor="k",
        linewidth=0.3,
        color="red",
    )

    # Overlay failed runs
    ax.scatter(
        not_reached["alpha_idx"],
        not_reached["success_idx"],
        not_reached[z_col],
        c="red",
        marker="x",
        s=60,
        label="Goal NOT reached"
    )

    ax.set_xticks(range(len(alpha_order)))
    ax.set_xticklabels(alpha_order)

    ax.set_yticks(range(len(success_order)))
    ax.set_yticklabels(success_order)

    ax.view_init(elev=30, azim=135)
    ax.set_xlabel("Learning rate (alpha)")
    ax.set_ylabel("Probability of Successful Transition")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.legend()
    plt.show()


# =========================
# Load and parse data
# =========================
df = pd.read_csv(CSV_PATH)

df[["alpha", "success_prob"]] = df["Algorithm"].apply(
    lambda x: pd.Series(extract_alpha_and_success(x))
)

alpha_order = sorted(df["alpha"].unique())
success_order = sorted(df["success_prob"].unique())

df["alpha_idx"] = df["alpha"].map({v: i for i, v in enumerate(alpha_order)})
df["success_idx"] = df["success_prob"].map({v: i for i, v in enumerate(success_order)})

# =========================
# Generate plots
# =========================
plot_3d(
    df,
    z_col="Avg Iterations",
    z_label="Average iterations (episodes)",
    title="Avg Iterations vs Alpha and Probability of Successful Transition (epsilon = 0.9)"
)

plot_3d(
    df,
    z_col="Avg Time",
    z_label="Average time (seconds)",
    title="Avg Time vs Alpha and Probability of Successful Transition (epsilon = 0.9)"
)

plot_3d(
    df,
    z_col="Avg action count",
    z_label="Average action count",
    title="Avg Action Count vs Alpha and Probability of Successful Transition (epsilon = 0.9)"
)

plot_3d(
    df,
    z_col="Avg convergence action",
    z_label="Average convergence action (100k = no conv)",
    title="Avg Convergence Action vs Alpha and Probability of Successful Transition (epsilon = 0.9)"
)
