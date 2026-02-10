import re
import pandas as pd
import matplotlib.pyplot as plt

# -------- Configuration --------
EPSILON_REGEX = re.compile(r"epsilon\s*=\s*([0-9]*\.?[0-9]+)")
TIME_COLUMN = "Avg Time Goal Discovered"
ALGO_COLUMN = "Algorithm"

SUCCESS_REGEX = re.compile(r"\((\d*\.?\d+)\s*success\)")
RHO_REGEX = re.compile(r"alpha\s*=\s*([0-9]*\.?[0-9]+)")
EPSILON_REGEX = re.compile(r"epsilon\s*=\s*([0-9]*\.?[0-9]+)")


TARGET_SUCCESS = 0.999

files = []
full = range(0,17)
different = [3,9,10,12]
without = [x for x in full if x not in different]

for i in full:
    files.append(f"example_{i}_stochastic_results_10_samples.csv")

if not files:
    raise FileNotFoundError("No matching CSV files found.")

records = []

# -------- Load and collect data --------
for file in files:
    example_id = file.replace("_stochastic_results_10_samples.csv", "")

    df = pd.read_csv(file)

    for _, row in df.iterrows():
        algo_name = str(row.get(ALGO_COLUMN, ""))

        # ---- Filter by success probability ----
        success_match = SUCCESS_REGEX.search(algo_name)
        if not success_match:
            continue

        success_prob = float(success_match.group(1))
        if success_prob != TARGET_SUCCESS:
            continue

        # ---- Extract rho (learning rate = alpha) ----
        rho_match = RHO_REGEX.search(algo_name)
        if not rho_match:
            continue

        rho = float(rho_match.group(1))

        # ---- Extract epsilon ----
        eps_match = EPSILON_REGEX.search(algo_name)
        if not eps_match:
            continue

        epsilon = float(eps_match.group(1))

        time_to_goal = row.get(TIME_COLUMN)
        if pd.isna(time_to_goal):
            continue

        # ---- Clean algorithm name ----
        algo_base = algo_name
        algo_base = SUCCESS_REGEX.sub("", algo_base)
        algo_base = RHO_REGEX.sub("", algo_base)
        algo_base = EPSILON_REGEX.sub("", algo_base)
        algo_base = algo_base.strip(" ,()")

        records.append({
            "example": example_id,
            "algorithm": algo_base,
            "rho": rho,          # learning rate
            "epsilon": epsilon,
            "time": time_to_goal
        })  

data = pd.DataFrame(records)

if data.empty:
    raise ValueError("No valid epsilon-based data found.")

# -------- Normalize per (example, algorithm) --------
data["normalized_time"] = data.groupby(
    ["example", "algorithm", "rho"]
)["time"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
    if x.max() != x.min() else 0.0
)


data["example_num"] = data["example"].str.extract(r"example_(\d+)").astype(int)

#print(data.loc[(data["rho"] == 0.5) & (data["example_num"] == 7)].sort_values("epsilon"))

# -------- Small multiples: one subplot per rho --------
rhos = sorted(data["rho"].unique())
n_rho = len(rhos)

fig, axes = plt.subplots(
    1, n_rho,
    figsize=(6.5 * n_rho, 8),
    sharex=True,
    sharey=True
)

# If there's only one rho, axes is not iterable
if n_rho == 1:
    axes = [axes]

for ax, rho in zip(axes, rhos):
    rho_data = data[data["rho"] == rho]

    seen_examples = set()

    for (example, algorithm), group in rho_data.groupby(
        ["example_num", "algorithm"]
    ):
        group = group.sort_values("epsilon")

        label = example if example not in seen_examples else None
        seen_examples.add(example)

        ax.plot(
            group["epsilon"],
            group["normalized_time"],
            marker="o",
            alpha=0.7,
            label=label
        )
    ax.set_title(r"$\rho$ = " + str(rho), fontsize=18)
    ax.grid(True)


# -------- Shared labels and formatting --------
fig.supxlabel(r"$\epsilon$", fontsize=20)
fig.supylabel("Normalized Avg Time Goal Discovered", fontsize=20, horizontalalignment='right')
fig.suptitle(
    "Effect of $\\epsilon$ for Different Learning Rates ($\\rho$)\n"
    f"(Success probability = {TARGET_SUCCESS})",
    fontsize=24
)

epsilon_ticks = sorted(data["epsilon"].unique())
for ax in axes:
    ax.set_xticks(epsilon_ticks)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

# Legend only once (from first subplot)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Problem",
    fontsize=12,
    title_fontsize=16,
    loc="upper right"
)

plt.tight_layout(rect=[0, 0, 0.92, 0.92])
plt.show()