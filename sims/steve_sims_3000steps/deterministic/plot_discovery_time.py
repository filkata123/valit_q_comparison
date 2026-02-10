import re
import pandas as pd
import matplotlib.pyplot as plt

# -------- Configuration --------
FILE_PATTERN = "example_*_results_100_samples.csv"
EPSILON_REGEX = re.compile(r"epsilon\s*=\s*([0-9]*\.?[0-9]+)")
TIME_COLUMN = "Avg Time Goal Discovered"
ALGO_COLUMN = "Algorithm"

files = []
full = range(0,17)
different = [3,9,10,12]
without = [x for x in full if x not in different]

for i in without:
    files.append(f"example_{i}_results_100_samples.csv")

if not files:
    raise FileNotFoundError("No matching CSV files found.")

records = []

# -------- Load and collect data --------
for file in files:
    example_id = file.replace("_results_100_samples.csv", "")

    df = pd.read_csv(file)

    for _, row in df.iterrows():
        algo_name = str(row.get(ALGO_COLUMN, ""))

        # Exclude π-based algorithms
        if "Fully-random (deterministic with pi)" in algo_name:
            continue
        if "with pi" in algo_name or "π" in algo_name:
            continue

        match = EPSILON_REGEX.search(algo_name)
        if not match:
            continue

        epsilon = float(match.group(1))
        time_to_goal = row.get(TIME_COLUMN)

        if pd.isna(time_to_goal):
            continue

        algo_base = EPSILON_REGEX.sub("", algo_name).strip(" ,")

        records.append({
            "example": example_id,
            "algorithm": algo_base,
            "epsilon": epsilon,
            "time": time_to_goal
        })

data = pd.DataFrame(records)

if data.empty:
    raise ValueError("No valid epsilon-based data found.")

# -------- Normalize per (example, algorithm) --------
data["normalized_time"] = data.groupby(
    ["example", "algorithm"]
)["time"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
    if x.max() != x.min() else 0.0
)

data["example_num"] = data["example"].str.extract(r"example_(\d+)").astype(int)

# -------- Single combined plot --------
plt.figure(figsize=(12, 7))

seen_examples = set()

for (example, algorithm), group in data.groupby(["example_num", "algorithm"]):
    group = group.sort_values("epsilon")

    # Only label the first occurrence of each example
    label = example if example not in seen_examples else None
    seen_examples.add(example)

    plt.plot(
        group["epsilon"],
        group["normalized_time"],
        marker="o",
        alpha=0.7,
        label=label
    )

plt.xlabel("$\epsilon$", fontsize=20)
plt.ylabel("Normalized Avg Time Goal Discovered", fontsize = 20)
plt.title("Effect of Epsilon on Goal Discovery Time", fontsize = 24)
epsilon_ticks = sorted(data["epsilon"].unique())
plt.xticks(epsilon_ticks, fontsize=16)
plt.legend(title="Problem", fontsize=12, title_fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()
