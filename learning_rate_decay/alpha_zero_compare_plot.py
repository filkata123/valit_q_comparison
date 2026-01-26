import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional but recommended for readability
sns.set(style="whitegrid")

# If loading from CSV
df = pd.read_csv("stochastic_learning_rate_decay_results_10_samples_old.csv")

# Extract alpha_zero from the Algorithm string
df["alpha_zero"] = df["Algorithm"].str.extract(r"alpha_zero\s*=\s*([0-9.]+)").astype(float)

df["omega"] = (
    df["Algorithm"]
    .str.extract(r"omega\s*=\s*([0-9.]+)(?=.*decay_function\s*=\s*inverse_time_decay)")
    .astype(float)
)

# Ensure Goal reached is boolean
df["Goal reached"] = df["Goal reached"].astype(bool)

success_rate = df.groupby("alpha_zero")["Goal reached"].mean()

plt.figure()
success_rate.plot(kind="bar")
plt.ylabel("Success Rate")
plt.xlabel("alpha_zero")
plt.title("Goal Reach Rate by alpha_zero")
plt.ylim(0, 1)
plt.show()

plt.figure()
sns.boxplot(data=df, x="alpha_zero", y="Avg Time")
plt.title("Average Time by alpha_zero")
plt.show()

plt.figure()
sns.boxplot(data=df, x="alpha_zero", y="Avg Iterations")
plt.title("Average Iterations by alpha_zero")
plt.show()

plt.figure()
sns.scatterplot(
    data=df,
    x="Avg Iterations",
    y="Avg Time",
    hue="alpha_zero",
    style="Goal reached"
)

plt.xscale("log")
plt.yscale("log")

plt.title("Avg Time vs Avg Iterations (log–log)")
plt.show()

g = sns.catplot(
    data=df,
    x="alpha_zero",
    y="Avg Time",
    col="omega",
    kind="box",
    col_wrap=3,
    sharey=False
)

g.fig.suptitle("Average Time by alpha_zero, Faceted by omega", y=1.02)
plt.show()

sns.pairplot(
    df,
    vars=[
        "Avg Time",
        "Avg Iterations",
        "Avg action count"
    ],
    hue="alpha_zero",
    corner=True
)
plt.show()

# import pandas as pd
# import re
# import matplotlib.pyplot as plt

# # ---------- CONFIG ----------
# CSV_PATH = "results.csv"   # change if needed

# METRICS = [
#     "Avg Time",
#     "Avg Iterations",
#     "Avg action count",
#     "Avg convergence action"
# ]

# # ---------- LOAD DATA ----------
# df =  pd.read_csv("stochastic_learning_rate_decay_results_10_samples_old.csv")

# # ---------- FILTER USING REGEX ----------
# algo_pattern = re.compile(
#     r"0\.9 success.*decay_function = inverse_time_decay"
# )

# df = df[df["Algorithm"].str.contains(algo_pattern, regex=True)]

# # ---------- EXTRACT PARAMETERS ----------
# def extract_param(pattern, text):
#     match = re.search(pattern, text)
#     return float(match.group(1)) if match else None

# df["alpha_zero"] = df["Algorithm"].apply(
#     lambda x: extract_param(r"alpha_zero\s*=\s*([0-9.]+)", x)
# )
# df["omega"] = df["Algorithm"].apply(
#     lambda x: extract_param(r"omega\s*=\s*([0-9.]+)", x)
# )
# df["decay_rate"] = df["Algorithm"].apply(
#     lambda x: extract_param(r"decay_rate\s*=\s*([0-9.]+)", x)
# )

# # ---------- KEEP ONLY ALPHA 0.1 AND 1 ----------
# df = df[df["alpha_zero"].isin([0.1, 1])]

# # ---------- GROUP AND COMPUTE DIFFERENCES ----------
# group_cols = ["omega", "decay_rate"]

# diff_rows = []

# diff_rows = []
# goal_diff_rows = []

# for (omega, decay_rate), group in df.groupby(["omega", "decay_rate"]):

#     if set(group["alpha_zero"]) != {0.1, 1}:
#         continue

#     row_1 = group[group["alpha_zero"] == 1].iloc[0]
#     row_01 = group[group["alpha_zero"] == 0.1].iloc[0]

#     # ---------- CASE 1: BOTH REACHED GOAL ----------
#     if row_1["Goal reached"] and row_01["Goal reached"]:

#         diff = {
#             "omega": omega,
#             "decay_rate": decay_rate
#         }

#         for m in METRICS:
#             diff[m] = row_1[m] - row_01[m]

#         diff_rows.append(diff)

#     # ---------- CASE 2: GOAL REACHED DIFFERS ----------
#     elif row_1["Goal reached"] != row_01["Goal reached"]:

#         goal_diff_rows.append({
#             "omega": omega,
#             "decay_rate": decay_rate,
#             "alpha_1_goal": row_1["Goal reached"],
#             "alpha_01_goal": row_01["Goal reached"]
#         })

# diff_df = pd.DataFrame(diff_rows)
# goal_diff_df = pd.DataFrame(goal_diff_rows)
# diff_df["label"] = diff_df.apply(
#     lambda r: f"ω={r['omega']}, dr={r['decay_rate']}", axis=1
# )

# goal_diff_df["label"] = goal_diff_df.apply(
#     lambda r: f"ω={r['omega']}, dr={r['decay_rate']}", axis=1
# )

# # ---------- PLOTTING ----------
# plt.figure(figsize=(10, 4))

# y = range(len(goal_diff_df))

# plt.scatter(
#     y,
#     goal_diff_df["alpha_1_goal"].astype(int),
#     label="alpha_zero = 1",
#     marker="o"
# )

# plt.scatter(
#     y,
#     goal_diff_df["alpha_01_goal"].astype(int),
#     label="alpha_zero = 0.1",
#     marker="x"
# )

# plt.yticks([0, 1], ["Not reached", "Reached"])
# plt.xticks(y, goal_diff_df["label"], rotation=45)
# plt.ylabel("Goal reached")
# plt.title("Configurations where Goal Reached differs by alpha_zero")
# plt.legend()
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 4 * len(METRICS)))

# if len(METRICS) == 1:
#     axes = [axes]

# for ax, metric in zip(axes, METRICS):
#     ax.plot(diff_df["label"], diff_df[metric], marker="o")
#     ax.axhline(0, linestyle="--")
#     ax.set_title(f"Difference in {metric} (alpha_zero = 1 − 0.1)")
#     ax.set_ylabel("Difference")
#     ax.set_xlabel("omega / decay_rate")
#     ax.tick_params(axis="x", rotation=45)

# plt.tight_layout()
# plt.show()