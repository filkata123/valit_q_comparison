import pandas as pd
import re

PROBLEMS = [1, 10, 11]

def extract_success_prob(name: str) -> str:
    m = re.search(r"\((0\.\d+)\s+success\)", name)
    return m.group(1) if m else "unknown"

def extract_alpha_epsilon(name: str):
    alpha = re.search(r"alpha\s*=\s*([^,)\s]+)", name)
    epsilon = re.search(r"epsilon\s*=\s*([^,)\s]+)", name)
    return (
        alpha.group(1) if alpha else None,
        epsilon.group(1) if epsilon else None,
    )

def pretty_algorithm_name(name: str):
    name_lower = name.lower()

    if "async value iteration" in name_lower:
        return "Async Value Iteration"

    if "value iteration" in name_lower:
        return "Value Iteration"

    if "q-learning" in name_lower:
        alpha, epsilon = extract_alpha_epsilon(name_lower)
        if epsilon == "decaying":
            eps_str = r"$\epsilon$-decay"
        else:
            eps_str = rf"$\epsilon={epsilon}$"
        return rf"Q-learning ($\rho={alpha}$, {eps_str})"

    return name

# Load and tag all problems
all_dfs = []
for ex_num in PROBLEMS:
    file = f"example_{ex_num}_stochastic_results_10_samples.csv"
    df = pd.read_csv(file)
    df["Problem"] = ex_num
    df["SuccessProb"] = df["Algorithm"].apply(extract_success_prob)
    df["PrettyAlg"] = df["Algorithm"].apply(pretty_algorithm_name)
    df = df.drop_duplicates()
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)

# One table per gamma value
for prob, group_df in combined.groupby("SuccessProb"):
    latex_lines = []
    latex_lines.append(r"\begin{table}[!h]")
    latex_lines.append(
        rf"\caption{{Performance of dynamic programming methods and \ql{{}} as we change the learning rate $\rho$ across Problems {', '.join(str(p) for p in PROBLEMS)} "
        rf"with $\gamma = {prob}$ and $\epsilon$-decay.}}"
    )
    latex_lines.append(rf"\label{{tab:stoch{prob}_eps_decay_all_problems}}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{llccccccccc}")
    latex_lines.append(r"\hline")

    latex_lines.append(
        r"\makecell{Problem} "
        rf"& \makecell{{Algorithm \\ ($\gamma$ = {prob})}} "
        r"& \makecell{Runtime \\ (mean $\pm$ std)}"
        r"& \makecell{\#actions \\ (mean $\pm$ std)} "
        r"& Convergence "
        r"& \makecell{Discover goal \\ time \\ (mean $\pm$ std)} "
        r"& \makecell{Discover goal \\ \#actions \\ (mean $\pm$ std)} "
        r"& \makecell{Optimal Initial \\ Cost2Go Time \\ (mean $\pm$ std)} "
        r"& \makecell{Optimal Initial \\ Cost2Go \#actions \\ (mean $\pm$ std)}"
        r"& \makecell{Optimal Initial \\ Cost2Go \\ Convergence}"
        r"& \makecell{Shortest/Longest \\ Path} \\"
    )
    latex_lines.append(r"\hline")

    # Group by problem, and within each problem list all its algorithms
    for problem_num, problem_df in group_df.groupby("Problem"):
        first_row = True
        n_rows = len(problem_df)

        for _, row in problem_df.iterrows():
            is_value_iteration = "value iteration" in row["Algorithm"].lower()

            runtime_col = f"{row['Avg Time']:.5f} $\\pm$ {row['STD Time']:.4f}"
            actions_col = f"{row['Avg action count']:.4f} $\\pm$ {row['STD action count']:.4f}"

            # Use \multirow for the problem number, spanning all its algorithm rows
            if first_row:
                problem_cell = rf"\multirow{{{n_rows}}}{{*}}{{Problem {problem_num}}}"
                first_row = False
            else:
                problem_cell = ""

            if is_value_iteration:
                latex_lines.append(
                    f"{problem_cell} & "
                    f"{row['PrettyAlg']} & "
                    f"{runtime_col} & "
                    f"{actions_col} & "
                    "N/A & N/A & N/A & N/A & N/A & N/A & "
                    f"{row['Shortest Path']}/{row['Longest Path']} \\\\"
                )
            else:
                latex_lines.append(
                    f"{problem_cell} & "
                    f"{row['PrettyAlg']} & "
                    f"{runtime_col} & "
                    f"{actions_col} & "
                    f"{100 * row['Convergence rate']:.1f}\\% & "
                    f"{row['Avg Time Goal Discovered']:.4f} $\\pm$ {row['STD Time Goal Discovered']:.4f} & "
                    f"{row['Avg Actions Goal Discovered']:.4f} $\\pm$ {row['STD Actions Goal Discovered']:.4f} & "
                    f"{row['Avg Time Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Time Optimal Initial Cost2Go']:.4f} & "
                    f"{row['Avg Actions Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Actions Optimal Initial Cost2Go']:.4f} & "
                    f"{100 * row['Convergence rate Optimal Initial Cost2Go']:.1f}\\% & "
                    f"{row['Shortest Path']}/{row['Longest Path']} \\\\"
                )

        # Horizontal rule after each problem block
        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")

    print("\n".join(latex_lines))
    print("\n\n")