import pandas as pd
import re
import string

dfs = []

for ex_num in [1, 10]:
    file = f"example_{ex_num}_stochastic_results_100_samples.csv"
    df = pd.read_csv(file)
    df["Problem"] = ex_num
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


def q_rho_099_ret(name: str) -> bool:
    """
    Keep:
    - all non–Q-learning algorithms
    - Q-learning only if rho (alpha) == 0.99
    """
    name_lower = name.lower()
    if "q-learning" not in name_lower:
        return True

    alpha, _ = extract_alpha_epsilon(name_lower)
    return alpha == "0.99"

def q_rho_099_and_05_ret(name: str) -> bool:
    """
    Keep:
    - all non–Q-learning algorithms
    - Q-learning only if:
        rho (alpha) in {0.5, 0.99}
        epsilon in {0, 0.5, 1}
    """
    name_lower = name.lower()

    if "q-learning" not in name_lower:
        return True

    alpha, epsilon = extract_alpha_epsilon(name_lower)

    if alpha is None or epsilon is None:
        return False

    valid_alphas = {"0.5", "0.99"}
    valid_epsilons = {"0", "0.5", "1"}

    return alpha in valid_alphas and epsilon in valid_epsilons


def extract_success_prob(name: str) -> str:
    """
    Extract success probability from Algorithm column.
    Example: '(0.999 success)' -> '0.999'
    """
    m = re.search(r"\((0\.\d+)\s+success\)", name)
    return m.group(1) if m else "unknown"

def extract_alpha_epsilon(name: str):
    alpha = re.search(r"alpha\s*=\s*([0-9.]+)", name)
    epsilon = re.search(r"epsilon\s*=\s*([0-9.]+)", name)
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
        return rf"Q-learning ($\rho={alpha}, \epsilon={epsilon}$)"

    return name

# ---------- Add parsed columns ----------

df["SuccessProb"] = df["Algorithm"].apply(extract_success_prob)
df["PrettyAlg"] = df["Algorithm"].apply(pretty_algorithm_name)
df = df[df["Algorithm"].apply(q_rho_099_and_05_ret)]

df = df.drop_duplicates()

for prob, subdf in df.groupby("SuccessProb"):
    latex_lines = []
    latex_lines.append(r"\begin{table}[!h]")
    latex_lines.append(
        rf"\caption{{Comparing performance across Problems 1, 10"
        rf"for stochastic problem with $\gamma = {prob}$. "
        rf"Only Q-learning with $\rho=0.99$ is shown; all $\epsilon$ values included.}}"
    )
    latex_lines.append(fr"\label{{tab:stoch_gamma_{prob}}}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{lccccc}")
    latex_lines.append(r"\hline")

    latex_lines.append(
        r"\makecell{Algorithm \\ ($\gamma= " + prob + r"$)} "
        r"& \makecell{Runtime \\ (mean $\pm$ std)}"
        r"& Convergence "
        r"& \makecell{Optimal Initial \\ Cost2Go Time \\ (mean $\pm$ std)} "
        r"& \makecell{Optimal Initial \\ Cost2Go \\ Convergence}"
        r"& \makecell{Shortest/Longest \\ Path} \\"
    )

    latex_lines.append(r"\hline")

    for _, row in subdf.iterrows():
        is_value_iteration = "value iteration" in row["Algorithm"].lower()

        runtime_col = (
            f"{row['Avg Time']:.5f} $\\pm$ {row['STD Time']:.4f}"
        )

        if is_value_iteration:
            latex_lines.append(
                f"(Pr {row['Problem']}) {row['PrettyAlg']} & "
                f"{runtime_col} & "
                "N/A & N/A & N/A & "
                f"{row['Shortest Path']}/{row['Longest Path']} \\\\"
            )
        else:
            latex_lines.append(
                f"(Pr {row['Problem']}) {row['PrettyAlg']} & "
                f"{runtime_col} & "
                f"{100 * row['Convergence rate']:.1f}\\% & "
                f"{row['Avg Time Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Time Optimal Initial Cost2Go']:.4f} & "
                f"{100 * row['Convergence rate Optimal Initial Cost2Go']:.1f}\\% & "
                f"{row['Shortest Path']}/{row['Longest Path']} \\\\"
            )

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")

    print("\n".join(latex_lines))
    print("\n\n")