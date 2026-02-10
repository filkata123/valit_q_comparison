import pandas as pd
import re
import string

# Load CSV
for ex_num in [0]: #range(17):
    if ex_num in [1, 6, 10, 11]:
        continue
    file = f"examples_i5-12400F_16GBRam_RTX3060/example_{ex_num}_stochastic_results_10_samples.csv"
    df = pd.read_csv(file)

    
    def extract_success_prob(name: str) -> str:
        """
        Extract success probability from Algorithm column.
        Example: '(0.999 success)' -> '0.999'
        """
        m = re.search(r"\((0\.\d+)\s+success\)", name)
        return m.group(1) if m else "unknown"

    def extract_alpha_epsilon(name: str):
        alpha = re.search(r"alpha\s*=\s*([^,)\s]+)", name)
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
            # if alpha != "0.1":
            #     alpha = "\\frac{1}{n(x,u)}^{(0.7)}"
            return rf"Q-learning ($\rho={alpha}, \epsilon={epsilon}$)"

        return name

    # ---------- Add parsed columns ----------

    df["SuccessProb"] = df["Algorithm"].apply(extract_success_prob)
    df["PrettyAlg"] = df["Algorithm"].apply(pretty_algorithm_name)

    df = df.drop_duplicates()

    for prob, subdf in df.groupby("SuccessProb"):
        latex_lines = []
        latex_lines.append(r"\begin{table}[!h]")
        latex_lines.append(rf"\caption{{Comparing performance of \ql{{}} as we change the learning rate $\rho$ in stochastic problem with $\gamma = {prob}$ (Problem {ex_num}). Two versions of value iteration are also appended for contrast}}")
        latex_lines.append(fr"\label{{tab:prob{ex_num}stoch{prob}}}")
        latex_lines.append(r"\resizebox{\textwidth}{!}{")
        latex_lines.append(r"\begin{tabular}{lccccccccc}")
        latex_lines.append(r"\hline")

        latex_lines.append(
            rf"\makecell{{Algorithm (Problem {ex_num}) \\ ($\gamma$ = {prob})}} "
            r"& \makecell{Runtime \\ (mean $\pm$ std)}"
            r"& \makecell{\#actions \\ (mean $\pm$ std)} "
            r"& Convergence "
            r"& \makecell{Discover goal \\ time \\ (mean $\pm$ std)} "
            r"& \makecell{Discover goal \\ \#actions \\ (mean $\pm$ std) }"
            r"& \makecell{Optimal Initial \\ Cost2Go Time \\ (mean $\pm$ std)} "
            r"& \makecell{Optimal Initial \\ Cost2Go \#actions \\ (mean $\pm$ std)}"
            r"& \makecell{Optimal Initial \\ Cost2Go \\ Convergence}"
            r"& \makecell{Shortest/Longest \\ Path} \\"
        )

        latex_lines.append(r"\hline")

        # Enumerate algorithms
        for _, row in subdf.iterrows():
            is_value_iteration = "value iteration" in row["Algorithm"].lower()

            runtime_col = (
                f"{row['Avg Time']:.5f} $\\pm$ {row['STD Time']:.4f}"
            )
            actions_col = (
                f"{row['Avg action count']:.4f} $\\pm$ {row['STD action count']:.4f}"
            )

            if is_value_iteration:
                latex_lines.append(
                    f"{row['PrettyAlg']} & "
                    f"{runtime_col} & "
                    f"{actions_col} & "
                    "N/A & N/A & N/A & N/A & N/A & N/A & "
                    f"{row['Shortest Path']}/{row['Longest Path']} \\\\"
                )
            else:
                latex_lines.append(
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
        latex_lines.append(r"\hline")
        latex_lines.append(r"\end{tabular}}")
        latex_lines.append(r"\end{table}")

        print("\n".join(latex_lines))
        print("\n\n")