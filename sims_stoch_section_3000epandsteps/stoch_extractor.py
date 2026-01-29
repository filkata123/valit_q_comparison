import pandas as pd
import re
import string

# Load CSV
for ex_num in [1,10,11]:#range(17):
    if ex_num == 6:
        continue
    file = f"example_{ex_num}_stochastic_results_2_samples.csv"
    df = pd.read_csv(file)

    
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
            return rf"Q-learning ($\alpha={alpha}, \epsilon={epsilon}$)"

        return name

    # ---------- Add parsed columns ----------

    df["SuccessProb"] = df["Algorithm"].apply(extract_success_prob)
    df["PrettyAlg"] = df["Algorithm"].apply(pretty_algorithm_name)

    for prob, subdf in df.groupby("SuccessProb"):
        latex_lines = []
        latex_lines.append(r"\begin{table}[!h]")
        latex_lines.append(r"\resizebox{\textwidth}{!}{")
        latex_lines.append(r"\begin{tabular}{lcccccccc}")
        latex_lines.append(r"\hline")

        latex_lines.append(
            rf"(Prob. Success = {prob}) Algorithm (Problem {ex_num}) "
            r"& Runtime (mean $\pm$ std) "
            r"& \#actions (mean $\pm$ std) "
            r"& Convergence "
            r"& Discover goal time (mean $\pm$ std) "
            r"& Discover goal \#actions (mean $\pm$ std) "
            r"& Optimal Initial Cost2Go Time (mean $\pm$ std) "
            r"& Optimal Initial Cost2Go \#actions (mean $\pm$ std)"
            r"& Optimal Initial Cost2Go Convergence \\"
        )

        latex_lines.append(r"\hline")

        # Enumerate algorithms
        for _, (_, row) in zip(string.ascii_lowercase, subdf.iterrows()):
            latex_lines.append(
                f"{row['PrettyAlg']} & "
                f"{row['Avg Time']:.5f} $\\pm$ {row['STD Time']:.4f} & "
                f"{row['Avg action count']:.4f} $\\pm$ {row['STD action count']:.4f} & "
                f"{100 * row['Convergence rate']:.1f}\\% & "
                f"{row['Avg Time Goal Discovered']:.4f} $\\pm$ {row['STD Time Goal Discovered']:.4f} & "
                f"{row['Avg Actions Goal Discovered']:.4f} $\\pm$ {row['STD Actions Goal Discovered']:.4f} & "
                f"{row['Avg Time Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Time Optimal Initial Cost2Go']:.4f} & "
                f"{row['Avg Actions Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Actions Optimal Initial Cost2Go']:.4f} & "
                f"{100 * row['Convergence rate Optimal Initial Cost2Go']:.1f}\\% & \\\\"
            )

        latex_lines.append(r"\hline")
        latex_lines.append(r"\end{tabular}}")
        latex_lines.append(r"\end{table}")

        print("\n".join(latex_lines))
        print("\n\n")