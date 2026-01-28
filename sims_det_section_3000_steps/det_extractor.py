import pandas as pd
import re

# Load CSV
for ex_num in range(17):
    if ex_num == 6:
        continue
    file = f"example_{ex_num}_results_100_samples.csv"
    df = pd.read_csv(file)

    # ---------- Helper: shorten algorithm names ----------
    def shorten_algorithm_name(name: str) -> str:
        name_lower = name.lower()

        # Model-free Dijkstra: keep name
        if "synchronous" in name_lower:
            return "Model-free \\\ Synchronous Value Iteration"

        # Extract epsilon
        eps_match = re.search(r"epsilon\s*=\s*([0-9.]+)", name_lower)
        epsilon = eps_match.group(1) if eps_match else "?"

        # Deterministic with pi
        if "deterministic" in name_lower and "pi" in name_lower:
            return rf"Q-learning\;($\pi$) ($\epsilon={epsilon})$"

        # Q-learning variants
        if "q-learning" in name_lower:
            return rf"Q-learning ($\epsilon={epsilon}$)"

        # Fallback
        return name


    # Apply name shortening
    df["Algorithm"] = df["Algorithm"].apply(shorten_algorithm_name)

    # ---------- Generate LaTeX ----------
    latex_lines = []
    latex_lines.append(r"\begin{table}[!h]")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{lccccccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(
        f"Algorithm (Problem {ex_num}) & Runtime (mean $\pm$ std) & \#actions (mean $\pm$ std) & Convergence & Discover goal time (mean $\pm$ std) & Discover goal \#actions (mean $\pm$ std) & Optimal Initial Cost2Go Time (mean $\pm$ std) & Optimal Initial Cost2Go \#actions (mean $\pm$ std) \\\\"
    )
    latex_lines.append(r"\hline")

    for _, row in df.iterrows():
        line = (
            f"{row['Algorithm']} & "
            f"{row['Avg Time']:.5f} $\\pm$ {row['STD Time']:.4f} & "
            f"{row['Avg action count']:.4f} $\\pm$ {row['STD action count']:.4f} & "
            f"{100 * row['Convergence rate']:.1f} \% &"
            f"{row['Avg Time Goal Discovered']:.4f} $\\pm$ {row['STD Time Goal Discovered']:.4f} & "
            f"{row['Avg Actions Goal Discovered']:.4f} $\\pm$ {row['STD Actions Goal Discovered']:.4f} & "
            f"{row['Avg Time Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Time Optimal Initial Cost2Go']:.4f} & "
            f"{row['Avg Actions Optimal Initial Cost2Go']:.4f} $\\pm$ {row['STD Actions Optimal Initial Cost2Go']:.4f} \\\\"
        )
        latex_lines.append(line)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append("\end{table}")

    latex_code = "\n".join(latex_lines)
    print(latex_code)
