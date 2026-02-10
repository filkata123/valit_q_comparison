import pandas as pd
import re
import matplotlib.pyplot as plt

# === Problem numbers and file pattern ===
problem_numbers = [1, 10, 11]  # list of problems
file_pattern = "example_{}_stochastic_results_100_samples.csv"  # x will be replaced

# === Step 1: Initialize plot ===
plt.figure(figsize=(10, 7))

# === Step 2: Loop over problems ===
for problem_num in problem_numbers:
    csv_file = file_pattern.format(problem_num)
    df = pd.read_csv(csv_file)

    # === Step 3: Filter only (0.999 success) rows ===
    df_999 = df[df['Algorithm'].str.contains(r'\(0\.999 success\)')].copy()

    # === Step 4: Extract alpha as rho and epsilon ===
    def extract_alpha(name):
        match = re.search(r'alpha\s*=\s*([\d.]+)', name)
        return float(match.group(1)) if match else None

    def extract_epsilon(name):
        match = re.search(r'epsilon\s*=\s*([\d.]+)', name)
        return float(match.group(1)) if match else None

    df_999['rho'] = df_999['Algorithm'].apply(extract_alpha)
    df_999['epsilon'] = df_999['Algorithm'].apply(extract_epsilon)

    df_999 = df_999.dropna(subset=['epsilon'])

    # === Step 5: Normalize Avg Time per epsilon ===
    df_999['Avg Time Norm'] = 0.0
    for eps in df_999['epsilon'].unique():
        mask = df_999['epsilon'] == eps
        min_time = df_999.loc[mask, 'Avg Time'].min()
        max_time = df_999.loc[mask, 'Avg Time'].max()
        df_999.loc[mask, 'Avg Time Norm'] = (
            (df_999.loc[mask, 'Avg Time'] - min_time) / (max_time - min_time)
        )

    # === Step 6: Plot normalized Avg Time vs rho for each epsilon ===
    for eps in sorted(df_999['epsilon'].unique()):
        subset = df_999[df_999['epsilon'] == eps]
        plt.plot(subset['rho'], subset['Avg Time Norm'], marker='o', linestyle='-', label=f'Problem {problem_num}, epsilon={eps}')

# === Step 7: Final plot formatting ===
plt.xlabel('$\\rho$', fontsize=20)
plt.ylabel('Normalized Runtime', fontsize=20)
plt.title('Normalized Runtime vs \n $\\rho$ for different $\epsilon$ across problems 1, 10 and 11 ($\gamma = 0.999$)', fontsize=24)
rho = sorted(df_999["rho"].unique())
plt.xticks(rho, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=8)
plt.grid(True)
plt.show()
