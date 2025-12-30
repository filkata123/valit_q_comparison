#ChatGPT generated

import pandas as pd

# --- CONFIG ---
metric = "Avg Time"  #"Avg Time", "Avg action count" or any of the other colum names
std_metric = "STD Time"  # "STD Time" "STD action count" Corresponding standard deviation column

# Mapping from LaTeX table columns to CSV rows (row indices for the 10 parameter settings)
# You can adjust these indices based on your CSV
table_indices = [
    # 0,  
    # 1,  
    # 2,  
    # 3,  
    # 4,  
    # 7,  
    # 8,  
    # 9,  
    # 12, 
    # 13
    9,
    16
]

# --- LOAD CSV ---
print(f"metric: {metric}")
for i in range(17):
    if i == 6:
        continue
    df = pd.read_csv(f'example_{i}_results_100_samples.csv')
    # --- BUILD LATEX ROW ---
    latex_cells = []
    for idx in table_indices:
        row = df.iloc[idx]
        if metric in ["Avg Time", "Avg action count"]:
            value = row[metric]
            std = row[std_metric]
            cell = f"${value:.5f} \\pm {std:.5f}$\n"
            latex_cells.append(cell)
        else:
            value = row[metric]
            shortest_path = row["Shortest Path"]
            longest_path = row["Longest Path"]
            cell = f"{value}({shortest_path}-{longest_path})\n"
            latex_cells.append(cell)

    latex_row = "\makecell[c]{Problem "+ str(i) +"}\n & " + " & ".join(latex_cells) + " \\\\ \n \hline"
    print(latex_row)

# # --- OPTIONAL: Avg action count ---
# metric = "Avg action count"
# std_metric = "STD action count"

# latex_cells_actions = []
# for idx in table_indices:
#     row = df.iloc[idx]
#     value = row[metric]
#     std = row[std_metric]
#     cell = f"${value:.2f} \\pm {std:.2f}$"
#     latex_cells_actions.append(cell)

# latex_row_actions = " & ".join(latex_cells_actions) + " \\\\"
# print(f"% LaTeX row (Avg actions)")
# print(latex_row_actions)
