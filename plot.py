import pandas as pd
import matplotlib.pyplot as plt

# Sample data (you can replace this with your actual DataFrame)

df = pd.read_csv("csv_files/hp_tuning(Sheet1).csv")

df = pd.read_csv("csv_files/hp_tuning(pdm_iteration).csv")  

x_col = "pdm_iteration"

phantom_choice ='resolution'

# Helper function for string-type x-axis sorting
def sort_if_string(df_subset, x_col):
    if df_subset[x_col].dtype == 'object':
        unique_order = sorted(df_subset[x_col].unique())
        df_subset[x_col] = pd.Categorical(df_subset[x_col], categories=unique_order, ordered=True)
    return df_subset.sort_values(x_col)

# Plot for one phantom type
def plot_single(df, phantom_type, x_col):
    df_sub = df[df["phantom_type"] == phantom_type].copy()
    df_sub = sort_if_string(df_sub, x_col)

    plt.figure(figsize=(8, 5))
    plt.plot(df_sub[x_col], df_sub["rmse"], marker='o', linewidth=2)
    plt.title(f"RMSE vs Number of PDM iterations ({phantom_type} phantom)", fontsize=14)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot for all phantom types
def plot_all(df, x_col):
    plt.figure(figsize=(10, 6))
    for phantom in df["phantom_type"].unique():
        df_sub = df[df["phantom_type"] == phantom].copy()
        df_sub = sort_if_string(df_sub, x_col)
        plt.plot(df_sub[x_col], df_sub["rmse"], marker='o', label=phantom)

    plt.title(f"RMSE vs Number of PDM iterations  (All Phantoms)", fontsize=14)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.legend(title="Phantom Type")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.show()
    

# Execute based on choice
if phantom_choice == "all":
    plot_all(df, x_col)
else:
    if phantom_choice not in df["phantom_type"].unique():
        print(f"Phantom type '{phantom_choice}' not found in the data.")
    else:
        plot_single(df, phantom_choice, x_col)
