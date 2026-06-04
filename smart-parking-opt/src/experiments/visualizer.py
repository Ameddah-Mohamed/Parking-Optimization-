import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_benchmark_results(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Plots the benchmark comparison (Cost and Time) using Seaborn.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for instance in df['Instance'].unique():
        sub_df = df[df['Instance'] == instance]
        
        # 1. Cost Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Algorithm", y="Cost", data=sub_df, palette="viridis")
        plt.title(f"Total Assignment Cost by Algorithm\nInstance: {instance}")
        plt.ylabel("Total Cost (Lower is better)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{instance}_cost_comparison.png"))
        plt.close()

        # 2. Runtime Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Algorithm", y="Time (s)", data=sub_df, palette="magma")
        plt.title(f"Execution Time by Algorithm\nInstance: {instance}")
        plt.ylabel("Time in seconds (Log scale, lower is better)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{instance}_time_comparison.png"))
        plt.close()

def plot_convergence(ga_history: list, sa_history: list, instance_name: str, output_dir: str = "plots"):
    """
    Plots the convergence history of GA and SA.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))
    
    # We might have different lengths. Plot against iteration/generation percentage
    ga_pct = [i/len(ga_history) * 100 for i in range(len(ga_history))]
    sa_pct = [i/len(sa_history) * 100 for i in range(len(sa_history))]
    
    plt.plot(ga_pct, ga_history, label='Genetic Algorithm (GA)', color='blue', linewidth=2)
    plt.plot(sa_pct, sa_history, label='Simulated Annealing (SA)', color='red', linewidth=2)
    
    plt.title(f"Metaheuristic Convergence Comparison\nInstance: {instance_name}")
    plt.xlabel("Progress (%)")
    plt.ylabel("Best Cost Found")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{instance_name}_convergence.png"))
    plt.close()
