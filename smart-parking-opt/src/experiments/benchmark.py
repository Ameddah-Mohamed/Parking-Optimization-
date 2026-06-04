import time
import pandas as pd
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.data.generator import generate_parking_instance
from src.evaluation.cost_func import get_default_weights
from src.algorithms.exact_milp import ExactMILPSolver
from src.algorithms.exact_bb import ExactBranchAndBound
from src.algorithms.meta_ga import GeneticAlgorithm
from src.algorithms.meta_sa import SimulatedAnnealing

def run_benchmarks():
    print("==========================================================")
    print(" SMART MULTI-LEVEL PARKING OPTIMIZATION - BENCHMARK RUN")
    print("==========================================================")
    
    # 1. Define Benchmark Instances
    instance_configs = [
        {"name": "Small_10V_20S", "v": 10, "s": 20, "l": 2},
        {"name": "Medium_30V_50S", "v": 30, "s": 50, "l": 3},
        # To run faster, skip the large one by default
        # {"name": "Large_100V_100S", "v": 100, "s": 100, "l": 4} 
    ]
    
    results = []
    
    for conf in instance_configs:
        print(f"\n[+] Generating Instance: {conf['name']}")
        inst = generate_parking_instance(
            name=conf['name'], 
            num_vehicles=conf['v'], 
            num_spots_per_level=conf['s'], 
            num_levels=conf['l']
        )
        
        # Exact MILP
        print(f"    -> Running MILP...")
        try:
            milp = ExactMILPSolver(time_limit=60).solve(inst)
            milp_cost = milp.get('best_cost', 'N/A') # We need evaluation to get true cost
            # Re-evaluate with exact dynamic cost
            from src.evaluation.cost_func import evaluate_assignment
            milp_eval = evaluate_assignment(milp['mapping'], inst, get_default_weights())
            milp_cost = milp_eval['total_cost']
            results.append({
                "Instance": inst.name, "Algorithm": "MILP", 
                "Cost": milp_cost, "Time (s)": milp['runtime']
            })
        except Exception as e:
            print(f"    -> MILP failed: {e}")

        # Exact B&B
        print(f"    -> Running Branch & Bound...")
        bb = ExactBranchAndBound(time_limit=10).solve(inst)
        bb_eval = evaluate_assignment(bb['mapping'], inst, get_default_weights())
        results.append({
            "Instance": inst.name, "Algorithm": "Branch & Bound", 
            "Cost": bb_eval['total_cost'], "Time (s)": bb['runtime']
        })

        # Genetic Algorithm
        print(f"    -> Running Genetic Algorithm...")
        ga = GeneticAlgorithm(pop_size=20, generations=50).solve(inst)
        results.append({
            "Instance": inst.name, "Algorithm": "GA", 
            "Cost": ga['best_cost'], "Time (s)": ga['runtime']
        })

        # Simulated Annealing
        print(f"    -> Running Simulated Annealing...")
        sa = SimulatedAnnealing(initial_temp=1000, alpha=0.9, iterations_per_temp=20).solve(inst)
        results.append({
            "Instance": inst.name, "Algorithm": "SA", 
            "Cost": sa['best_cost'], "Time (s)": sa['runtime']
        })
        
        # Plot convergence for this instance
        from src.experiments.visualizer import plot_convergence
        plot_convergence(ga['history'], sa['history'], inst.name)
        
    print("\n==========================================================")
    print(" BENCHMARK RESULTS")
    print("==========================================================")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save benchmark charts
    from src.experiments.visualizer import plot_benchmark_results
    plot_benchmark_results(df)
    print("\n[+] Academic plots saved to the 'plots/' directory!")

if __name__ == "__main__":
    run_benchmarks()
