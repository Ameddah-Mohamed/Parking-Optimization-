import argparse
import sys
import os

from src.experiments.benchmark import run_benchmarks
from src.data.generator import generate_parking_instance
from src.algorithms.exact_milp import ExactMILPSolver
from src.algorithms.meta_sa import SimulatedAnnealing
from src.evaluation.cost_func import evaluate_assignment, get_default_weights

def main():
    parser = argparse.ArgumentParser(description="Smart Multi-Level Parking Optimization")
    parser.add_argument("--benchmark", action="store_true", help="Run the benchmark suite")
    parser.add_argument("--vehicles", type=int, default=50, help="Number of vehicles")
    parser.add_argument("--spots", type=int, default=30, help="Spaces per floor")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmarks()
    else:
        print(f"Running single test: {args.vehicles} vehicles, {args.spots} spots per floor")
        inst = generate_parking_instance(num_vehicles=args.vehicles, num_spots_per_level=args.spots)
        
        sa = SimulatedAnnealing().solve(inst)
        res = evaluate_assignment(sa['mapping'], inst, get_default_weights())
        
        print("\n[Simulated Annealing Result]")
        print(f"Total Cost: {res['total_cost']}")
        print(f"Avg Distance: {res['avg_distance']:.1f} m")
        print(f"Violations: {res['size_violations']} Size, {res['ev_violations']} EV")
        print(f"Runtime: {sa['runtime']:.3f} s")
        
        print("\n[Brief Assignment Mapping (First 10 Vehicles)]")
        mapping = sa['mapping']
        for i, vid in enumerate(list(mapping.keys())[:10]):
            v = next((v for v in inst.vehicles if v.id == vid), None)
            s_id = mapping[vid]
            print(f"  {vid} (Type: {v.v_type.upper()}, Arrival: {v.arrival_time}m)  --->  Assigned to Spot: {s_id}")

if __name__ == "__main__":
    main()
