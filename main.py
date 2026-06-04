"""
main.py
-------
Orchestrates the full parking assignment pipeline:

  1. Load real data → extract distributions
  2. Generate synthetic parking instance
  3. Solve with Greedy and ILP
  4. Evaluate and compare

Run:
  python main.py
  python main.py --floors 4 --spaces 25 --vehicles 80 --seed 7
"""

import argparse
import sys
import os

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from data_loader   import load, extract_distributions, summarise
from instance_gen  import create_instance
from solver        import GreedySolver, ILPSolver
from evaluator     import print_report, compare


# ── default configuration ─────────────────────────────────────────────────────

DEFAULT_CSV        = "data.csv"
DEFAULT_FLOORS     = 3
DEFAULT_SPACES     = 20
DEFAULT_VEHICLES   = 60
DEFAULT_SEED       = 42
ILP_TIME_LIMIT     = 60   # seconds — increase for larger instances


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    csv_path:   str = DEFAULT_CSV,
    n_floors:   int = DEFAULT_FLOORS,
    spaces_per_floor: int = DEFAULT_SPACES,
    n_vehicles: int = DEFAULT_VEHICLES,
    seed:       int = DEFAULT_SEED,
    skip_ilp:   bool = False,
):
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Multi-Level Parking Assignment Pipeline   ║")
    print("╚══════════════════════════════════════════════╝")

    # ── Step 1: Load data ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading data from '{csv_path}' ...")
    try:
        df   = load(csv_path)
        dist = extract_distributions(df)
        print(f"      OK — {len(df)} records. "
              f"EV fraction: {dist['ev_fraction']*100:.0f}%, "
              f"Mean inter-arrival: {dist['mean_interarrival_min']:.1f} min")
    except FileNotFoundError:
        print(f"      WARNING: '{csv_path}' not found. Using built-in defaults.")
        dist = None

    # ── Step 2: Generate instance ──────────────────────────────────────────
    print(f"\n[2/4] Generating synthetic parking instance (seed={seed}) ...")
    inst = create_instance(
        n_floors=n_floors,
        spaces_per_floor=spaces_per_floor,
        n_vehicles=n_vehicles,
        distributions=dist,
        seed=seed,
    )
    print(f"      {inst.n_spaces} spaces across {n_floors} floors | "
          f"{inst.n_vehicles} vehicles arriving")
    ev_count = inst.vehicles["needs_charger"].sum()
    charger_count = inst.spaces["has_charger"].sum()
    print(f"      EV vehicles: {ev_count} | EV-capable spaces: {charger_count}")

    # ── Step 3: Solve ──────────────────────────────────────────────────────
    print("\n[3/4] Running solvers ...")

    print("      → Greedy solver ...", end=" ", flush=True)
    greedy_result = GreedySolver().solve(inst)
    print(f"done ({greedy_result.runtime:.3f}s)  cost={greedy_result.cost:.1f}")

    assignments = [greedy_result]

    if not skip_ilp:
        if n_vehicles > 80:
            print(f"      → ILP solver ... (large instance: {n_vehicles} vehicles, "
                  f"time limit={ILP_TIME_LIMIT}s)")
        else:
            print("      → ILP solver ...", end=" ", flush=True)

        ilp_result = ILPSolver(time_limit_sec=ILP_TIME_LIMIT).solve(inst)
        print(f"done ({ilp_result.runtime:.3f}s)  cost={ilp_result.cost:.1f}")
        assignments.append(ilp_result)

    # ── Step 4: Evaluate ───────────────────────────────────────────────────
    print("\n[4/4] Evaluation results:")
    print_report(assignments, inst)

    # Return for programmatic use (e.g. notebooks)
    return inst, assignments


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parking assignment pipeline")
    parser.add_argument("--csv",      default=DEFAULT_CSV,    help="Path to data CSV")
    parser.add_argument("--floors",   type=int, default=DEFAULT_FLOORS,   help="Number of parking floors")
    parser.add_argument("--spaces",   type=int, default=DEFAULT_SPACES,   help="Spaces per floor")
    parser.add_argument("--vehicles", type=int, default=DEFAULT_VEHICLES, help="Number of vehicles")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED,     help="Random seed")
    parser.add_argument("--no-ilp",   action="store_true",                help="Skip ILP solver (faster)")
    args = parser.parse_args()

    run(
        csv_path=args.csv,
        n_floors=args.floors,
        spaces_per_floor=args.spaces,
        n_vehicles=args.vehicles,
        seed=args.seed,
        skip_ilp=args.no_ilp,
    )


if __name__ == "__main__":
    main()
