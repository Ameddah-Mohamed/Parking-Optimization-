"""
evaluator.py
------------
Computes evaluation metrics for an Assignment and prints comparison reports.

Metrics:
  - avg_exit_distance   : mean walking distance (metres) per vehicle
  - ev_satisfaction     : fraction of EVs assigned to a charger
  - unassigned_rate     : fraction of vehicles with no space
  - total_cost          : raw objective value from the solver
  - size_violation_rate : fraction of assignments with size mismatch
  - floor_load          : dict {floor: peak simultaneous vehicles assigned}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Metrics:
    solver:               str
    total_cost:           float
    avg_exit_distance:    float   # metres
    ev_satisfaction:      float   # 0–1
    unassigned_rate:      float   # 0–1
    size_violation_rate:  float   # 0–1 (should be 0)
    floor_load:           dict
    runtime_sec:          float

    def __str__(self):
        ev_pct  = self.ev_satisfaction * 100
        un_pct  = self.unassigned_rate * 100
        sv_pct  = self.size_violation_rate * 100
        fl_str  = "  ".join(f"F{k}:{v}" for k, v in sorted(self.floor_load.items()))
        return (
            f"[{self.solver}]\n"
            f"  Total cost          : {self.total_cost:.1f}\n"
            f"  Avg exit distance   : {self.avg_exit_distance:.1f} m\n"
            f"  EV satisfaction     : {ev_pct:.1f}%\n"
            f"  Unassigned vehicles : {un_pct:.1f}%\n"
            f"  Size violations     : {sv_pct:.1f}%\n"
            f"  Peak floor occupancy: {fl_str}\n"
            f"  Runtime             : {self.runtime_sec:.3f}s"
        )


def evaluate(assignment, instance) -> Metrics:
    """
    Compute all metrics for one assignment against one instance.

    Parameters
    ----------
    assignment : Assignment  (from solver.py)
    instance   : ParkingInstance  (from instance_gen.py)
    """
    from solver import is_size_compatible   # local import to avoid circular

    spaces    = instance.spaces
    vehicles  = instance.vehicles
    exit_dist = instance.exit_dist

    space_idx_map = {row["space_id"]: i for i, row in spaces.iterrows()}
    spc_map       = {row["space_id"]: row for _, row in spaces.iterrows()}

    distances      = []
    ev_satisfied   = []
    unassigned     = 0
    size_violations = 0
    floor_events   = {int(f): [] for f in spaces["floor"].unique()}

    for _, v in vehicles.iterrows():
        sid = assignment.mapping.get(v["vehicle_id"])

        if sid is None:
            unassigned += 1
            continue

        idx = space_idx_map[sid]
        s   = spc_map[sid]

        distances.append(exit_dist[idx])

        if v["needs_charger"]:
            ev_satisfied.append(bool(s["has_charger"]))

        if not is_size_compatible(v["size_needed"], s["size"]):
            size_violations += 1

        floor = int(s["floor"])
        floor_events[floor].append((float(v["arrival_time"]), 1))
        floor_events[floor].append((float(v["arrival_time"] + v["duration_min"]), -1))

    n = len(vehicles)
    avg_dist    = float(np.mean(distances)) if distances else 0.0
    ev_rate     = float(np.mean(ev_satisfied)) if ev_satisfied else 1.0  # no EVs = perfect
    unassign_rt = unassigned / n if n > 0 else 0.0
    size_vio_rt = size_violations / n if n > 0 else 0.0

    peak_floor_load = {}
    for floor, events in floor_events.items():
        current = 0
        peak = 0
        # Departures before arrivals at the same timestamp, matching solver reuse.
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            current += delta
            peak = max(peak, current)
        peak_floor_load[floor] = peak

    return Metrics(
        solver=assignment.solver,
        total_cost=assignment.cost,
        avg_exit_distance=avg_dist,
        ev_satisfaction=ev_rate,
        unassigned_rate=unassign_rt,
        size_violation_rate=size_vio_rt,
        floor_load=peak_floor_load,
        runtime_sec=assignment.runtime,
    )


def compare(assignments: list, instance) -> pd.DataFrame:
    """
    Evaluate multiple assignments on the same instance and return a
    comparison DataFrame — one row per solver.
    """
    rows = []
    for a in assignments:
        m = evaluate(a, instance)
        rows.append({
            "Solver":             m.solver,
            "Total cost":         m.total_cost,
            "Avg distance (m)":   round(m.avg_exit_distance, 1),
            "EV satisfied (%)":   round(m.ev_satisfaction * 100, 1),
            "Unassigned (%)":     round(m.unassigned_rate * 100, 1),
            "Size violations (%)":round(m.size_violation_rate * 100, 1),
            "Runtime (s)":        round(m.runtime_sec, 3),
        })
    return pd.DataFrame(rows).set_index("Solver")


def print_report(assignments: list, instance) -> None:
    """Pretty-print a full comparison report to stdout."""
    print("\n" + "=" * 56)
    print("  PARKING ASSIGNMENT — EVALUATION REPORT")
    print("=" * 56)
    print(f"  Instance: {instance.n_floors} floors × {instance.spaces_per_floor} spaces "
          f"| {instance.n_vehicles} vehicles | seed={instance.seed}")

    ev_count = instance.vehicles["needs_charger"].sum()
    print(f"  EV vehicles: {ev_count}  |  "
          f"EV charger spaces: {instance.spaces['has_charger'].sum()}")
    print("=" * 56)

    for a in assignments:
        m = evaluate(a, instance)
        print(f"\n{m}")

    print("\n--- Side-by-side ---")
    df = compare(assignments, instance)
    print(df.to_string())
    print()


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from instance_gen import create_instance
    from solver import GreedySolver, ILPSolver

    inst = create_instance(n_floors=3, spaces_per_floor=20, n_vehicles=30, seed=42)
    g   = GreedySolver().solve(inst)
    ilp = ILPSolver(time_limit_sec=30).solve(inst)
    print_report([g, ilp], inst)
