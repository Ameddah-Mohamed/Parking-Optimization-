"""
solver.py
---------
Two assignment strategies:

  1. GreedySolver   — O(V × S) online algorithm. Fast, no look-ahead.
                      Assigns each arriving vehicle to the nearest compatible
                      available space. Serves as the baseline.

  2. ILPSolver      — Batch Integer Linear Program (PuLP + CBC).
                      Optimal for a fixed snapshot of vehicles and spaces.
                      Used for comparison on small/medium instances.

Both return an Assignment object.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import time


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class Assignment:
    """
    Stores the result of one solver run.

    mapping : dict  {vehicle_id: space_id}  — the actual assignment
    cost    : float — total objective value
    solver  : str   — name of the solver used
    runtime : float — wall-clock seconds
    violations: list[str] — any constraint violations found (should be empty)
    """
    mapping:    dict
    cost:       float
    solver:     str
    runtime:    float
    violations: list = field(default_factory=list)

    def to_dataframe(self, vehicles: pd.DataFrame, spaces: pd.DataFrame) -> pd.DataFrame:
        """Merge assignment back onto vehicles for easy inspection."""
        rows = []
        for vid, sid in self.mapping.items():
            v = vehicles[vehicles["vehicle_id"] == vid].iloc[0]
            s = spaces[spaces["space_id"] == sid].iloc[0]
            rows.append({
                "vehicle_id":   vid,
                "vehicle_type": v["vehicle_type"],
                "size_needed":  v["size_needed"],
                "needs_charger":v["needs_charger"],
                "space_id":     sid,
                "floor":        s["floor"],
                "spot_in_floor":s["spot_in_floor"],
                "space_size":   s["size"],
                "has_charger":  s["has_charger"],
            })
        return pd.DataFrame(rows)


# ── compatibility helpers ─────────────────────────────────────────────────────

SIZE_ORDER = {"compact": 0, "suv": 1, "van": 2}

def is_size_compatible(vehicle_size: str, space_size: str) -> bool:
    """A vehicle fits in a space if the space is >= the vehicle's size."""
    return SIZE_ORDER.get(space_size, 0) >= SIZE_ORDER.get(vehicle_size, 0)

def compute_cost(
    vehicle: pd.Series,
    space: pd.Series,
    exit_dist: np.ndarray,
    space_idx: int,
    occupancy_on_floor: dict,
    weights: dict,
) -> float:
    """
    Cost of assigning one vehicle to one space.

    Components:
      w1 * exit_distance
      w2 * floor_congestion_penalty
      w3 * ev_penalty  (EV not assigned to charger)
      w3 * size_penalty (oversized space)
    """
    w1 = weights.get("distance", 1.0)
    w2 = weights.get("congestion", 0.5)
    w3 = weights.get("penalty", 50.0)

    dist_cost   = w1 * exit_dist[space_idx]
    floor_occ   = occupancy_on_floor.get(int(space["floor"]), 0)
    cong_cost   = w2 * floor_occ * 10   # 10m equivalent per occupied neighbour

    ev_penalty   = w3 if (vehicle["needs_charger"] and not space["has_charger"]) else 0.0
    size_penalty = w3 * 0.5 if SIZE_ORDER.get(space["size"], 0) > SIZE_ORDER.get(vehicle["size_needed"], 0) else 0.0

    return dist_cost + cong_cost + ev_penalty + size_penalty


# ── greedy solver ─────────────────────────────────────────────────────────────

class GreedySolver:
    """
    Online greedy assignment.

    For each vehicle (in arrival order):
      1. Filter spaces that are available and size-compatible.
      2. Among those, prefer EV chargers if the vehicle needs one.
      3. Pick the space with the lowest cost (primarily: exit distance).
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or {"distance": 1.0, "congestion": 0.5, "penalty": 50.0}

    def solve(self, instance) -> Assignment:
        from instance_gen import ParkingInstance  # avoid circular import
        t0 = time.perf_counter()

        spaces   = instance.spaces.copy()
        vehicles = instance.vehicles.copy()
        exit_dist = instance.exit_dist

        available = set(spaces.index)      # indices of free spaces
        mapping   = {}                     # vehicle_id → space_id
        occupied_until = {}                # space_idx → departure time
        floor_occupancy = {f: 0 for f in spaces["floor"].unique()}

        for _, v in vehicles.iterrows():
            arr = v["arrival_time"]
            dep = arr + v["duration_min"]

            # Free spaces whose occupant has already departed
            newly_freed = {
                idx for idx, dep_t in list(occupied_until.items())
                if dep_t <= arr
            }
            for idx in newly_freed:
                available.add(idx)
                floor_occupancy[int(spaces.loc[idx, "floor"])] -= 1
                del occupied_until[idx]

            # Filter to compatible spaces
            candidates = [
                idx for idx in available
                if is_size_compatible(v["size_needed"], spaces.loc[idx, "size"])
            ]

            # Prefer charger-equipped spaces for EVs
            if v["needs_charger"]:
                charger_cands = [i for i in candidates if spaces.loc[i, "has_charger"]]
                if charger_cands:
                    candidates = charger_cands

            if not candidates:
                # No compatible space available — record violation
                mapping[v["vehicle_id"]] = None
                continue

            # Pick minimum-cost candidate
            best_idx = min(
                candidates,
                key=lambda i: compute_cost(v, spaces.loc[i], exit_dist, i, floor_occupancy, self.weights)
            )

            mapping[v["vehicle_id"]] = spaces.loc[best_idx, "space_id"]
            available.discard(best_idx)
            occupied_until[best_idx] = dep
            floor_occupancy[int(spaces.loc[best_idx, "floor"])] += 1

        cost = self._total_cost(mapping, vehicles, spaces, exit_dist)
        violations = self._check_violations(mapping, vehicles, spaces)

        return Assignment(
            mapping=mapping,
            cost=cost,
            solver="Greedy",
            runtime=time.perf_counter() - t0,
            violations=violations,
        )

    def _total_cost(self, mapping, vehicles, spaces, exit_dist):
        total = 0.0
        space_idx_map = {row["space_id"]: i for i, row in spaces.iterrows()}
        for _, v in vehicles.iterrows():
            sid = mapping.get(v["vehicle_id"])
            if sid is None:
                total += self.weights.get("penalty", 50.0) * 10  # big penalty for unassigned
                continue
            idx = space_idx_map[sid]
            s = spaces.loc[idx]
            ev_pen  = self.weights["penalty"] if (v["needs_charger"] and not s["has_charger"]) else 0
            total  += self.weights["distance"] * exit_dist[idx] + ev_pen
        return round(total, 2)

    def _check_violations(self, mapping, vehicles, spaces):
        violations = []
        space_idx_map = {row["space_id"]: i for i, row in spaces.iterrows()}
        assigned_spaces = [v for v in mapping.values() if v is not None]

        # Duplicate assignment check
        if len(assigned_spaces) != len(set(assigned_spaces)):
            violations.append("DUPLICATE: same space assigned to multiple vehicles")

        for _, v in vehicles.iterrows():
            sid = mapping.get(v["vehicle_id"])
            if sid is None:
                violations.append(f"UNASSIGNED: {v['vehicle_id']} has no space")
                continue
            idx = space_idx_map[sid]
            s = spaces.loc[idx]
            if not is_size_compatible(v["size_needed"], s["size"]):
                violations.append(f"SIZE: {v['vehicle_id']} ({v['size_needed']}) → {sid} ({s['size']})")

        return violations


# ── ILP solver ────────────────────────────────────────────────────────────────

class ILPSolver:
    """
    Batch Integer Linear Program using PuLP.

    Decision variables:
      x[v, s] ∈ {0, 1}  =  1 if vehicle v is assigned to space s

    Objective (minimise):
      Σ_{v,s} x[v,s] * cost(v, s)

    Constraints:
      (1) Each vehicle assigned to exactly one space.
      (2) Each space used by at most one vehicle.
      (3) Size compatibility: x[v,s] = 0 if space too small.
      (4) EV soft constraint via high penalty in objective.

    Note: this is a static/batch solver — it sees all vehicles at once.
    For large instances (>60 vehicles) use time_limit_sec to cap runtime.
    """

    def __init__(self, weights: dict = None, time_limit_sec: int = 60):
        self.weights = weights or {"distance": 1.0, "congestion": 0.5, "penalty": 50.0}
        self.time_limit_sec = time_limit_sec

    def solve(self, instance) -> Assignment:
        try:
            import pulp
        except ImportError:
            raise ImportError("PuLP is required for ILPSolver. Run: pip install pulp")

        t0 = time.perf_counter()
        spaces   = instance.spaces
        vehicles = instance.vehicles
        exit_dist = instance.exit_dist
        w = self.weights

        V = list(vehicles["vehicle_id"])
        S = list(spaces["space_id"])
        space_idx = {row["space_id"]: i for i, row in spaces.iterrows()}
        veh_map   = {row["vehicle_id"]: row for _, row in vehicles.iterrows()}
        spc_map   = {row["space_id"]:   row for _, row in spaces.iterrows()}

        # Pre-compute cost matrix and feasibility mask
        cost = {}
        feasible = {}
        for vid in V:
            v = veh_map[vid]
            for sid in S:
                s = spc_map[sid]
                idx = space_idx[sid]
                if not is_size_compatible(v["size_needed"], s["size"]):
                    feasible[(vid, sid)] = False
                    cost[(vid, sid)] = 1e9
                else:
                    feasible[(vid, sid)] = True
                    ev_pen = w["penalty"] if (v["needs_charger"] and not s["has_charger"]) else 0
                    cost[(vid, sid)] = w["distance"] * exit_dist[idx] + ev_pen

        prob = pulp.LpProblem("ParkingAssignment", pulp.LpMinimize)

        x = pulp.LpVariable.dicts("x", [(v, s) for v in V for s in S], cat="Binary")

        # Force infeasible pairs to 0
        for vid in V:
            for sid in S:
                if not feasible[(vid, sid)]:
                    prob += x[(vid, sid)] == 0

        # Objective
        prob += pulp.lpSum(cost[(v, s)] * x[(v, s)] for v in V for s in S)

        # Constraint 1: each vehicle gets exactly one space
        for vid in V:
            prob += pulp.lpSum(x[(vid, s)] for s in S) == 1

        # Constraint 2: each space used at most once
        for sid in S:
            prob += pulp.lpSum(x[(v, sid)] for v in V) <= 1

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit_sec)
        prob.solve(solver)

        mapping = {}
        for vid in V:
            for sid in S:
                if pulp.value(x[(vid, sid)]) and pulp.value(x[(vid, sid)]) > 0.5:
                    mapping[vid] = sid
                    break
            if vid not in mapping:
                mapping[vid] = None

        total_cost = round(pulp.value(prob.objective) or 0.0, 2)
        violations = GreedySolver()._check_violations(mapping, vehicles, spaces)

        return Assignment(
            mapping=mapping,
            cost=total_cost,
            solver="ILP",
            runtime=time.perf_counter() - t0,
            violations=violations,
        )


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from instance_gen import create_instance

    inst = create_instance(n_floors=3, spaces_per_floor=20, n_vehicles=30, seed=42)
    print(inst.summary())

    print("\n--- Greedy ---")
    g = GreedySolver().solve(inst)
    print(f"  Cost: {g.cost}  |  Runtime: {g.runtime:.3f}s  |  Violations: {g.violations}")

    print("\n--- ILP ---")
    ilp = ILPSolver(time_limit_sec=30).solve(inst)
    print(f"  Cost: {ilp.cost}  |  Runtime: {ilp.runtime:.3f}s  |  Violations: {ilp.violations}")
