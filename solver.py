"""
solver.py
---------
Two assignment strategies:

  1. GreedySolver   — O(V × S) online algorithm. Fast, no look-ahead.
                      Assigns each arriving vehicle to the nearest compatible
                      available space. Serves as the baseline.

  2. GeneticSolver  — population-based metaheuristic for offline assignment.

  3. SimulatedAnnealingSolver — local-search metaheuristic for offline assignment.

  4. ILPSolver      — Time-indexed binary integer program.
                      Optimal for a known arrival/departure stream on
                      small/medium instances.

Both return an Assignment object.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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


def _unassigned_penalty(weights: dict) -> float:
    return weights.get("penalty", 50.0) * 10


def _vehicle_intervals(vehicles: pd.DataFrame) -> dict:
    return {
        row["vehicle_id"]: (
            float(row["arrival_time"]),
            float(row["arrival_time"] + row["duration_min"]),
        )
        for _, row in vehicles.iterrows()
    }


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


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
                total += _unassigned_penalty(self.weights)
                continue
            idx = space_idx_map[sid]
            s = spaces.loc[idx]
            ev_pen  = self.weights["penalty"] if (v["needs_charger"] and not s["has_charger"]) else 0
            total  += self.weights["distance"] * exit_dist[idx] + ev_pen
        return round(total, 2)

    def _check_violations(self, mapping, vehicles, spaces):
        violations = []
        space_idx_map = {row["space_id"]: i for i, row in spaces.iterrows()}
        intervals = _vehicle_intervals(vehicles)
        by_space = {}

        for _, v in vehicles.iterrows():
            sid = mapping.get(v["vehicle_id"])
            if sid is None:
                violations.append(f"UNASSIGNED: {v['vehicle_id']} has no space")
                continue
            idx = space_idx_map[sid]
            s = spaces.loc[idx]
            if not is_size_compatible(v["size_needed"], s["size"]):
                violations.append(f"SIZE: {v['vehicle_id']} ({v['size_needed']}) → {sid} ({s['size']})")
            if bool(v["needs_charger"]) and not bool(s["has_charger"]):
                violations.append(f"EV: {v['vehicle_id']} requires charger but got {sid}")
            by_space.setdefault(sid, []).append(v["vehicle_id"])

        for sid, vids in by_space.items():
            for i, vid_a in enumerate(vids):
                for vid_b in vids[i + 1:]:
                    if _intervals_overlap(intervals[vid_a], intervals[vid_b]):
                        violations.append(f"TIME: {sid} assigned to overlapping vehicles {vid_a}, {vid_b}")

        return violations


# ── metaheuristic helpers ─────────────────────────────────────────────────────

class _MetaheuristicBase:
    """Shared repair/evaluation logic for offline stochastic solvers."""

    def __init__(self, weights: dict = None, seed: int = 42):
        self.weights = weights or {"distance": 1.0, "congestion": 0.5, "penalty": 50.0}
        self.seed = seed

    def _prepare(self, instance):
        spaces = instance.spaces.reset_index(drop=True)
        vehicles = instance.vehicles.reset_index(drop=True)
        exit_dist = instance.exit_dist
        feasible = []

        for _, v in vehicles.iterrows():
            candidates = []
            for s_idx, s in spaces.iterrows():
                if not is_size_compatible(v["size_needed"], s["size"]):
                    continue
                if bool(v["needs_charger"]) and not bool(s["has_charger"]):
                    continue
                candidates.append(int(s_idx))
            feasible.append(candidates)

        return spaces, vehicles, exit_dist, feasible

    def _random_chromosome(self, feasible, rng):
        chrom = []
        for candidates in feasible:
            choices = candidates + [-1]
            chrom.append(int(rng.choice(choices)))
        return np.array(chrom, dtype=int)

    def _greedy_chromosome(self, instance, spaces, vehicles):
        greedy = GreedySolver(weights=self.weights).solve(instance)
        space_idx_map = {row["space_id"]: i for i, row in spaces.iterrows()}
        return np.array([
            space_idx_map.get(greedy.mapping.get(v["vehicle_id"]), -1)
            for _, v in vehicles.iterrows()
        ], dtype=int)

    def _repair_and_score(self, chromosome, spaces, vehicles, exit_dist, feasible):
        available = set(range(len(spaces)))
        occupied_until = {}
        repaired = np.full(len(vehicles), -1, dtype=int)
        total = 0.0

        for v_idx, v in vehicles.iterrows():
            arr = float(v["arrival_time"])
            dep = float(v["arrival_time"] + v["duration_min"])

            newly_freed = [
                s_idx for s_idx, dep_t in list(occupied_until.items())
                if dep_t <= arr
            ]
            for s_idx in newly_freed:
                available.add(s_idx)
                del occupied_until[s_idx]

            candidates = [s_idx for s_idx in feasible[v_idx] if s_idx in available]
            preferred = int(chromosome[v_idx])

            if preferred in candidates:
                chosen = preferred
            elif candidates:
                chosen = min(candidates, key=lambda s_idx: exit_dist[s_idx])
            else:
                chosen = -1

            repaired[v_idx] = chosen
            if chosen == -1:
                total += _unassigned_penalty(self.weights)
                continue

            available.discard(chosen)
            occupied_until[chosen] = dep
            total += self.weights.get("distance", 1.0) * exit_dist[chosen]

        return repaired, round(float(total), 2)

    def _mapping_from_chromosome(self, chromosome, spaces, vehicles):
        mapping = {}
        for v_idx, v in vehicles.iterrows():
            s_idx = int(chromosome[v_idx])
            mapping[v["vehicle_id"]] = None if s_idx == -1 else spaces.loc[s_idx, "space_id"]
        return mapping


class GeneticSolver(_MetaheuristicBase):
    """
    Offline genetic algorithm.

    Chromosome: one preferred space index per vehicle. A repair step converts it
    into a feasible dynamic assignment by respecting availability, size, and EV
    charger constraints.
    """

    def __init__(
        self,
        weights: dict = None,
        seed: int = 42,
        population_size: int = 40,
        generations: int = 80,
        mutation_rate: float = 0.08,
        elite_count: int = 4,
    ):
        super().__init__(weights=weights, seed=seed)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count

    def solve(self, instance) -> Assignment:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        spaces, vehicles, exit_dist, feasible = self._prepare(instance)

        population = [self._greedy_chromosome(instance, spaces, vehicles)]
        population.extend(
            self._random_chromosome(feasible, rng)
            for _ in range(self.population_size - 1)
        )

        best_chrom = None
        best_cost = float("inf")

        def score(chrom):
            repaired, cost = self._repair_and_score(chrom, spaces, vehicles, exit_dist, feasible)
            return repaired, cost

        def tournament(scored, k=3):
            contenders = rng.choice(len(scored), size=k, replace=False)
            return min((scored[i] for i in contenders), key=lambda item: item[1])[0]

        for _ in range(self.generations):
            scored = [score(chrom) for chrom in population]
            scored.sort(key=lambda item: item[1])

            if scored[0][1] < best_cost:
                best_chrom = scored[0][0].copy()
                best_cost = scored[0][1]

            next_pop = [chrom.copy() for chrom, _ in scored[:self.elite_count]]
            while len(next_pop) < self.population_size:
                parent_a = tournament(scored)
                parent_b = tournament(scored)
                mask = rng.random(len(parent_a)) < 0.5
                child = np.where(mask, parent_a, parent_b)

                for gene_idx in range(len(child)):
                    if rng.random() < self.mutation_rate:
                        choices = feasible[gene_idx] + [-1]
                        child[gene_idx] = int(rng.choice(choices))

                next_pop.append(child)

            population = next_pop

        mapping = self._mapping_from_chromosome(best_chrom, spaces, vehicles)
        violations = GreedySolver(weights=self.weights)._check_violations(mapping, vehicles, spaces)
        return Assignment(
            mapping=mapping,
            cost=best_cost,
            solver="Genetic",
            runtime=time.perf_counter() - t0,
            violations=violations,
        )


class SimulatedAnnealingSolver(_MetaheuristicBase):
    """
    Offline simulated annealing heuristic.

    Starts from the greedy solution, repeatedly mutates one vehicle assignment,
    repairs the schedule, and accepts worse moves with a temperature-controlled
    probability to escape local minima.
    """

    def __init__(
        self,
        weights: dict = None,
        seed: int = 42,
        iterations: int = 1500,
        initial_temp: float = 200.0,
        cooling_rate: float = 0.995,
    ):
        super().__init__(weights=weights, seed=seed)
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def solve(self, instance) -> Assignment:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        spaces, vehicles, exit_dist, feasible = self._prepare(instance)

        current = self._greedy_chromosome(instance, spaces, vehicles)
        current, current_cost = self._repair_and_score(current, spaces, vehicles, exit_dist, feasible)
        best = current.copy()
        best_cost = current_cost
        temp = self.initial_temp

        for _ in range(self.iterations):
            candidate = current.copy()
            gene_idx = int(rng.integers(0, len(candidate)))
            choices = feasible[gene_idx] + [-1]
            candidate[gene_idx] = int(rng.choice(choices))

            if len(candidate) > 1 and rng.random() < 0.20:
                other_idx = int(rng.integers(0, len(candidate)))
                candidate[gene_idx], candidate[other_idx] = candidate[other_idx], candidate[gene_idx]

            candidate, candidate_cost = self._repair_and_score(
                candidate, spaces, vehicles, exit_dist, feasible
            )
            delta = candidate_cost - current_cost

            if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-9)):
                current = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

            temp *= self.cooling_rate

        mapping = self._mapping_from_chromosome(best, spaces, vehicles)
        violations = GreedySolver(weights=self.weights)._check_violations(mapping, vehicles, spaces)
        return Assignment(
            mapping=mapping,
            cost=best_cost,
            solver="Simulated Annealing",
            runtime=time.perf_counter() - t0,
            violations=violations,
        )


# ── ILP solver ────────────────────────────────────────────────────────────────

class ILPSolver:
    """
    Binary Integer Linear Program.

    Decision variables:
      x[v, s] ∈ {0, 1}  =  1 if vehicle v is assigned to space s
      y[v]    ∈ {0, 1}  =  1 if vehicle v cannot be assigned

    Objective (minimise):
      Σ_{v,s} x[v,s] * cost(v, s) + big_penalty * Σ_v y[v]

    Constraints:
      (1) Each vehicle is assigned to one real space or the unassigned dummy.
      (2) Two vehicles with overlapping parking intervals cannot use the same
          physical space.
      (3) Size compatibility is hard: x[v,s] = 0 if space too small.
      (4) Required EV charging is hard: x[v,s] = 0 if an EV needs charging and
          the space has no charger.

    Note: this is an offline solver. It sees all arrivals/departures in advance,
    so it is a benchmark lower bound rather than a deployable online policy.
    For large instances (>60 vehicles) use time_limit_sec to cap runtime.
    """

    def __init__(self, weights: dict = None, time_limit_sec: int = 60, unassigned_penalty: float = None):
        self.weights = weights or {"distance": 1.0, "congestion": 0.5, "penalty": 50.0}
        self.time_limit_sec = time_limit_sec
        self.unassigned_penalty = (
            unassigned_penalty
            if unassigned_penalty is not None
            else _unassigned_penalty(self.weights)
        )

    def solve(self, instance) -> Assignment:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_array
        except ImportError:
            raise ImportError("SciPy >= 1.11 is required for ILPSolver.")

        t0 = time.perf_counter()
        spaces   = instance.spaces
        vehicles = instance.vehicles
        exit_dist = instance.exit_dist
        w = self.weights

        V = list(vehicles["vehicle_id"])
        S = list(spaces["space_id"])
        n_v = len(V)
        n_s = len(S)
        space_idx = {row["space_id"]: i for i, row in spaces.iterrows()}
        veh_map   = {row["vehicle_id"]: row for _, row in vehicles.iterrows()}
        spc_map   = {row["space_id"]:   row for _, row in spaces.iterrows()}

        def x_idx(v_i: int, s_i: int) -> int:
            return v_i * n_s + s_i

        def y_idx(v_i: int) -> int:
            return n_v * n_s + v_i

        n_vars = n_v * n_s + n_v
        c = np.zeros(n_vars, dtype=float)
        lower = np.zeros(n_vars, dtype=float)
        upper = np.ones(n_vars, dtype=float)
        integrality = np.ones(n_vars, dtype=int)

        # Pre-compute objective and hard feasibility bounds.
        for v_i, vid in enumerate(V):
            v = veh_map[vid]
            for s_i, sid in enumerate(S):
                s = spc_map[sid]
                idx = space_idx[sid]
                var = x_idx(v_i, s_i)
                c[var] = w["distance"] * exit_dist[idx]

                if not is_size_compatible(v["size_needed"], s["size"]):
                    upper[var] = 0
                if bool(v["needs_charger"]) and not bool(s["has_charger"]):
                    upper[var] = 0

            c[y_idx(v_i)] = self.unassigned_penalty

        rows = []
        cols = []
        vals = []
        lb = []
        ub = []
        row = 0

        # Each vehicle is assigned once, either to a real space or dummy y[v].
        for v_i in range(n_v):
            for s_i in range(n_s):
                rows.append(row)
                cols.append(x_idx(v_i, s_i))
                vals.append(1.0)
            rows.append(row)
            cols.append(y_idx(v_i))
            vals.append(1.0)
            lb.append(1.0)
            ub.append(1.0)
            row += 1

        # Same space cannot be assigned to two vehicles whose time intervals overlap.
        arrivals = vehicles["arrival_time"].to_numpy(dtype=float)
        departures = arrivals + vehicles["duration_min"].to_numpy(dtype=float)
        overlap_pairs = [
            (i, j)
            for i in range(n_v)
            for j in range(i + 1, n_v)
            if max(arrivals[i], arrivals[j]) < min(departures[i], departures[j])
        ]
        for s_i in range(n_s):
            for i, j in overlap_pairs:
                rows.extend([row, row])
                cols.extend([x_idx(i, s_i), x_idx(j, s_i)])
                vals.extend([1.0, 1.0])
                lb.append(0.0)
                ub.append(1.0)
                row += 1

        constraints = LinearConstraint(
            coo_array((vals, (rows, cols)), shape=(row, n_vars)).tocsr(),
            np.array(lb),
            np.array(ub),
        )

        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"time_limit": self.time_limit_sec, "disp": False},
        )

        if not result.success and result.x is None:
            return Assignment(
                mapping={vid: None for vid in V},
                cost=float("inf"),
                solver="ILP",
                runtime=time.perf_counter() - t0,
                violations=[f"OPTIMIZER: {result.message}"],
            )

        sol = result.x
        mapping = {}
        for v_i, vid in enumerate(V):
            if sol[y_idx(v_i)] > 0.5:
                mapping[vid] = None
                continue
            for s_i, sid in enumerate(S):
                if sol[x_idx(v_i, s_i)] > 0.5:
                    mapping[vid] = sid
                    break
            if vid not in mapping:
                mapping[vid] = None

        total_cost = round(float(c @ np.rint(sol)), 2)
        violations = GreedySolver()._check_violations(mapping, vehicles, spaces)
        if not result.success:
            violations.append(f"OPTIMIZER: {result.message}")

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
