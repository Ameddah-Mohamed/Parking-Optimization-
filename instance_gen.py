"""
instance_gen.py
---------------
Generates reproducible synthetic parking instances.

A ParkingInstance contains:
  - spaces     : DataFrame  (one row per parking space)
  - vehicles   : DataFrame  (one row per vehicle, sorted by arrival_time)
  - dist_matrix: 2-D ndarray  dist_matrix[i, j] = walking metres from space i to space j
  - exit_dist  : 1-D ndarray  exit_dist[i] = walking metres from space i to the exit

All randomness is controlled by `seed`, so results are fully reproducible.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── constants ─────────────────────────────────────────────────────────────────

# Walking distances (metres) from the exit ramp per floor
FLOOR_BASE_DIST = {1: 50, 2: 130, 3: 210, 4: 290, 5: 370}

# Walking distance added per spot position within a floor (spot 1 is nearest ramp)
DIST_PER_SPOT = 3.5   # metres

# Ramp/elevator capacity: max vehicles in transit between floors simultaneously
RAMP_CAPACITY = 4

# Probability that a space has an EV charger, depending on position
EV_CHARGER_PROB_NEAR  = 0.40   # first 5 spots on each floor
EV_CHARGER_PROB_OTHER = 0.10

# Space size distribution per floor (compact, suv, van)
SIZE_PROBS = [0.55, 0.35, 0.10]
SIZE_LABELS = ["compact", "suv", "van"]


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ParkingInstance:
    """Complete benchmark instance passed to the solver."""
    n_floors:        int
    spaces_per_floor: int
    spaces:          pd.DataFrame          # schema described in generate_spaces()
    vehicles:        pd.DataFrame          # schema described in generate_vehicles()
    dist_matrix:     np.ndarray            # shape (n_spaces, n_spaces)
    exit_dist:       np.ndarray            # shape (n_spaces,)
    ramp_capacity:   int = RAMP_CAPACITY
    seed:            int = 42

    @property
    def n_spaces(self) -> int:
        return len(self.spaces)

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    def summary(self) -> str:
        s = self.spaces
        ev_spaces = s["has_charger"].sum()
        size_counts = s["size"].value_counts().to_dict()
        return (
            f"ParkingInstance(seed={self.seed})\n"
            f"  Floors: {self.n_floors}  |  Spaces/floor: {self.spaces_per_floor}"
            f"  |  Total spaces: {self.n_spaces}\n"
            f"  EV chargers: {ev_spaces}  |  Sizes: {size_counts}\n"
            f"  Vehicles to assign: {self.n_vehicles}\n"
            f"  Exit dist range: {self.exit_dist.min():.0f}m – {self.exit_dist.max():.0f}m"
        )


# ── space generation ──────────────────────────────────────────────────────────

def generate_spaces(
    n_floors: int = 3,
    spaces_per_floor: int = 20,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Build the spaces table.

    Columns:
      space_id       str     e.g. "F1-S03"
      floor          int     1-indexed
      spot_in_floor  int     1-indexed (1 = nearest ramp/exit)
      size           str     compact | suv | van
      has_charger    bool    True if EV charging available
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows = []
    for floor in range(1, n_floors + 1):
        for spot in range(1, spaces_per_floor + 1):
            # EV charger: more common near the ramp (low spot numbers)
            charger_prob = (
                EV_CHARGER_PROB_NEAR if spot <= 5 else EV_CHARGER_PROB_OTHER
            )
            rows.append({
                "space_id":      f"F{floor}-S{spot:02d}",
                "floor":         floor,
                "spot_in_floor": spot,
                "size":          rng.choice(SIZE_LABELS, p=SIZE_PROBS),
                "has_charger":   bool(rng.random() < charger_prob),
            })

    return pd.DataFrame(rows).reset_index(drop=True)


# ── distance matrix ───────────────────────────────────────────────────────────

def build_distance_matrix(spaces: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute:
      exit_dist[i]      = walking distance from space i to the ground-floor exit (metres)
      dist_matrix[i, j] = walking distance between space i and space j (metres)

    Model:
      - Exit is on floor 1, at the ramp (spot 0).
      - Within-floor travel: |spot_i - spot_j| * DIST_PER_SPOT
      - Between-floor travel: |floor_i - floor_j| * FLOOR_BASE_DIST[2]
        (floor 2 base = 130m ≈ one full ramp traversal)
    """
    n = len(spaces)
    floors = spaces["floor"].values
    spots  = spaces["spot_in_floor"].values

    # Distance to exit: floor base + spot offset
    exit_dist = np.array([
        FLOOR_BASE_DIST.get(int(f), f * 80) + (s - 1) * DIST_PER_SPOT
        for f, s in zip(floors, spots)
    ], dtype=float)

    # Space-to-space matrix
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            floor_penalty = abs(int(floors[i]) - int(floors[j])) * 80  # 80m per floor change
            spot_dist     = abs(int(spots[i]) - int(spots[j])) * DIST_PER_SPOT
            dist_matrix[i, j] = floor_penalty + spot_dist

    return dist_matrix, exit_dist


# ── vehicle stream generation ─────────────────────────────────────────────────

def generate_vehicles(
    n_vehicles: int = 100,
    distributions: dict = None,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate a stream of arriving vehicles.

    Columns:
      vehicle_id      str     e.g. "V042"
      arrival_time    float   minutes from simulation start (sorted ascending)
      duration_min    float   how long the vehicle will stay
      vehicle_type    str     car | motorcycle | ev
      size_needed     str     compact | suv | van  (what space size the vehicle needs)
      needs_charger   bool    True iff EV and requires charging
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Default distributions (used if no Kaggle data provided)
    if distributions is None:
        distributions = {
            "vehicle_type_probs":    {"car": 0.65, "motorcycle": 0.15, "ev": 0.20},
            "ev_fraction":           0.20,
            "duration_lognormal":    {"mu": 3.5, "sigma": 0.6},
            "mean_interarrival_min": 3.0,
        }

    vtype_labels = list(distributions["vehicle_type_probs"].keys())
    vtype_probs  = list(distributions["vehicle_type_probs"].values())
    # Normalise (may not sum to exactly 1.0 after rounding)
    total = sum(vtype_probs)
    vtype_probs = [p / total for p in vtype_probs]

    ln = distributions["duration_lognormal"]
    mean_gap = distributions.get("mean_interarrival_min", 3.0)

    # Arrival times: Poisson process → exponential inter-arrivals
    inter_arrivals = rng.exponential(scale=mean_gap, size=n_vehicles)
    arrival_times  = np.cumsum(inter_arrivals)

    # Vehicle types
    vtypes = rng.choice(vtype_labels, size=n_vehicles, p=vtype_probs)

    # Parking durations (log-normal, clipped to [5, 720] minutes)
    raw_durations = rng.lognormal(mean=ln["mu"], sigma=ln["sigma"], size=n_vehicles)
    durations = np.clip(raw_durations, 5, 720)

    # Size needed: EVs and motorcycles get compact, cars/SUVs are mixed
    def size_for(vtype):
        if vtype == "motorcycle":
            return "compact"
        if vtype == "ev":
            return rng.choice(["compact", "suv"], p=[0.6, 0.4])
        return rng.choice(["compact", "suv", "van"], p=[0.50, 0.38, 0.12])

    sizes = [size_for(v) for v in vtypes]

    rows = []
    for i in range(n_vehicles):
        is_ev = vtypes[i] == "ev"
        rows.append({
            "vehicle_id":    f"V{i:03d}",
            "arrival_time":  round(arrival_times[i], 2),
            "duration_min":  round(durations[i], 1),
            "vehicle_type":  vtypes[i],
            "size_needed":   sizes[i],
            "needs_charger": is_ev,   # EVs always need a charger in our model
        })

    return pd.DataFrame(rows).sort_values("arrival_time").reset_index(drop=True)


# ── top-level factory ─────────────────────────────────────────────────────────

def create_instance(
    n_floors: int = 3,
    spaces_per_floor: int = 20,
    n_vehicles: int = 100,
    distributions: dict = None,
    seed: int = 42,
) -> ParkingInstance:
    """
    Create a complete, reproducible ParkingInstance.

    Parameters
    ----------
    n_floors          : number of parking levels
    spaces_per_floor  : spaces on each level
    n_vehicles        : vehicles to simulate
    distributions     : output of data_loader.extract_distributions()
                        (uses built-in defaults if None)
    seed              : RNG seed for full reproducibility
    """
    rng = np.random.default_rng(seed)

    spaces      = generate_spaces(n_floors, spaces_per_floor, rng)
    dist_matrix, exit_dist = build_distance_matrix(spaces)
    vehicles    = generate_vehicles(n_vehicles, distributions, rng)

    return ParkingInstance(
        n_floors=n_floors,
        spaces_per_floor=spaces_per_floor,
        spaces=spaces,
        vehicles=vehicles,
        dist_matrix=dist_matrix,
        exit_dist=exit_dist,
        ramp_capacity=RAMP_CAPACITY,
        seed=seed,
    )


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inst = create_instance(n_floors=3, spaces_per_floor=20, n_vehicles=60, seed=42)
    print(inst.summary())
    print("\nFirst 5 spaces:")
    print(inst.spaces.head())
    print("\nFirst 5 vehicles:")
    print(inst.vehicles.head())
    print(f"\nExit distances (first 10): {inst.exit_dist[:10].round(1)}")
