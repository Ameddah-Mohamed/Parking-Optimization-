import numpy as np
from typing import List, Dict
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.models.datatypes import Vehicle, ParkingSpot, ParkingLevel, ParkingInstance

def generate_parking_instance(
    name: str = "instance_01",
    num_vehicles: int = 100,
    num_spots_per_level: int = 30,
    num_levels: int = 3,
    seed: int = 42
) -> ParkingInstance:
    """Generates a synthetic parking instance based on given parameters."""
    rng = np.random.default_rng(seed)

    # 1. Generate Levels
    levels = []
    base_dist = 50.0 # Floor 1 distance
    for i in range(1, num_levels + 1):
        levels.append(ParkingLevel(id=i, capacity=num_spots_per_level, base_distance=base_dist))
        base_dist += 80.0 # Each floor adds 80m walking distance to exit

    # 2. Generate Spots
    spots = []
    for level in levels:
        for spot_idx in range(1, level.capacity + 1):
            # Size: 1 (Compact 50%), 2 (SUV 35%), 3 (Van 15%)
            size = rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15])
            
            # EV Charger logic
            charger_prob = 0.40 if level.id == 1 else 0.10
            has_charger = rng.random() < charger_prob
            
            # Distance
            dist = level.base_distance + (spot_idx * 3.5)
            
            spot_id = f"L{level.id}_S{spot_idx:03d}"
            spots.append(ParkingSpot(
                id=spot_id, level_id=level.id, spot_index=spot_idx,
                size=size, has_charger=has_charger, walk_dist_to_exit=dist
            ))

    # 3. Generate Distance Matrix
    num_spots = len(spots)
    dist_matrix = np.zeros((num_spots, num_spots))
    for i in range(num_spots):
        for j in range(num_spots):
            if i != j:
                s1, s2 = spots[i], spots[j]
                level_diff = abs(s1.level_id - s2.level_id)
                spot_diff = abs(s1.spot_index - s2.spot_index)
                dist_matrix[i, j] = (level_diff * 80.0) + (spot_diff * 3.5)

    # 4. Generate Vehicles
    vehicles = []
    current_time = 0.0
    for i in range(num_vehicles):
        # Inter-arrival time exponential with mean 3 minutes
        inter_arrival = rng.exponential(scale=3.0)
        current_time += inter_arrival
        
        # Duration lognormal (mean ~120 mins)
        duration = min(max(rng.lognormal(mean=4.5, sigma=0.8), 10.0), 720.0)
        
        # Type and EV
        is_ev = rng.random() < 0.20
        v_type = 'ev' if is_ev else rng.choice(['car', 'suv', 'van'], p=[0.50, 0.35, 0.15])
        
        if v_type == 'car' or v_type == 'ev':
            size_needed = rng.choice([1, 2], p=[0.7, 0.3])
        elif v_type == 'suv':
            size_needed = 2
        else:
            size_needed = 3

        vehicles.append(Vehicle(
            id=f"V{i:04d}", v_type=v_type, size_needed=size_needed,
            arrival_time=round(current_time, 2), duration=round(duration, 2),
            needs_charger=is_ev
        ))

    return ParkingInstance(
        name=name, vehicles=vehicles, spots=spots, levels=levels, distance_matrix=dist_matrix
    )

if __name__ == "__main__":
    inst = generate_parking_instance()
    print(f"Generated instance {inst.name} with {len(inst.vehicles)} vehicles and {len(inst.spots)} spots.")
