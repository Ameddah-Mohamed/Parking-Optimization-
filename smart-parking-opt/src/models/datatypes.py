from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Vehicle:
    id: str
    v_type: str              # 'car', 'ev', 'van', 'motorcycle'
    size_needed: int         # 1: compact, 2: suv, 3: van
    arrival_time: float      # Minutes from start of day
    duration: float          # Parking duration in minutes
    needs_charger: bool      # True if EV needs charging

    @property
    def departure_time(self) -> float:
        return self.arrival_time + self.duration

@dataclass
class ParkingSpot:
    id: str
    level_id: int
    spot_index: int
    size: int                # 1: compact, 2: suv, 3: van
    has_charger: bool        # True if equipped with EV charger
    walk_dist_to_exit: float # Pre-calculated distance to nearest exit

@dataclass
class ParkingLevel:
    id: int
    capacity: int
    base_distance: float     # Distance from this level to the main exit

@dataclass
class ParkingInstance:
    name: str
    vehicles: List[Vehicle]
    spots: List[ParkingSpot]
    levels: List[ParkingLevel]
    distance_matrix: np.ndarray # spot-to-spot distance for neighborhood ops
