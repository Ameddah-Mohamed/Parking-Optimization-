import time
import copy
from typing import Dict, Any, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.models.datatypes import ParkingInstance
from src.evaluation.cost_func import get_default_weights, compute_cost

class ExactBranchAndBound:
    def __init__(self, time_limit: int = 300, weights: Optional[Dict[str, float]] = None):
        self.time_limit = time_limit
        self.weights = weights or get_default_weights()
        self.best_cost = float('inf')
        self.best_mapping = {}
        self.start_time = 0.0

    def solve(self, instance: ParkingInstance) -> Dict[str, Any]:
        self.start_time = time.perf_counter()
        self.best_cost = float('inf')
        
        # Sort vehicles by arrival time
        sorted_vehicles = sorted(instance.vehicles, key=lambda v: v.arrival_time)
        
        # Precompute feasible spots for each vehicle to speed up branching
        feasible_spots = {}
        for v in sorted_vehicles:
            valid = []
            for s in instance.spots:
                if s.size >= v.size_needed and (not v.needs_charger or s.has_charger):
                    valid.append(s)
            # Sort valid spots by base distance (greedy ordering)
            valid.sort(key=lambda s: s.walk_dist_to_exit)
            feasible_spots[v.id] = valid
            
        initial_occupancy = {lvl.id: 0 for lvl in instance.levels}
        
        # Branch and bound recursive call
        self._branch(0, {}, 0.0, sorted_vehicles, feasible_spots, initial_occupancy, [])
        
        return {
            'mapping': self.best_mapping,
            'runtime': time.perf_counter() - self.start_time,
            'status': 'Optimal' if time.perf_counter() - self.start_time < self.time_limit else 'TimeLimit'
        }
        
    def _branch(self, v_idx: int, current_mapping: Dict[str, str], current_cost: float,
                vehicles: List[Any], feasible_spots: Dict[str, List[Any]], 
                occupancy: Dict[int, int], active_intervals: List[tuple]):
        
        # Check time limit
        if time.perf_counter() - self.start_time > self.time_limit:
            return
            
        # Base case: all vehicles assigned
        if v_idx == len(vehicles):
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_mapping = copy.deepcopy(current_mapping)
            return
            
        v = vehicles[v_idx]
        
        # Clean up active_intervals (vehicles that have departed by v.arrival_time)
        new_active = []
        for interval in active_intervals:
            dep_time, sid, lvl = interval
            if dep_time > v.arrival_time:
                new_active.append(interval)
            else:
                occupancy[lvl] -= 1
                
        active_sids = {interval[1] for interval in new_active}
        
        # Calculate lower bound for remaining unassigned
        lb_remaining = 0.0
        # Very loose bound: sum of best possible non-overlapping cost ignoring congestion
        # For a tight bound, we'd solve an assignment problem, but this is simple.
        lb = current_cost + lb_remaining
        
        if lb >= self.best_cost:
            return # Prune
            
        # Try to assign to a valid spot
        assigned = False
        for s in feasible_spots[v.id]:
            # Check overlap
            if s.id in active_sids:
                continue
                
            assigned = True
            # Compute cost 
            c = compute_cost(v, s, self.weights, occupancy[s.level_id])
            
            if current_cost + c >= self.best_cost:
                continue # Prune this branch
                
            # Branch
            current_mapping[v.id] = s.id
            occupancy[s.level_id] += 1
            new_active.append((v.departure_time, s.id, s.level_id))
            
            self._branch(v_idx + 1, current_mapping, current_cost + c, vehicles, 
                         feasible_spots, occupancy, new_active)
                         
            # Backtrack
            del current_mapping[v.id]
            occupancy[s.level_id] -= 1
            new_active.pop()
            
        # Option to leave unassigned
        unassigned_pen = self.weights.get('unassigned_penalty', 1e5)
        if current_cost + unassigned_pen < self.best_cost:
            current_mapping[v.id] = None
            self._branch(v_idx + 1, current_mapping, current_cost + unassigned_pen, vehicles,
                         feasible_spots, occupancy, new_active)
            del current_mapping[v.id]
