import math
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.models.datatypes import Vehicle, ParkingSpot, ParkingInstance

def get_default_weights() -> Dict[str, float]:
    return {
        'w1': 1.0,                    # Walk distance multiplier
        'w2': 0.5,                    # Congestion multiplier
        'w3': 10.0,                   # Level penalty multiplier
        'congestion_multiplier': 5.0, # Base cost per occupied spot on same level
        'level_multiplier': 20.0,     # Penalty per floor level
        'ev_penalty': 1e6,            # Soft constraint violation penalty
        'size_penalty': 1e6           # Soft constraint violation penalty
    }

def compute_cost(v: Vehicle, s: ParkingSpot, weights: Dict[str, float], level_occupancy: int) -> float:
    """
    Computes the objective cost of assigning Vehicle v to ParkingSpot s.
    
    Parameters:
      level_occupancy: The number of vehicles currently parked on the spot's level
                       at the exact time the vehicle arrives.
    """
    cost = 0.0
    
    # 1. Size compatibility
    if s.size < v.size_needed:
        cost += weights['size_penalty']
        
    # 2. EV compatibility
    if v.needs_charger and not s.has_charger:
        cost += weights['ev_penalty']
        
    # 3. Distance Cost
    dist_cost = weights['w1'] * s.walk_dist_to_exit
    cost += dist_cost
    
    # 4. Congestion Penalty (penalty based on how full the level is)
    cong_cost = weights['w2'] * (level_occupancy * weights['congestion_multiplier'])
    cost += cong_cost
    
    # 5. Level Penalty (discourage higher floors if lower are available)
    level_pen = weights['w3'] * (s.level_id * weights['level_multiplier'])
    cost += level_pen
    
    return cost

def evaluate_assignment(mapping: Dict[str, str], instance: ParkingInstance, weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluates a full mapping of {vehicle_id: spot_id} and returns metrics.
    Assumes assignments are valid (no overlapping times).
    """
    v_dict = {v.id: v for v in instance.vehicles}
    s_dict = {s.id: s for s in instance.spots}
    
    total_cost = 0.0
    unassigned = 0
    size_violations = 0
    ev_violations = 0
    total_dist = 0.0
    
    # Track occupancy over time for congestion calculation
    # Sort events
    events = []
    for vid, sid in mapping.items():
        if sid is None: continue
        v = v_dict[vid]
        events.append((v.arrival_time, 'start', vid, sid))
        events.append((v.departure_time, 'end', vid, sid))
    events.sort(key=lambda x: (x[0], x[1]=='start'))
    
    # Simulate time to compute exact costs
    active_levels = {lvl.id: 0 for lvl in instance.levels}
    
    for t, e_type, vid, sid in events:
        s = s_dict[sid]
        v = v_dict[vid]
        
        if e_type == 'start':
            # Compute cost at the moment of arrival
            c = compute_cost(v, s, weights, active_levels[s.level_id])
            total_cost += c
            total_dist += s.walk_dist_to_exit
            
            if s.size < v.size_needed: size_violations += 1
            if v.needs_charger and not s.has_charger: ev_violations += 1
            
            active_levels[s.level_id] += 1
        else:
            active_levels[s.level_id] -= 1
            
    for v in instance.vehicles:
        if mapping.get(v.id) is None:
            unassigned += 1
            total_cost += weights.get('unassigned_penalty', 1e5)
            
    assigned_count = len(instance.vehicles) - unassigned
    avg_dist = (total_dist / assigned_count) if assigned_count > 0 else 0.0
    
    return {
        'total_cost': total_cost,
        'avg_distance': avg_dist,
        'unassigned': unassigned,
        'size_violations': size_violations,
        'ev_violations': ev_violations
    }
