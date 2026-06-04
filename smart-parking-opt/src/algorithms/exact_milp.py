import time
import pulp
from typing import Dict, Any, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.models.datatypes import ParkingInstance
from src.evaluation.cost_func import get_default_weights, compute_cost

class ExactMILPSolver:
    def __init__(self, time_limit: int = 300, weights: Optional[Dict[str, float]] = None):
        self.time_limit = time_limit
        self.weights = weights or get_default_weights()

    def solve(self, instance: ParkingInstance) -> Dict[str, Any]:
        """
        Solves the dynamic assignment problem optimally using PuLP.
        """
        t0 = time.perf_counter()
        
        prob = pulp.LpProblem("SmartParking_MILP", pulp.LpMinimize)
        
        V = [v.id for v in instance.vehicles]
        S = [s.id for s in instance.spots]
        
        # Variables
        # x[v, s] = 1 if vehicle v is assigned to spot s
        x = pulp.LpVariable.dicts("x", [(v, s) for v in V for s in S], cat="Binary")
        
        # y[v] = 1 if vehicle v is unassigned
        y = pulp.LpVariable.dicts("y", V, cat="Binary")
        
        v_dict = {v.id: v for v in instance.vehicles}
        s_dict = {s.id: s for s in instance.spots}
        
        # Precompute costs and filter infeasible
        cost = {}
        feasible = {}
        for vid in V:
            v = v_dict[vid]
            for sid in S:
                s = s_dict[sid]
                # For static/batch MILP, congestion penalty is hard to linearize perfectly
                # without huge number of variables. We use a static base cost based on distance and level.
                c = compute_cost(v, s, self.weights, level_occupancy=0)
                
                # Check hard constraints encoded as large penalties
                if c >= self.weights['size_penalty']:
                    feasible[(vid, sid)] = False
                    cost[(vid, sid)] = 0.0
                else:
                    feasible[(vid, sid)] = True
                    cost[(vid, sid)] = c

        # Force infeasible to 0
        for vid in V:
            for sid in S:
                if not feasible[(vid, sid)]:
                    prob += x[(vid, sid)] == 0

        # Objective
        unassigned_penalty = self.weights.get('unassigned_penalty', 1e5)
        prob += pulp.lpSum(cost[(v, s)] * x[(v, s)] for v in V for s in S if feasible[(v, s)]) + \
                pulp.lpSum(unassigned_penalty * y[v] for v in V)

        # Constraint 1: exactly one assignment or unassigned
        for vid in V:
            prob += pulp.lpSum(x[(vid, s)] for s in S) + y[vid] == 1

        # Constraint 2: Interval Graph cliques (no overlapping times on same spot)
        events = []
        for vid in V:
            v = v_dict[vid]
            events.append((v.arrival_time, 'start', vid))
            events.append((v.departure_time, 'end', vid))
            
        events.sort(key=lambda e: (e[0], e[1] == 'start'))
        
        active = set()
        cliques = []
        for t, e_type, vid in events:
            if e_type == 'start':
                active.add(vid)
                if len(active) > 1:
                    cliques.append(list(active))
            else:
                active.remove(vid)
                
        for sid in S:
            for clique in cliques:
                prob += pulp.lpSum(x[(vid, sid)] for vid in clique) <= 1

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)
        prob.solve(solver)

        # Extract solution
        mapping = {}
        for vid in V:
            mapping[vid] = None
            for sid in S:
                if pulp.value(x[(vid, sid)]) and pulp.value(x[(vid, sid)]) > 0.5:
                    mapping[vid] = sid
                    break
                    
        total_time = time.perf_counter() - t0
        
        # We don't return prob.objective because actual evaluation uses dynamic congestion
        return {
            'mapping': mapping,
            'runtime': total_time,
            'status': pulp.LpStatus[prob.status]
        }
