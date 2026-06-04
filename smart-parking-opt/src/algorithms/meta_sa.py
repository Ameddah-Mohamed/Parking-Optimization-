import time
import random
import math
from typing import Dict, Any, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.models.datatypes import ParkingInstance
from src.evaluation.cost_func import get_default_weights, evaluate_assignment

class SimulatedAnnealing:
    def __init__(self, initial_temp: float = 1000.0, alpha: float = 0.95, 
                 iterations_per_temp: int = 50, min_temp: float = 0.1,
                 weights: Optional[Dict[str, float]] = None):
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp
        self.min_temp = min_temp
        self.weights = weights or get_default_weights()

    def solve(self, instance: ParkingInstance) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        self.vehicles = instance.vehicles
        self.spots = instance.spots
        self.instance = instance
        
        # Valid spots per vehicle
        self.valid_spots = {}
        for v in self.vehicles:
            valid = [s.id for s in self.spots if s.size >= v.size_needed and (not v.needs_charger or s.has_charger)]
            self.valid_spots[v.id] = valid if valid else [None]
            
        # Initial solution
        current_sol = self._generate_initial_solution()
        current_cost = self._evaluate(current_sol)
        
        best_sol = current_sol.copy()
        best_cost = current_cost
        
        temp = self.initial_temp
        cost_history = []
        
        while temp > self.min_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self._get_neighbor(current_sol)
                neighbor_cost = self._evaluate(neighbor)
                
                delta = neighbor_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_sol = neighbor
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_sol = current_sol.copy()
                        
            cost_history.append(best_cost)
            temp *= self.alpha
            
        runtime = time.perf_counter() - t0
        return {
            'mapping': best_sol,
            'runtime': runtime,
            'best_cost': best_cost,
            'history': cost_history
        }

    def _generate_initial_solution(self) -> Dict[str, str]:
        sol = {}
        for v in self.vehicles:
            sol[v.id] = random.choice(self.valid_spots[v.id])
        return self._repair(sol)

    def _evaluate(self, sol: Dict[str, str]) -> float:
        res = evaluate_assignment(sol, self.instance, self.weights)
        return res['total_cost']

    def _get_neighbor(self, sol: Dict[str, str]) -> Dict[str, str]:
        neighbor = sol.copy()
        v = random.choice(self.vehicles)
        neighbor[v.id] = random.choice(self.valid_spots[v.id])
        return self._repair(neighbor)

    def _repair(self, chromo: Dict[str, str]) -> Dict[str, str]:
        sorted_v = sorted(self.vehicles, key=lambda x: x.arrival_time)
        active_intervals = []
        
        for v in sorted_v:
            sid = chromo[v.id]
            if sid is None: continue
            
            active_intervals = [i for i in active_intervals if i[0] > v.arrival_time]
            active_sids = {i[1] for i in active_intervals}
            
            if sid in active_sids:
                valid = [s for s in self.valid_spots[v.id] if s not in active_sids]
                if valid:
                    new_sid = random.choice(valid)
                    chromo[v.id] = new_sid
                    active_intervals.append((v.departure_time, new_sid))
                else:
                    chromo[v.id] = None
            else:
                active_intervals.append((v.departure_time, sid))
                
        return chromo
