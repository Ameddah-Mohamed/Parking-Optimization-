import time
import random
import numpy as np
from typing import Dict, Any, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.models.datatypes import ParkingInstance
from src.evaluation.cost_func import get_default_weights, evaluate_assignment

class GeneticAlgorithm:
    def __init__(self, pop_size: int = 50, generations: int = 100, mutation_rate: float = 0.2, 
                 weights: Optional[Dict[str, float]] = None):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
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

        # Initialize population
        population = self._initialize_population()
        best_solution = None
        best_cost = float('inf')
        cost_history = []
        
        for gen in range(self.generations):
            # Evaluate fitness
            scored_pop = []
            for chromo in population:
                cost = self._evaluate_chromosome(chromo)
                scored_pop.append((cost, chromo))
                if cost < best_cost:
                    best_cost = cost
                    best_solution = chromo
                    
            cost_history.append(best_cost)
                    
            # Selection (Tournament)
            scored_pop.sort(key=lambda x: x[0])
            new_pop = [scored_pop[0][1]] # Elitism
            
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(scored_pop)
                p2 = self._tournament_selection(scored_pop)
                child = self._crossover(p1, p2)
                self._mutate(child)
                child = self._repair(child) # Fix overlaps
                new_pop.append(child)
                
            population = new_pop
            
        runtime = time.perf_counter() - t0
        return {
            'mapping': best_solution,
            'runtime': runtime,
            'best_cost': best_cost,
            'history': cost_history
        }

    def _initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            chromo = {}
            for v in self.vehicles:
                chromo[v.id] = random.choice(self.valid_spots[v.id])
            pop.append(self._repair(chromo))
        return pop

    def _evaluate_chromosome(self, chromo: Dict[str, str]) -> float:
        res = evaluate_assignment(chromo, self.instance, self.weights)
        return res['total_cost']

    def _tournament_selection(self, scored_pop, k=3):
        candidates = random.sample(scored_pop, k)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _crossover(self, p1: Dict[str, str], p2: Dict[str, str]) -> Dict[str, str]:
        # Uniform crossover
        child = {}
        for v in self.vehicles:
            child[v.id] = p1[v.id] if random.random() < 0.5 else p2[v.id]
        return child

    def _mutate(self, chromo: Dict[str, str]):
        for v in self.vehicles:
            if random.random() < self.mutation_rate:
                chromo[v.id] = random.choice(self.valid_spots[v.id])

    def _repair(self, chromo: Dict[str, str]) -> Dict[str, str]:
        # Sort vehicles by arrival time to resolve conflicts chronologically
        sorted_v = sorted(self.vehicles, key=lambda x: x.arrival_time)
        active_intervals = [] # (departure, sid)
        
        for v in sorted_v:
            sid = chromo[v.id]
            if sid is None: continue
            
            # Clean active
            active_intervals = [i for i in active_intervals if i[0] > v.arrival_time]
            active_sids = {i[1] for i in active_intervals}
            
            if sid in active_sids:
                # Conflict! Find a new spot
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
