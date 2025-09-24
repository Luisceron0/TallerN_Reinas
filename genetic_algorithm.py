"""
Implementación del Algoritmo Genético para el problema de N-Reinas.
"""

import random
import copy
import numpy as np
from utils import conflicts_count, max_pairs

class GeneticPermutation:
    def __init__(self, N=8, mu=50, lam=None, generations=500,
                 mutation_prob=0.15, tournament_k=3,
                 replacement_mode='mu_plus_lambda', seed=None):
        self.N = int(N)
        self.mu = int(mu)
        self.lam = int(lam) if lam is not None else int(mu)
        self.generations = int(generations)
        self.mutation_prob = float(mutation_prob)
        self.tournament_k = max(2, int(tournament_k))
        assert replacement_mode in ('mu_plus_lambda', 'mu_comma_lambda')
        self.replacement_mode = replacement_mode
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._init_population()

    def _init_population(self):
        self.population = []
        for _ in range(self.mu):
            ind = list(range(self.N))
            random.shuffle(ind)
            self.population.append(ind)
        self.aptitudes = [self._apt(ind) for ind in self.population]
        self.generation = 0

    def _apt(self, ind):
        return max_pairs(self.N) - conflicts_count(ind)

    @staticmethod
    def ox_crossover(p1, p2):
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b+1] = p1[a:b+1]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                while p2[idx] in child:
                    idx += 1
                child[i] = p2[idx]
                idx += 1
        return child

    def swap_mutation(self, ind):
        ind = ind.copy()
        if random.random() < self.mutation_prob:
            i, j = random.sample(range(self.N), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    def _tournament_select(self, population, aptitudes, k, select_count):
        selected = []
        pop_len = len(population)
        k = min(k, pop_len)
        for _ in range(select_count):
            participants = random.sample(range(pop_len), k)
            winner_idx = max(participants, key=lambda x: aptitudes[x])
            selected.append(copy.deepcopy(population[winner_idx]))
        return selected

    def _roulette_indices(self, aptitudes_array, k):
        arr = np.array(aptitudes_array, dtype=float)
        arr = arr - arr.min() + 1e-8
        probs = arr / arr.sum()
        replace_flag = False if k <= len(arr) else True
        return np.random.choice(len(arr), size=k, replace=replace_flag, p=probs)

    def _replace(self, parents, parents_fit, offspring, offspring_fit):
        if self.replacement_mode == 'mu_plus_lambda':
            combined = parents + offspring
            combined_fit = parents_fit + offspring_fit
            idxs = self._roulette_indices(combined_fit, self.mu)
            new_pop = [copy.deepcopy(combined[i]) for i in idxs]
            new_fit = [self._apt(ind) for ind in new_pop]
            return new_pop, new_fit
        else:
            if len(offspring) >= self.mu:
                idxs = self._roulette_indices(offspring_fit, self.mu)
                new_pop = [copy.deepcopy(offspring[i]) for i in idxs]
                new_fit = [self._apt(ind) for ind in new_pop]
                return new_pop, new_fit
            else:
                new_pop = [copy.deepcopy(ind) for ind in offspring]
                new_fit = [self._apt(ind) for ind in new_pop]
                ordered_parents = sorted(zip(parents, parents_fit), key=lambda x: x[1], reverse=True)
                i = 0
                while len(new_pop) < self.mu:
                    new_pop.append(copy.deepcopy(ordered_parents[i][0]))
                    new_fit.append(self._apt(ordered_parents[i][0]))
                    i += 1
                return new_pop, new_fit

    def step(self):
        # produce lambda parents via tournament
        parents = self._tournament_select(self.population, self.aptitudes, self.tournament_k, self.lam)
        # crossover + mutation
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                p1, p2 = parents[i], parents[i + 1]
                c1 = self.ox_crossover(p1, p2)
                c2 = self.ox_crossover(p2, p1)
                offspring.append(self.swap_mutation(c1))
                offspring.append(self.swap_mutation(c2))
            else:
                offspring.append(self.swap_mutation(parents[i]))
        offspring = offspring[:self.lam]
        offspring_fit = [self._apt(ind) for ind in offspring]
        # replacement
        new_pop, new_fit = self._replace(self.population, self.aptitudes, offspring, offspring_fit)
        self.population = new_pop
        self.aptitudes = new_fit
        self.generation += 1
        best_idx = int(np.argmax(self.aptitudes))
        return {
            'generation': self.generation,
            'best_individual': self.population[best_idx],
            'best_fitness': self.aptitudes[best_idx],
            'avg_fitness': float(np.mean(self.aptitudes))
        }
