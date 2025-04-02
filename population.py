import numpy as np
import random
from fitness_function import fitness_function

class Population:
    def __init__(self, size, num_features, X, y):
        self.size = size
        self.X = X
        self.y = y
        self.individuals = [np.random.uniform(-1, 1, num_features) for _ in range(size)]

    def evaluate(self):
        scores = [fitness_function(ind, self.X, self.y) for ind in self.individuals]
        return scores

    def select_parents(self, scores):
        # Roulette wheel selection
        total_score = sum(scores)
        probs = [s / total_score for s in scores]
        parents = random.choices(self.individuals, weights=probs, k=2)
        return parents

    def crossover(self, parent1, parent2):
        alpha = 0.5
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def mutate(self, individual, mutation_rate=0.1):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
        return individual

    def evolve(self):
        scores = self.evaluate()
        new_generation = []
        for _ in range(self.size):
            p1, p2 = self.select_parents(scores)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_generation.append(child)
        self.individuals = new_generation
