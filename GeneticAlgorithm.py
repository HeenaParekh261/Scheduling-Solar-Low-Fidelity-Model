#Base code taken from https://gist.github.com/AnasBrital98/5802390bd20cba541af41f483fa8fe4c#file-geneticalgorithmexample1-py

import random
from Population import Population
from chromosome import Chromosome
import numpy as np
class GeneticAlgorithm : 
    
    def __init__(self , populationSize , chromosomeSize , tournamentSize , elitismSize , mutationRate , function):
        self.populationSize = populationSize
        self.chromosomeSize = chromosomeSize
        self.tournamentSize = tournamentSize
        self.elitismSize    = elitismSize
        self.mutationRate   = mutationRate
        self.function       = function
    
        
    def reproduction(self , population):
        temp = []
        temp[:self.elitismSize] = population.getNFittestChromosomes(self.elitismSize)
        for i in range(self.elitismSize , self.populationSize):
            parent1 = self.tournamentSelection(population)
            parent2 = self.tournamentSelection(population)
            
            child = self.onePointCrossOver(parent1, parent2)
            
            self.swapMutation(child)
            
            temp.append(child)
            
        newPopulation = Population(self.populationSize, self.chromosomeSize, self.function, False)
        newPopulation.chromosomes = temp
        newPopulation.findTheFittest()
        newPopulation.calculateTheFitnessForAll()
        return newPopulation

    def swapMutation(self , chromosome):
        if random.random() < self.mutationRate:
            layer_idx = random.randint(0, len(chromosome.weights) - 1)
            i = random.randint(0, chromosome.weights[layer_idx].shape[0] - 1)
            j = random.randint(0, chromosome.weights[layer_idx].shape[1] - 1)
            chromosome.weights[layer_idx][i][j] += np.random.normal()
            chromosome.fitness = chromosome.calculateFitness()
    
    def tournamentSelection(self , population):
        tournamentPool = []
        for i in range(self.tournamentSize):
            index = random.randint(0, len(population.chromosomes) -1)
            tournamentPool.append(population.chromosomes[index])
        tournamentPool.sort(key = lambda x:x.fitness)
        return tournamentPool[0]
    
    def onePointCrossOver(self , parent1 , parent2):
        child_weights = []
        child_biases = []
        for w1, w2 in zip(parent1.weights, parent2.weights):
            mask = np.random.rand(*w1.shape) < 0.5
            child_w = np.where(mask, w1, w2)
            child_weights.append(child_w)
        for b1, b2 in zip(parent1.biases, parent2.biases):
            mask = np.random.rand(*b1.shape) < 0.5
            child_b = np.where(mask, b1, b2)
            child_biases.append(child_b)
        return Chromosome(parent1.NeuralNetwork, weights=child_weights, biases=child_biases)
