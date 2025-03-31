# Base code from https://gist.github.com/AnasBrital98/6cc9964230a3aa956d78c6c22ab98616#file-populationexample1-py

from chromosome import Chromosome

class Population :
    
    def __init__(self , populationSize , chromosomeSize , function , init):
        self.chromosomes = []
        if init :
            self.chromosomes = [Chromosome(chromosomeSize , function) for i in range(populationSize)]
            self.chromosomes.sort(key = lambda x:x.fitness)
            self.fittest = self.chromosomes[0]
    
    def getNFittestChromosomes(self, n):
        self.chromosomes.sort(key = lambda x:x.fitness)
        return self.chromosomes[:n]
    
    def findTheFittest(self):
           self.chromosomes.sort(key = lambda x:x.fitness)
           self.fittest = self.chromosomes[0]
    
    def calculateTheFitnessForAll(self):
        for chromosome in self.chromosomes:
            chromosome.calculateTheFitness()
