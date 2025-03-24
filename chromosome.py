#Base code from https://gist.github.com/AnasBrital98/0fe7043beeda9d116ec1668350c33a22#file-chromosomeexample2-py
class Chromosome:
    def __init__(self , NeuralNetwork , weights = None , biases = None):
        self.NeuralNetwork = NeuralNetwork
        self.weights = weights if weights != None else [np.random.randn(self.NeuralNetwork.layers[i] , self.NeuralNetwork.layers[i-1]) for i in range(1 , len(self.NeuralNetwork.layers))]
        self.biases  =  biases if biases != None else [np.random.randn(self.NeuralNetwork.layers[i] , 1) for i in range(1 , len(self.NeuralNetwork.layers))]
        self.fitness = self.calculateFitness()
        
    def calculateFitness(self):
