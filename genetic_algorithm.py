import numpy as np
import pandas as pd
from population import Population


def genetic_algorithm(X, y, population_size=100, generations=100000, save_path="best_weights.csv"):
    """
    Main function of the genetic algorithm: Optimize linear model weights to minimize prediction error.

    Parameters:
        X : np.ndarray, feature matrix
        y : np.ndarray, target variable (in hours)
        population_size : int, number of individuals in each generation
        generations : int, number of iterations
        save_path : str, the path to save the best weights in CSV format

    Returns:
        best_weights : np.ndarray, the optimal weights
        best_fitness_per_gen : list[float], the best fitness value in each generation
    """
    pop = Population(size=population_size, num_features=X.shape[1], X=X, y=y)
    best_fitness_per_gen = []

    for gen in range(generations):
        pop.evolve()
        scores = pop.evaluate()
        best_score = max(scores)
        best_fitness_per_gen.append(best_score)
        print(f"Generation {gen + 1} - Best Fitness Score: {best_score:.6f}")

    final_scores = pop.evaluate()
    best_idx = np.argmax(final_scores)
    best_weights = pop.individuals[best_idx]

    # Save the best weights
    weights_df = pd.DataFrame(best_weights, columns=["Weight"])
    weights_df.index.name = "Feature_Index"
    weights_df.to_csv(save_path)
    print(f"Best weights saved to {save_path}")

    return best_weights, best_fitness_per_gen