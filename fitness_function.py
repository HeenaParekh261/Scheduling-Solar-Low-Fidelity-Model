import numpy as np

def fitness_function(individual, X, y):
    predictions = X @ individual
    predictions = predictions.reshape(-1, 1)  # reshape to (n, 1)
    mse = np.mean((predictions - y) ** 2)
    return 1 / (mse + 1e-6)

if __name__ == "__main__":
    from data_loader import load_data
    X, y = load_data("Copy of Dataset - Predictive Tool Development for Residential Solar Installation Duration.xlsx")
    weights = np.random.rand(X.shape[1])
    score = fitness_function(weights, X, y)
    print("Fitness Score:", score)