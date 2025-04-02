import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data
from genetic_algorithm import genetic_algorithm

# load_data
X, y = load_data()
if X is None or y is None:
    print("Data loader failed")
    exit()
y = y/60

# run genetic algorithm
best_weights, best_fitness_per_gen = genetic_algorithm(X, y, save_path="best_weights.csv")

print("\nBest Weights Found:")
print(best_weights)

# predict and compare
predictions = X @ best_weights
print("\nSample Predictions vs Actual:")
y = y.flatten()
for pred, actual in zip(predictions[:5], y[:5]):
    print(f"Predicted: {pred:.2f} h, Actual: {actual:.2f} h")

# Save the fitness curve
pd.DataFrame(best_fitness_per_gen, columns=["Fitness"]).to_csv("fitness_history.csv", index_label="Generation")

# Visualize the result
# Fitness according to generation
plt.figure(figsize=(10, 5))
plt.plot(best_fitness_per_gen, label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness Score (1/MSE)")
plt.title("Genetic Algorithm Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# actual V.S. prediction
plt.figure(figsize=(6, 6))
plt.scatter(y, predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Total Time (h)")
plt.ylabel("Predicted Total Time (h)")
plt.title("Prediction vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()