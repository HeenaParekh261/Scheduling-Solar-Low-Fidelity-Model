import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from data_loader import load_data
from genetic_algorithm import genetic_algorithm

if __name__ == "__main__":
    X, y = load_data()
    if X is None or y is None:
        print("Data loading failed. Exiting.")
        exit()

    X = np.hstack([X, np.ones((X.shape[0], 1))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    best_weights, best_fitness_per_gen = genetic_algorithm(
        X_train_scaled, y_train,
        population_size=200,
        generations=20000,
        elite_size=2,
        save_path="best_weights.csv"
    )

    y_pred_ga = X_test_scaled @ best_weights
    mse_ga = mean_squared_error(y_test, y_pred_ga)
    rmse_ga = np.sqrt(mse_ga)
    print(f"\n[GA-based Linear Model] Test MSE={mse_ga:.4f}, RMSE={rmse_ga:.4f}")

    plt.figure()
    plt.plot(best_fitness_per_gen, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score (-log(MSE))")
    plt.title("GA Training Fitness Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    print(f"[Ordinary LinearRegression] Test MSE={mse_lr:.4f}, RMSE={rmse_lr:.4f}")

    print("\nComparing GA vs. LinearRegression vs. Ground Truth (first 5 samples):")
    for i in range(5):
        print(f"Sample {i} => GA: {y_pred_ga[i]:.3f}, LR: {y_pred_lr[i]:.3f}, Actual: {y_test[i]:.3f}")
