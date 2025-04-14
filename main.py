import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from data_loader import load_data
from genetic_algorithm import genetic_algorithm

def generate_nonlinear_features(X):
    X_squared = X ** 2
    X_log = np.log1p(np.abs(X))
    X_sqrt = np.sqrt(np.abs(X))
    return np.hstack([X, X_squared, X_log, X_sqrt])

if __name__ == "__main__":
    X, y = load_data()
    if X is None or y is None:
        print("Data loading failed. Exiting.")
        exit()

    X = np.hstack([X, np.ones((X.shape[0], 1))])

    X = generate_nonlinear_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_all_scaled = scaler_X.transform(X)

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

    y_pred_all_ga = X_all_scaled @ best_weights

    plt.figure()
    plt.scatter(y, y_pred_all_ga, alpha=0.6, s=20, label="GA Prediction")
    min_val = min(np.min(y), np.min(y_pred_all_ga))
    max_val = max(np.max(y), np.max(y_pred_all_ga))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
    plt.xlabel("Actual Duration (minutes)")
    plt.ylabel("Predicted Duration (minutes)")
    plt.title("GA Prediction vs. Actual (All Data)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    from sklearn.linear_model import LinearRegression

    bias_adjust = LinearRegression(fit_intercept=True)
    bias_adjust.fit(y_pred_all_ga.reshape(-1, 1), y)  # y ≈ prediction + b
    y_pred_adjusted = bias_adjust.predict(y_pred_all_ga.reshape(-1, 1))

    plt.figure()
    plt.scatter(y, y_pred_adjusted, alpha=0.6, s=20, label="GA + Bias Adjust")
    min_val = min(np.min(y), np.min(y_pred_adjusted))
    max_val = max(np.max(y), np.max(y_pred_adjusted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
    plt.xlabel("Actual Duration (minutes)")
    plt.ylabel("Adjusted Predicted Duration (minutes)")
    plt.title("Bias-Corrected GA Prediction vs. Actual")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    from sklearn.metrics import mean_squared_error

    mse_corrected = mean_squared_error(y, y_pred_adjusted)
    rmse_corrected = np.sqrt(mse_corrected)
    print(f"[GA + Bias Adjust] RMSE on all data = {rmse_corrected:.4f}")