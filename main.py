import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from data_loader import load_data
from genetic_algorithm import genetic_algorithm

from sklearn.linear_model import LinearRegression

def generate_nonlinear_features(X):
    # Add nonlinear features: square, log, and square root
    X_squared = X ** 2
    X_log = np.log1p(np.abs(X))
    X_sqrt = np.sqrt(np.abs(X))
    return np.hstack([X, X_squared, X_log, X_sqrt])

if __name__ == "__main__":
    X, y = load_data()
    if X is None or y is None:
        print("Data loading failed. Exiting.")
        exit()

    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

    # Add nonlinear features
    X = generate_nonlinear_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_all_scaled = scaler_X.transform(X)  # Normalize all data

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

    # Plot GA fitness over generations
    plt.figure()
    plt.plot(best_fitness_per_gen, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score (-log(MSE))")
    plt.title("GA Training Fitness Curve")
    plt.grid(True)
    plt.legend()
    plt.show()


    # GA predictions on all data
    y_pred_all_ga = X_all_scaled @ best_weights

    # Calibrate GA prediction (fit bias to match y)
    bias_adjust = LinearRegression()
    bias_adjust.fit(y_pred_all_ga.reshape(-1, 1), y)
    y_pred_adjusted = bias_adjust.predict(y_pred_all_ga.reshape(-1, 1))

    # Plot: Bias-Corrected GA vs Actual + Linear Regression
    plt.figure()
    plt.scatter(y, y_pred_adjusted, alpha=0.6, s=20, label="GA + Bias Adjust")
    min_val = min(np.min(y), np.min(y_pred_adjusted))
    max_val = max(np.max(y), np.max(y_pred_adjusted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
    plt.xlabel("Actual Duration (minutes)")
    plt.ylabel("Predicted Duration (minutes)")
    plt.title("Bias-Corrected GA vs. Actual")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Output RMSE and mean bias after bias adjustment
    mse_adjusted = mean_squared_error(y, y_pred_adjusted)
    rmse_adjusted = np.sqrt(mse_adjusted)
    print(f"[GA + Bias Adjust] RMSE on all data = {rmse_adjusted:.4f}")
    mean_bias = np.mean(y_pred_adjusted - y)
    print(f"[GA + Bias Adjust] Mean Bias on all data = {mean_bias:.4f} minutes")
    max_bias = np.max(abs(y_pred_adjusted - y))
    print(f"[GA + Bias Adjust] Max Bias on all data = {max_bias:.4f} minutes")

    bias_constant = np.mean(y - y_pred_all_ga)
    y_pred_const_adjusted = y_pred_all_ga + bias_constant

    rmse_const = np.sqrt(mean_squared_error(y, y_pred_const_adjusted))
    mean_bias_const = np.mean(y_pred_const_adjusted - y)

    print(f"[GA + Constant Bias Adjust] RMSE = {rmse_const:.4f}")
    print(f"[GA + Constant Bias Adjust] Mean Bias = {mean_bias_const:.4f}")
    print(f"[GA + Constant Bias Adjust] Bias Term Added = {bias_constant:.4f}")

    plt.figure()
    plt.scatter(y, y_pred_const_adjusted, alpha=0.6, label="Constant Bias Adjust")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="y = x")
    plt.xlabel("Actual Duration")
    plt.ylabel("Predicted Duration")
    plt.title("Constant Bias Adjustment vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()