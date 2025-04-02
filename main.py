import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_data
from genetic_algorithm import genetic_algorithm

# 加载数据
X, y = load_data()
if X is None or y is None:
    print("数据加载失败，退出程序。")
    exit()
y = y/60

# 运行遗传算法并保存最佳权重
best_weights, best_fitness_per_gen = genetic_algorithm(X, y, save_path="best_weights.csv")

print("\nBest Weights Found:")
print(best_weights)

# 预测和实际对比
predictions = X @ best_weights
print("\nSample Predictions vs Actual:")
y = y.flatten()
for pred, actual in zip(predictions[:5], y[:5]):
    print(f"Predicted: {pred:.2f} h, Actual: {actual:.2f} h")

# 保存 fitness 曲线
pd.DataFrame(best_fitness_per_gen, columns=["Fitness"]).to_csv("fitness_history.csv", index_label="Generation")

# 绘图：适应度随代数变化
plt.figure(figsize=(10, 5))
plt.plot(best_fitness_per_gen, label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness Score (1/MSE)")
plt.title("Genetic Algorithm Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘图：预测 vs 实际
plt.figure(figsize=(6, 6))
plt.scatter(y, predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Total Time (h)")
plt.ylabel("Predicted Total Time (h)")
plt.title("Prediction vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()