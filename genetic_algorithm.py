import numpy as np
import pandas as pd
from population import Population


def genetic_algorithm(X, y, population_size=50, generations=100, save_path="best_weights.csv"):
    """
    遗传算法主函数：优化线性模型权重以最小化预测误差。

    参数：
        X : np.ndarray, 特征矩阵
        y : np.ndarray, 目标变量（以小时为单位）
        population_size : int, 每一代种群个体数量
        generations : int, 迭代代数
        save_path : str, 保存最佳权重的 CSV 文件路径

    返回：
        best_weights : np.ndarray, 最优权重
        best_fitness_per_gen : list[float], 每一代的最佳适应度值
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

    # 保存最佳权重
    weights_df = pd.DataFrame(best_weights, columns=["Weight"])
    weights_df.index.name = "Feature_Index"
    weights_df.to_csv(save_path)
    print(f"Best weights saved to {save_path}")

    return best_weights, best_fitness_per_gen