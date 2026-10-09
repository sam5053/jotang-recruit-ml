# Jotang ML Task 1

## 1. 原始数据可视化与数据集划分

这一部分的实现主要在 `src/data.py`。目标是先看清楚原始数据长什么样，再把数据严格划分成训练集、验证集和测试集，保证模型开发过程不会把测试信息提前用掉。

### 第一步：生成原始数据

代码使用 `sklearn.datasets.make_moons` 生成一个二维二分类数据集：

```python
X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=seed,
)
```

- `X` 的形状是 `(1000, 2)`，表示 1000 个样本，每个样本有 2 个特征。
- `y` 的形状是 `(1000,)`，表示每个样本的类别标签，取值为 `0` 或 `1`。
- `noise=0.2` 会给数据加入噪声，让两个类别在边界附近有一些重叠，更接近真实分类问题。
- `random_state=seed` 固定随机过程，保证每次运行得到相同的数据，便于复现实验。

### 第二步：可视化原始数据

`plot_raw_data(X, y)` 会把所有样本画成二维散点图：

```python
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    s=25,
    edgecolors="black",
    linewidths=0.2,
)
```

图中横轴是 `Feature 1`，纵轴是 `Feature 2`，颜色表示类别。输出文件是：

```text
outputs/figures/raw_data.png
```

从图里可以看到数据是两个交错的月牙形分布，不是简单直线就能分开的，所以后面使用一个小型 MLP 来学习非线性分类边界是合理的。

### 第三步：划分训练集、验证集、测试集

划分分两次完成。

第一次先切出训练集：

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=seed,
    stratify=y,
)
```

这里 `test_size=0.3` 表示把 30% 数据临时留出来，所以训练集占 70%：

- 训练集：700 个样本
- 临时集：300 个样本

第二次把临时集平均分成验证集和测试集：

```python
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=seed,
    stratify=y_temp,
)
```

因此最终比例是：

- 训练集：70%，700 个样本
- 验证集：15%，150 个样本
- 测试集：15%，150 个样本

两次划分都使用了 `stratify`，作用是尽量保持每个子集中类别 0 和类别 1 的比例一致，避免某个集合类别分布明显偏掉。

### 第四步：可视化三个子集

`plot_data_splits(...)` 会把训练集、验证集、测试集分别画在三个子图里，输出文件是：

```text
outputs/figures/data_split.png
```

这样可以直观看到三个子集是否仍然覆盖了原始数据的大致分布。如果验证集或测试集只落在某个很窄的区域，评估结果就可能不稳定。

## 三个数据集的职责

训练集只用于模型学习参数。`src/train.py` 里训练时只对 `train_loader` 调用反向传播和 `optimizer.step()`，模型真正从这部分数据中更新权重。

验证集用于训练过程中选择模型和调参。当前代码每个 epoch 后在验证集上计算 `val_loss` 和 `val_accuracy`，并保存验证损失最低的模型权重。验证集参与模型选择，但不参与梯度更新。

测试集只用于最后一次最终评估。当前代码在训练结束、选出最佳验证模型之后，才用测试集计算 `test_loss` 和 `test_accuracy`。测试集不应该用于调学习率、改模型结构或选择 epoch，否则测试结果会变成间接参与训练的结果。

## 如何避免数据泄漏

本项目避免数据泄漏的关键点是：

- 先划分数据，再分别构造 `train_loader`、`val_loader` 和 `test_loader`。
- 训练阶段只有训练集会执行 `loss.backward()` 和 `optimizer.step()`。
- 验证集只用于选择最佳模型，不更新参数。
- 测试集只在训练完成后用于最终评估。
- 使用固定 `seed` 保证划分可复现，避免每次运行得到不同结果。
- 使用 `stratify` 保持类别比例，让三个子集的类别分布更一致。

如果后续加入标准化、特征选择或数据增强，也要注意：只能在训练集上 `fit`，再把同一个变换应用到验证集和测试集，不能用全量数据一起计算均值、方差或其他统计量。
