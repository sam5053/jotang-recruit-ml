- 原始数据如何生成、可视化并划分，训练集、验证集和测试集各有什么职责？如何避免数据泄漏？

  `src/data.py` 使用 `sklearn.datasets.make_moons(n_samples=1000, noise=0.2, random_state=seed)` 生成二维二分类数据。`X` 的形状是 `(1000, 2)`，每个样本有两个特征；`y` 是 0 或 1。噪声使两个月牙的边缘出现重叠，分类边界不再是简单直线。

  `plot_raw_data()` 将全部数据画成散点图，类别由颜色表示，结果保存在 `outputs/figures/raw_data.png`。`plot_data_splits()` 分别画出训练、验证、测试集，保存在 `outputs/figures/data_split.png`。

  数据先按 70%/30% 划分，再将后 30% 平分：训练集 700 个样本、验证集 150 个、测试集 150 个。两次划分都使用 `stratify`，尽量保持各集合中的类别比例。

  训练集用于计算梯度和更新模型参数；验证集用于训练过程中比较模型、选择验证损失最低的权重；测试集留到所有选择结束后评估最终泛化表现。代码只在训练集上调用 `loss.backward()` 和 `optimizer.step()`。测试集不能用于挑模型、调超参数或决定训练轮数；若加入标准化，也只能在训练集上拟合均值和方差，再将同一变换用于验证集和测试集。

- 如何用 PyTorch 搭建至少有一个隐藏层的 MLP，并完成训练、验证、测试及保存/加载？

  `src/model.py` 的基线结构为 `2 -> 32 -> 1`：两个输入特征经过 `Linear(2, 32)`、ReLU 和 `Linear(32, 1)`，输出一个 logit。`src/train.py` 将输入转成 `float32` 张量，组成 `TensorDataset` 和 `DataLoader`，按 batch 训练 100 轮；每轮结束后在验证集计算 loss 和 accuracy，并保留验证 loss 最低时的参数。训练结束后才评估测试集。

  `save_model()` 将模型权重 `state_dict` 和网络配置一起保存到 `outputs/models/best_model.pt`。`load_model()` 根据配置重建网络、载入权重并调用 `eval()`。`src/train.py` 的主程序会重新加载模型，再次计算测试指标，确认保存和加载流程可用。实验运行命令为：

  ```powershell
  conda activate jotang-ml
  python -m src.data
  python -m src.train
  python -m src.evaluate
  python -m src.experiments
  ```

- Loss、accuracy 曲线和二维决策边界是如何绘制的？

  `src/train.py` 每轮把训练集和验证集的 loss、accuracy 记录到 `history`。`plot_loss_curve()` 生成 `outputs/figures/loss_curve.png`；`plot_accuracy_curve()` 生成 `outputs/figures/accuracy_curve.png`。训练 loss 反映模型对训练数据的拟合，验证 loss 反映未参与权重更新的数据上的表现；训练和验证曲线持续分开通常是过拟合信号。

  `src/evaluate.py` 的 `plot_decision_boundary()` 在二维特征平面生成密集网格，将网格点输入模型并计算类别 1 的概率。概率为 0.5 的等高线就是分类边界，背景颜色表示预测概率，黑色叉号标出测试集错例。图片保存在 `outputs/figures/decision_boundary.png`。

- 至少改变两个因素的对照实验做了什么？指标如何变化？

  `src/experiments.py` 对隐藏层宽度和学习率分别做单变量对照。每组实验保持数据划分、随机种子、训练轮数及其他配置相同，只改变当前考察的因素。完整结果在 `outputs/tables/controlled_experiments.csv`，图表在 `outputs/figures/controlled_experiments.png`。

  隐藏层宽度为 8、32、128 时，参数量分别为 33、129、513，测试 accuracy 都是 97.33%；最佳验证 loss 分别为 0.0871、0.0919、0.0833，训练时间约为 1.27、1.22、1.28 秒。加宽显著增加参数量，但本数据集上没有提高测试 accuracy。

  学习率为 0.001、0.01、0.1 时，测试 accuracy 分别为 90.00%、97.33%、97.33%，最佳验证 loss 分别为 0.1932、0.0919、0.0839，最佳轮数分别为 100、52、9。0.001 在 100 轮内收敛较慢；0.1 在这次实验中收敛较快，但不代表较大学习率对所有数据都稳定。

  训练运行在 CPU，因此峰值 GPU 显存记为不适用。代码在 GPU 可用时通过 `torch.cuda.max_memory_allocated()` 记录显存。小数据集上的训练时间只有约 1 秒，差异很小，不能据此推断大型任务的速度差别。

- 混淆矩阵在哪里生成？错分样本如何挑选和分析？

  `src/evaluate.py` 使用测试集预测调用 `sklearn.metrics.confusion_matrix`，绘图保存在 `outputs/figures/confusion_matrix.png`。基线测试集的矩阵按“行是真实类别、列是预测类别”排列：真实 0 中 74 个预测为 0、1 个预测为 1；真实 1 中 3 个预测为 0、72 个预测为 1，总计错 4 个。

  `save_error_analysis()` 将所有错例写入 `outputs/tables/misclassified_samples.csv`，并选取最多 5 个写入 `outputs/tables/selected_errors.csv`。表格包含特征坐标、真实和预测标签、类别 1 的概率，以及预测概率到 0.5 阈值的距离。例如样本 101 的真实标签是 1、预测为 0，类别 1 概率为 0.1646；样本 62 的真实标签是 0、预测为 1，类别 1 概率为 0.9286。可以结合坐标和决策边界观察这些错误是否落在类别重叠区域，以及模型是否在错误判断时仍很自信。

- 这个任务选择了什么激活函数，为什么？

  隐藏层选择 ReLU，即 `max(0, x)`。它计算简单，正半轴梯度不会像 Sigmoid 那样容易变得很小，常用于小型 MLP。更关键的是它提供了非线性；如果删掉激活函数，多层线性层仍可合并成一个线性变换，无法表示弯月形分类边界。ReLU 的局限是神经元若长期只收到负输入，输出和梯度可能一直为零，也就是“死亡 ReLU”。

- 选择了什么损失函数，为什么？如果改成三分类呢？

  二分类模型输出一个 logit，因此使用 `nn.BCEWithLogitsLoss()`。它把 Sigmoid 和二元交叉熵合并计算，数值上比先手动做 Sigmoid 再计算交叉熵更稳定。需要概率时再对 logit 调用 `torch.sigmoid()`。

  三分类时，最后一层输出 3 个 logits，例如 `Linear(hidden_dim, 3)`，使用 `nn.CrossEntropyLoss()`。标签使用形状为 `(batch_size,)` 的整数类别编号 0、1、2；预测类别用 logits 的 `argmax(dim=1)`。`CrossEntropyLoss` 内部已经处理 softmax，因此训练前不应手动对 logits 做 softmax。

- `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()` 分别做什么？不清空梯度会怎样？

  `optimizer.zero_grad()` 清除参数中上一个 batch 留下的 `.grad`；`loss.backward()` 根据链式法则计算 loss 对模型参数的梯度；`optimizer.step()` 根据梯度和优化器规则（本项目使用 Adam）更新权重。PyTorch 默认累加梯度，如果不清零，新 batch 的梯度会叠加旧梯度，参数更新就不再只反映当前 batch。除非有意实现梯度累积，否则每次更新前应清空。

- 更深或更宽的神经网络一定更好吗？模型太简单或太复杂会怎样？

  不一定。增加层数或宽度会增加模型表示能力，也会增加参数量、计算量和过拟合风险。本实验把隐藏层从 8 加宽到 128，参数量从 33 增至 513，但测试 accuracy 都是 97.33%，说明这个简单数据集上更大的模型没有带来准确率提升。模型太简单会欠拟合，训练集和验证集表现都差，无法学到弯曲边界；模型太复杂则可能记住训练数据中的偶然噪声，训练表现很好而验证、测试表现变差。

- 学习率过大或过小会怎样？Loss 曲线能显示什么？

  学习率过小，每次权重更新幅度很小，loss 通常下降缓慢，在限定训练轮数内可能还没收敛；本实验学习率 0.001 时，100 轮后测试 accuracy 为 90%，验证 loss 仍高于其他配置。学习率过大可能越过较好的参数区域，使 loss 震荡、难以下降，极端时出现 NaN。本实验的 0.1 在当前数据上反而较快收敛，最佳轮数为 9。若训练 loss 持续下降而验证 loss 开始上升，则提示过拟合；若两者都高且下降缓慢，可能是学习不足或模型表达能力不足。

- 类别样本数量非常不均衡时模型会怎样？实验说明了什么？

  `src/experiments.py` 构造了 950 个类别 0、50 个类别 1 的月牙数据，不使用类别权重或重采样。测试集有 143 个类别 0、7 个类别 1。模型 accuracy 为 98%，但少数类 recall 只有 57.14%（7 个少数类样本只识别出 4 个），balanced accuracy 为 78.57%，少数类 F1 为 72.73%。因此只看 accuracy 会掩盖少数类漏判：模型只要偏向多数类，就可能得到很高准确率，却无法可靠识别少数类。应同时检查混淆矩阵、少数类 recall/F1、balanced accuracy 或 PR 曲线。实验图和指标分别保存在 `outputs/figures/imbalanced_data.png`、`outputs/figures/imbalance_confusion_matrix.png` 和 `outputs/tables/imbalance_metrics.json`。

- 如果故意让模型过拟合，可以怎样做？实验中训练和验证表现有什么不同，如何判断过拟合？

  `src/experiments.py` 将训练集缩小到 80 个样本，并翻转其中 24 个标签；验证集和测试集保持原样。模型改为 4 个隐藏层、每层 128 个单元，共 50,049 个参数，并训练 1600 轮，最后故意使用最后一轮权重而不恢复验证 loss 最低的权重。小数据、错误标签、大模型和长时间训练，使模型有机会记住训练样本及其错误标签。

  最终训练 accuracy 为 97.50%，验证 accuracy 为 64.67%，测试 accuracy 为 70.00%；训练 loss 为 0.0387，验证 loss 升至 3.2867。训练曲线持续改善而验证曲线明显恶化，训练与验证准确率相差 32.83 个百分点，这是过拟合的直接证据。数据和曲线分别在 `outputs/figures/overfit_data.png`、`outputs/figures/overfit_loss_curve.png`、`outputs/figures/overfit_accuracy_curve.png`，汇总指标在 `outputs/tables/overfit_metrics.json`。实际训练应增加数据、限制模型复杂度、使用正则化或早停，并根据验证集选择权重。
