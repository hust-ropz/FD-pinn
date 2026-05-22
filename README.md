# FD-PINN 复现

当前仓库从论文第一个算例开始复现：

1. 一维热传导方程的频域参考解与频率截断基线。
2. 一维热传导方程的 `PINN` 与 `FD-PINN` 训练过程。
3. 速度势方程与一维 `Burgers` 方程算例。

## 当前进度

第一步先固定热传导算例的数值设置，避免后续训练阶段混入问题定义误差：

- 求解域：`x in [-20, 20)`，`t in [0, 5]`
- 空间网格数：`256`
- 初值：`u(x, 0) = 2 sech(x)`
- 方程参数：`a = 1`
- 论文给出的频率截断信息：`kt = 8`，保留 `50` 个非负频率成分

运行频域参考脚本：

```powershell
python -m scripts.heat_equation_reference
```

运行当前验证：

```powershell
python -m unittest discover -s tests
```

## 说明

当前环境已可使用 `numpy`，尚未检测到 `torch`。因此第一步只实现热传导方程的频域参考解、初值验证和截断基线；后续训练代码需要在可用的 `PyTorch` 环境中继续接入。
