# FD-PINN 复现

当前仓库从论文第一个算例开始复现：

1. 一维热传导方程的频域参考解与频率截断基线。
2. 一维热传导方程的 `PINN` 与 `FD-PINN` 训练过程。
3. 速度势方程与一维 `Burgers` 方程算例。

## 环境

项目环境由 `uv` 管理，当前固定使用 `Python 3.11`，首阶段依赖为 `numpy` 和 `torch`。

```powershell
uv sync
```

## 当前进度

第一步先固定热传导算例的数值设置，避免后续训练阶段混入问题定义误差：

- 求解域：`x in [-20, 20)`，`t in [0, 5]`
- 空间网格数：`256`
- 初值：`u(x, 0) = 2 sech(x)`
- 方程参数：`a = 1`
- 论文给出的频率截断信息：`kt = 8`，保留 `50` 个非负频率成分

运行频域参考脚本：

```powershell
uv run python -m scripts.heat_equation_reference
```

运行当前验证：

```powershell
uv run python -m unittest discover -s tests
```

## 热传导 FD-PINN 训练

运行最小 smoke test：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 3 --steps 2 --num-residual 5 --output-dir outputs/heat_fd_pinn_smoke
```

按论文热传导 `FD-PINN` 表 1 的训练步数和模式数运行：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 50 --steps 500
```

当前环境使用 CPU 版 `PyTorch`，完整训练可能耗时较长。

正式实验结果入口：

- [Heat FD-PINN m50 s500 结果报告](docs/heat_fd_pinn_m50_s500_report.md)

对正式实验输出运行只读误差诊断：

```powershell
uv run python -m scripts.audit_heat_fd_pinn_result --input-dir outputs/heat_fd_pinn_m50_s500 --output-dir outputs/heat_fd_pinn_m50_s500_audit
```

运行加强初值约束的受控实验：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 50 --steps 500 --num-residual 200 --initial-weight 10 --residual-weight 1 --output-dir outputs/heat_fd_pinn_m50_s500_icw10
```

受控实验结果入口：

- [Heat FD-PINN 初值权重实验 Round 1](docs/heat_fd_pinn_ic_weight_experiment_round1.md)

运行 L-BFGS 稳定性 debug 诊断：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 50 --steps 200 --num-residual 200 --initial-weight 10 --residual-weight 1 --debug-log-modes --debug-log-every 5 --save-best --output-dir outputs/heat_fd_pinn_m50_s200_icw10_debug
```

稳定性诊断报告入口：

- [Heat FD-PINN L-BFGS 稳定性诊断 Round 1](docs/heat_fd_pinn_lbfgs_instability_diagnosis_round1.md)

运行低频 joint/sequential 对照：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 6 --steps 500 --num-residual 200 --training-schedule joint --output-dir outputs/heat_fd_pinn_m6_s500_joint
uv run python -m scripts.train_heat_fd_pinn --modes 6 --steps 500 --num-residual 200 --training-schedule sequential --output-dir outputs/heat_fd_pinn_m6_s500_sequential
```

低频 schedule 对照报告入口：

- [Heat FD-PINN Sequential Mode Training Round 1](docs/heat_fd_pinn_sequential_mode_training_round1.md)

运行单 mode 闭式目标诊断：

```powershell
uv run python -m scripts.audit_heat_single_mode_targets --input-dir outputs/heat_fd_pinn_m50_s500 --output-dir outputs/heat_fd_pinn_single_mode_target_audit --modes 50
```

单 mode 目标审计报告入口：

- [Heat FD-PINN Single Mode Target Audit Round 1](docs/heat_fd_pinn_single_mode_target_audit_round1.md)

运行 isolated single-mode 训练 probe：

```powershell
uv run python -m scripts.probe_heat_single_mode_training --modes 2,3,4,5,6,7 --steps 500 --num-residual 200 --output-dir outputs/heat_single_mode_probe_m2_7
```

单 mode probe 报告入口：

- [Heat Single Mode Probe Round 1](docs/heat_single_mode_probe_round1.md)

运行 multi-seed isolated single-mode probe：

```powershell
uv run python -m scripts.probe_heat_single_mode_training --modes 2,3,4,5,6,7 --seeds 0,1,2,3,4 --steps 500 --num-residual 200 --output-dir outputs/heat_single_mode_probe_m2_7_multiseed
```

multi-seed probe 报告入口：

- [Heat Single Mode Multi-Seed Probe Round 1](docs/heat_single_mode_multiseed_probe_round1.md)

运行独立 multi-seed single-mode probe 诊断：

```powershell
uv run python -m scripts.probe_heat_single_mode_multiseed --modes 2 3 4 5 6 7 --seeds 0 1 2 3 4 --output-dir outputs/heat_single_mode_multiseed_probe_round1
```

独立 multi-seed probe 输出入口：

- `outputs/heat_single_mode_multiseed_probe_round1/report.md`
- [Heat Single Mode Multi-Seed Probe Round 1](docs/heat_single_mode_multiseed_probe_round1.md)

## 说明

当前实现已包含热传导方程的频域参考基线和最小 `FD-PINN` 训练闭环；经典 `PINN` 对照训练仍留在后续步骤。
