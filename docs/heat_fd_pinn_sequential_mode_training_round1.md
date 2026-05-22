# Heat FD-PINN Sequential Mode Training Round 1

## 实验目的

论文对线性问题描述为各频率网络可独立、依次训练。当前热传导 `FD-PINN` 已经使用每个 mode 一个小网络、每个 mode 一个独立 `L-BFGS` optimizer，但原始 `joint` schedule 会在每个 outer step 内依次更新所有 mode。本轮加入 `sequential` schedule，只做 `modes = 6` 的低频对照，检查把每个 mode 完整训练完再进入下一个 mode 是否能改善 mode `0` 到 mode `5`。

## 运行命令

低频 joint 对照：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 6 --steps 500 --num-residual 200 --training-schedule joint --output-dir outputs/heat_fd_pinn_m6_s500_joint
```

低频 sequential 对照：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 6 --steps 500 --num-residual 200 --training-schedule sequential --output-dir outputs/heat_fd_pinn_m6_s500_sequential
```

对应审计：

```powershell
uv run python -m scripts.audit_heat_fd_pinn_result --input-dir outputs/heat_fd_pinn_m6_s500_joint --output-dir outputs/heat_fd_pinn_m6_s500_joint_audit
uv run python -m scripts.audit_heat_fd_pinn_result --input-dir outputs/heat_fd_pinn_m6_s500_sequential --output-dir outputs/heat_fd_pinn_m6_s500_sequential_audit
```

## 指标比较

| Metric | `m6 joint` | `m6 sequential` |
| --- | --- | --- |
| `relative_l2_error` | `0.5541396070743727` | `0.5541396070743727` |
| `relative_l2_error_at_t0` | `0.3613688954286689` | `0.3613688954286689` |
| `relative_l2_error_at_t_final` | `0.675197601828321` | `0.675197601828321` |
| `final_loss` | `19.36934343590848` | `19.36934343590848` |
| `final_initial_loss` | `4.004713171767382` | `4.004713171767382` |
| `final_residual_loss` | `15.364630264141098` | `15.364630264141098` |

这两组低频实验在最终指标上完全一致。`sequential` 没有降低全局相对误差，也没有改善 `t = 0` 或 `t = 5` 的相对误差。

## mode 2 到 mode 5 频域误差

审计结果中，mode `2` 到 mode `5` 的总频域误差贡献也相同：

| Mode | `m6 joint` contribution | `m6 sequential` contribution |
| --- | --- | --- |
| `2` | `0.45403215379220835` | `0.45403215379220835` |
| `3` | `0.2324971108925615` | `0.2324971108925615` |
| `4` | `0.14840749832363653` | `0.14840749832363653` |
| `5` | `0.09828730405575309` | `0.09828730405575309` |

mode `2` 到 mode `5` 的贡献合计都是 `0.9332240670641594`。对应的 `mean_abs_spectral_error` 也逐项一致：

- mode `2`: `19.82867374263003`
- mode `3`: `14.692534455127483`
- mode `4`: `11.541698191551781`
- mode `5`: `9.556578920103469`

因此，`sequential` 没有改善本轮关注的低频 mode 误差。

## 训练历史含义

新 schedule 开关改变了 history 的记录方式：

- `joint` 的 `training_history.csv` 有 `500` 行，`active_mode = -1`，每行记录一个聚合 outer step。
- `sequential` 的 `training_history.csv` 有 `3000` 行，`active_mode` 从 `0` 到 `5`，每个 mode 各记录 `500` 个 step。

但对热传导线性频域方程而言，当前实现中每个 mode 本来就有独立网络、独立 optimizer、独立 loss；不同 mode 之间没有共享参数，也没有互相耦合的梯度。因此在相同初始化、相同 residual samples 和相同每 mode step 数下，joint 中的“mode 间交替推进”与 sequential 中的“一个 mode 训练完再下一个”得到相同终点，是符合当前实现结构的结果。

## 结论

本轮对照结论是明确的：

1. `sequential` 没有改善 `modes = 6` 的总误差。
2. `sequential` 没有改善 `t = 0` 初值误差。
3. `sequential` 没有改善 `t = 5` 末态误差。
4. `sequential` 没有降低 mode `2` 到 mode `5` 的频域误差。
5. `final_initial_loss` 与 `final_residual_loss` 也没有下降。

因此，当前不值得把 `sequential` 直接扩展到 `modes = 50` 做新一轮正式实验。当前 joint 训练并没有把不同 mode 混到同一个共享参数优化里；它已经是独立 mode 网络的 interleaved update。仅调整训练顺序不能解释低频 mode `2` 到 mode `5` 的偏差。

下一步更应转向单 mode 目标核查：

1. 先用解析闭式解检查单 mode 的目标、初值幅值和指数衰减轨迹是否与当前 loss 实现严格对齐。
2. 对 mode `2` 到 mode `5` 做独立单 mode 诊断，确认每个 mode 在固定 `L-BFGS` 设置下为何停在当前误差。
3. 在单 mode 目标核查完成前，不把 schedule 扩展到 `modes = 50`。
