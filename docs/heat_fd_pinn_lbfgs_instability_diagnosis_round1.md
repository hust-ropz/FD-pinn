# Heat FD-PINN L-BFGS 稳定性诊断 Round 1

## 诊断目标

上一轮 `initial_weight = 10` 的 `500` 步正式实验在第 `170` 步附近出现后期失稳，最终 mode `37` 成为主要误差源。本轮不修复优化器，只给训练脚本增加 opt-in debug logging，并用同一权重设置跑到 `200` 步，记录逐 mode 的损失与频谱幅值。

运行命令：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 50 --steps 200 --num-residual 200 --initial-weight 10 --residual-weight 1 --debug-log-modes --debug-log-every 5 --save-best --output-dir outputs/heat_fd_pinn_m50_s200_icw10_debug
```

本次 debug 输出包括：

- `training_history.csv`
- `mode_training_diagnostics.csv`
- `instability_events.csv`
- `prediction.npz`
- `metrics.json`
- `best_prediction.npz`
- `best_metrics.json`

## 是否在 200 步内复现失稳

失稳在 `200` 步内复现了，而且不是单次事件。

`instability_events.csv` 以 `current_loss > 100 * previous_loss` 或 loss 非有限为突变标准。首次突变发生在第 `3` 步：

| Step | Previous loss | Current loss | Growth ratio |
| --- | --- | --- | --- |
| `3` | `635.005405117079` | `60364204.89743647` | `95060.93083775693` |

在此前后的最大逐 mode 加权损失也发生了切换：

| Moment | Max mode | `weighted_loss_by_mode` |
| --- | --- | --- |
| 第 `2` 步 | `5` | `29168.346798305236` |
| 第 `3` 步 | `12` | `3018209135.671528` |

与上一轮关注的晚期窗口对应，本次第 `170` 步也再次发生突变：

| Step | Previous loss | Current loss | Growth ratio |
| --- | --- | --- | --- |
| `170` | `68.82644918887767` | `3300989.288801206` | `47961.05752517368` |

第 `170` 步突变前后最大 mode 都是 mode `33`：

| Moment | Max mode | `weighted_loss_by_mode` |
| --- | --- | --- |
| 第 `169` 步 | `33` | `3337.9048028268535` |
| 第 `170` 步 | `33` | `165048679.8143575` |

随后第 `171`、`172`、`175`、`178`、`185` 步还记录到额外突变。第 `195` 步后训练损失又回到个位数，最终第 `200` 步 loss 为 `2.5364970388671795`。因此本次 run 不是“最终 loss 爆掉就不回头”，而是一次会在 outer step 中剧烈失稳又回落的训练过程。

## mode 37 是否提前异常

在本次 `200` 步 debug run 中，没有看到 mode `37` 在第 `170` 步突变前已经异常，也没有看到它在突变后立即异常。

| Step | mode `37` weighted loss | mode `37` max predicted amplitude |
| --- | --- | --- |
| `160` | `5.2165205994474705e-05` | `0.0073801426023106294` |
| `165` | `4.5993538915443824e-05` | `0.007692333858880921` |
| `170` | `3.9964550617373215e-05` | `0.007855614156152978` |
| `175` | `3.576083718120371e-05` | `0.008040827841733053` |
| `200` | `2.9675715916474083e-05` | `0.007959357879265518` |

相反，第 `170` 步附近的主导异常来自 mode `33`；第 `178` 步与第 `185` 步的突变记录又转为 mode `5` 主导。本次 `200` 步日志只能说明：mode `37` 不是这段已复现失稳窗口的先导 mode。上一轮 `500` 步最终结果中 mode `37` 的异常，若要定位出现时点，需要在后续单独把同类 debug 日志延伸到更长训练区间。

## best snapshot 与 final snapshot

`--save-best` 保留了本次 debug run 中加权训练损失最低的预测：

| Metric | Best | Final |
| --- | --- | --- |
| Step | `166` | `200` |
| Weighted loss | `2.068371934364431` | `2.5364970388671795` |
| Initial loss | `0.055328397897403396` | `0.057010980498218206` |
| Residual loss | `1.515087955390397` | `1.9663872338849973` |
| Relative L2 error | `0.4334753310246079` | `0.43997158025344457` |

best snapshot 的相对误差略优于 final snapshot，但差距只有约 `0.0065`，还不能称为明显改善。它说明 save-best 能避开后续震荡中更差的最终点，但这本身不是精度问题的修复。

## 当前诊断判断

当前现象更像 **L-BFGS outer step 在加权 loss 尺度下出现强烈不稳定，并表现为单个 mode 网络阶段性发散**。

理由：

1. 突变事件发生在多个 outer step，且每次事件由某个 mode 的 `weighted_loss_by_mode` 急剧放大主导。
2. 第 `170` 步事件前后由 mode `33` 控制，第 `178` 和第 `185` 步又转为 mode `5`，说明不是固定的频谱重构错误。
3. mode `37` 在这次复现窗口中保持很小，说明上一轮 final mode `37` 现象更像后续训练路径上的某次 mode 发散结果，而不是从第 `170` 步前就存在的静态频谱问题。
4. `best_prediction.npz` 与 final `prediction.npz` 使用同一重构流程，训练中 loss 可从巨大值回落；这也不支持优先把问题归因为 `FFT/irFFT` 重构。

loss 权重尺度失衡仍是背景因素：`initial_weight = 10` 把当前优化目标改成了更强的初值约束，使 L-BFGS 在若干 mode 上出现了极端 step。但仅凭本轮日志还不能区分是 `L-BFGS` 线搜索/步长轨迹本身主导，还是某些 mode 的加权目标几何特别尖锐；这需要后续继续做受控诊断，而不是直接改网络结构。

## 下一步建议

下一步先继续诊断，不直接做修复实验：

1. 若仍需定位上一轮 final mode `37` 的起点，在不改训练设置的前提下把 debug logging 延伸到 `500` 步，观察 mode `37` 的幅值首次跳变 step。
2. 针对已确认的 mode `12`、mode `33`、mode `5` 事件，比较事件前后参数更新或逐 mode loss 曲线，确认是否集中在单个 mode 的 L-BFGS step。
3. 在明确失稳来源前，不继续自动扫权重，也不把 save-best 当作正式精度修复。
