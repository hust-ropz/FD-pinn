# Heat FD-PINN 初值权重实验 Round 1

## 实验目的

上一轮只读诊断显示，原始热传导 `FD-PINN` 正式实验的误差主要集中在低频 mode `2` 到 mode `5`，且初值损失与残差损失都没有压到很低。本轮只做一个最小训练修复实验：给训练脚本加入 loss 权重，并只运行一组 `initial_weight = 10`、`residual_weight = 1` 的正式实验。

训练脚本中当前总损失定义为：

```text
loss = initial_weight * initial_loss + residual_weight * residual_loss
```

`training_history.csv` 中的 `loss` 与 `weighted_loss` 都表示优化器实际使用的加权总损失；`initial_loss` 与 `residual_loss` 保持未加权，便于和原始实验比较。

## 运行命令

正式加权实验：

```powershell
uv run python -m scripts.train_heat_fd_pinn --modes 50 --steps 500 --num-residual 200 --initial-weight 10 --residual-weight 1 --output-dir outputs/heat_fd_pinn_m50_s500_icw10
```

对应诊断：

```powershell
uv run python -m scripts.audit_heat_fd_pinn_result --input-dir outputs/heat_fd_pinn_m50_s500_icw10 --output-dir outputs/heat_fd_pinn_m50_s500_icw10_audit
```

## 总体指标比较

| Metric | Original | `initial_weight = 10` |
| --- | --- | --- |
| `relative_l2_error` | `0.539048283904845` | `0.636465157435483` |
| `relative_l2_error_at_t0` | `0.10213279624431647` | `0.39680602925330694` |
| `relative_l2_error_at_t_final` | `0.6751864182885009` | `0.7615522680694791` |
| `final_loss` | `2.5788339174081027` | `11787.883221034219` |
| `final_initial_loss` | `0.6836130951941308` | `10.318972505490926` |
| `final_residual_loss` | `1.8952208222139717` | `11684.69349597931` |

这组 `initial_weight = 10` 实验没有改善当前结果：

- `t = 0` 误差没有下降，反而从约 `0.1021` 升到约 `0.3968`。
- `t = 5` 误差没有下降，反而从约 `0.6752` 升到约 `0.7616`。
- 未加权 `final_initial_loss` 没有下降，反而从约 `0.6836` 升到约 `10.3190`。
- 未加权 `final_residual_loss` 明显恶化，从约 `1.8952` 升到约 `11684.6935`。

## 训练过程观察

加权实验不是从头到尾都停在高损失。根据 `training_history.csv`：

- 第 `166` 步的最优加权损失为 `2.068371934364431`。
- 当时未加权 `initial_loss` 为 `0.055328397897403396`，未加权 `residual_loss` 为 `1.515087955390397`。
- 第 `169` 步总损失已跳到 `68.82644918887767`。
- 第 `170` 步总损失跳到 `3300989.288801206`。
- 第 `171` 步进一步失稳到 `2.641893241317648e+21`。

最终落盘指标来自第 `500` 步，因此这次正式加权结果反映的是一次后期失稳后的最终状态，而不是第 `166` 步附近的暂时低损失状态。

## 频域误差比较

原始实验的总频域误差贡献中，mode `2` 到 mode `5` 合计约为 `98.62%`。加权实验中，这四个 mode 的合计贡献降到约 `31.19%`。

这个占比下降不能解读为低频误差问题已经修好，因为新实验的总误差更高，且误差主导模式发生了转移：

| Mode | `initial_weight = 10` contribution |
| --- | --- |
| `37` | `0.6860603721457932` |
| `2` | `0.29529807290632537` |
| `5` | `0.01662209189838705` |
| `12` | `0.0020185025471086224` |

新实验中 mode `37` 的平均参考幅值只有约 `1.06e-04`，平均预测幅值却达到 `32.0`，单独贡献约 `68.61%` 的总频域误差。`t = 0` 诊断里 mode `37` 也贡献约 `99.23%` 的频谱误差。也就是说，这组权重实验把原先低频主导的误差，变成了高频 mode `37` 的明显失稳误差。

## 结论

本轮 `initial_weight = 10` 受控实验没有达到预期：

1. 没有改善 `t = 0` 初值误差。
2. 没有改善 `t = 5` 末态误差。
3. mode `2` 到 mode `5` 的相对贡献虽然下降，但原因是 mode `37` 失稳后占据了主要误差，不是整体误差下降。
4. `initial_loss` 与 `residual_loss` 的最终值都比原始实验更差。

因此，当前不值得直接自动展开更多权重扫描。更合理的下一步是先围绕这组失稳现象做最小诊断：确认加权训练为何在第 `170` 步附近爆开、mode `37` 为何在最终重构中出现大幅预测幅值，再决定是否需要受控地调整权重策略或优化过程。
