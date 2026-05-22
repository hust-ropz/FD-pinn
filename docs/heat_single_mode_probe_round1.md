# Heat Single Mode Probe Round 1

## 诊断目的

前一轮只读审计已经确认热传导单 mode 的闭式目标与 reference baseline 一致。本轮新增 isolated probe，不修改主训练脚本，只把 mode `2` 到 mode `7` 拿出来各自训练，检查 mode `2` 到 mode `5` 的异常是否还会在单 mode 独立训练循环里出现。

probe 保持当前 `FD-PINN` 单 mode 口径：

- 一个 mode 一个 `ModeNetwork`
- 输入为 `t`，输出为频谱系数的 real/imag
- 网络为 `3` 个 hidden layers，每层 `5` 个 neurons，激活为 `tanh`
- optimizer 为 `L-BFGS`，`max_iter = 3`
- loss 为 `initial_loss + residual_loss`
- 评估目标为 `u_hat_m(t) = u_hat_m(0) * exp(-k_m^2 * t)`

为了让 probe 与原始正式实验的同 mode 对比更直接，脚本在每个 isolated mode 中复现主训练脚本同一 `seed` 下该 mode 对应的初始化序列位置和 residual samples；训练循环、optimizer 和输出诊断仍然逐 mode 独立。

## 运行命令

smoke：

```powershell
uv run python -m scripts.probe_heat_single_mode_training --modes 2 --steps 2 --num-residual 5 --output-dir outputs/heat_single_mode_probe_smoke
```

正式 probe：

```powershell
uv run python -m scripts.probe_heat_single_mode_training --modes 2,3,4,5,6,7 --steps 500 --num-residual 200 --output-dir outputs/heat_single_mode_probe_m2_7
```

输出文件：

- `outputs/heat_single_mode_probe_m2_7/probe_training_history.csv`
- `outputs/heat_single_mode_probe_m2_7/probe_mode_summary.csv`
- `outputs/heat_single_mode_probe_m2_7/probe_summary.json`

## Probe 与 main 对比

`probe_summary.json` 读取到了原始正式实验输出：

- `main_output_available = true`
- `isolated_probe_matches_main_failure_pattern = true`
- `likely_main_loop_issue = false`
- `likely_single_mode_optimization_issue = true`

关键 mode 指标如下：

| Mode | Probe relative complex error | Theory `k^2` | Main fitted decay | Probe fitted decay | Main final complex error | Probe final complex error |
| --- | --- | --- | --- | --- | --- | --- |
| `2` | `0.7945151843183488` | `0.09869604401089357` | `3.5989331673397515` | `3.598933167343178` | `21.836365343275585` | `21.836365343275585` |
| `3` | `0.8175314035407516` | `0.2220660990245106` | `2.2217683411521385` | `2.2217683411469884` | `10.296024145952282` | `10.29602414595228` |
| `4` | `0.9826681417060319` | `0.3947841760435743` | `2.7450335978173916e-18` | `2.7450335978173916e-18` | `19.114125969698964` | `19.11412596969896` |
| `5` | `1.2017908149751428` | `0.6168502750680849` | `1.191614011986646e-16` | `1.191614011986646e-16` | `14.651033391045406` | `14.651033391045406` |
| `6` | `0.0005104875978084161` | `0.8882643960980424` | `0.8883019904635612` | `0.8883019904635612` | `0.00011333593680826609` | `0.00011333593680818526` |
| `7` | `0.0012518798441639792` | `1.2090265391334463` | `1.207467435950163` | `1.2074674359501638` | `0.0011901268356986998` | `0.0011901268356993666` |

这不是只在聚合指标上相似。mode `2` 到 mode `7` 的 initial error、final error 和拟合衰减率在 main 与 isolated probe 中都基本逐项重合。

## mode 2 到 mode 5

isolated probe 中 mode `2` 到 mode `5` 仍然失败：

- mode `2` 最终 `relative_complex_error_over_time = 0.7945151843183488`
- mode `3` 最终 `relative_complex_error_over_time = 0.8175314035407516`
- mode `4` 最终 `relative_complex_error_over_time = 0.9826681417060319`
- mode `5` 最终 `relative_complex_error_over_time = 1.2017908149751428`

轨迹也保留了前一轮审计看到的两类失败：

1. mode `2` 和 mode `3` 在 `500` step 后初值幅值已经贴近 reference，但拟合衰减率仍远快于理论，末态幅值塌到接近零。
2. mode `4` 和 mode `5` 的 final loss 仍然很高，拟合衰减率接近零，末态幅值被过量保留。

从 history 的首末行看，mode `2` 和 mode `3` 的 `initial_loss` 已经被压低到 `0.0039275` 和 `0.0213595`，但 `residual_loss` 仍停在 `1.5810541` 和 `3.2082474`；mode `4` 和 mode `5` 的 final residual 仍为 `40.4036080` 和 `46.5319963`。这说明本轮关注的异常并不是主训练输出重构后才出现。

## mode 6 与 mode 7

isolated probe 中 mode `6` 和 mode `7` 仍然成功：

| Mode | Final loss | Relative complex error | Theory `k^2` | Probe fitted decay |
| --- | --- | --- | --- | --- |
| `6` | `1.227776762578406e-06` | `0.0005104875978084161` | `0.8882643960980424` | `0.8883019904635612` |
| `7` | `7.0944250628349135e-06` | `0.0012518798441639792` | `1.2090265391334463` | `1.2074674359501638` |

因此 probe 没有把所有 selected modes 都推坏。相邻 mode `6` 和 mode `7` 仍能在同一网络宽度、同一 optimizer 配置和同一 step 数下学到正确衰减。

## 当前判断

本轮结论是明确的：

1. isolated probe 中 mode `2` 到 mode `5` 仍然失败。
2. isolated probe 中 mode `6` 和 mode `7` 仍然成功。
3. probe 与原始正式实验中同 mode 的初值误差、末态误差和衰减率基本一致。
4. 这不支持把当前异常优先归因于主训练脚本里的 mode loop、closure 或聚合重构。

因为 probe 刻意把 optimizer 和训练循环逐 mode 隔离，而失败模式仍与 main 输出一致，当前问题更像是 **mode `2` 到 mode `5` 的单 mode 优化轨迹本身停在了错误解附近**。在现有证据下，优先级应放在单 mode loss/trajectory 诊断，而不是继续扩展 schedule 或转向经典 `PINN`。

## 下一步建议

下一步建议继续保持诊断性质：

1. 针对 mode `2` 到 mode `5` 比较每 step residual 与闭式轨迹点误差，拆开“残差没有降下来”和“残差低但轨迹仍错”这两种情况。
2. 用闭式解生成同 residual points 的解析 residual 基准，核查当前 residual loss 数值尺度对不同初始幅值的 mode 是否不均衡。
3. 在确认单 mode 优化轨迹问题前，不继续做权重扫描、不扩展 `sequential` 到 `modes = 50`，也不把问题切换成主 loop 修复。
