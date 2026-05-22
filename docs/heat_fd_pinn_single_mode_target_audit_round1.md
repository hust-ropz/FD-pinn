# Heat FD-PINN Single Mode Target Audit Round 1

## 诊断目的

前面的低频对照已经排除了 joint/sequential 更新顺序这一层差异。本轮只读核查正式实验输出，目标是确认热传导单 mode 的闭式频域目标是否与 reference baseline 一致，并拆开 mode `2` 到 mode `5` 的初值误差、衰减率误差和末态幅值误差。

运行命令：

```powershell
uv run python -m scripts.audit_heat_single_mode_targets --input-dir outputs/heat_fd_pinn_m50_s500 --output-dir outputs/heat_fd_pinn_single_mode_target_audit --modes 50
```

输出文件：

- `outputs/heat_fd_pinn_single_mode_target_audit/single_mode_target_diagnostics.csv`
- `outputs/heat_fd_pinn_single_mode_target_audit/summary.json`

## 闭式目标与 reference

审计脚本以 `u_ref` 的 `t = 0` rFFT 频谱为初值，按热传导单 mode 闭式解推进：

```text
u_hat_m(t) = u_hat_m(0) * exp(-k_m^2 * t)
```

闭式频谱和 `u_ref` 的总相对误差为：

| Metric | Value |
| --- | --- |
| `exact_vs_reference_total_relative_error` | `1.669115521227978e-16` |

mode `2` 到 mode `5` 的 `exact_vs_reference_relative_error_by_mode` 都在 `1e-16` 量级。结论很直接：当前 reference baseline 与训练脚本使用的热方程频域 ODE 目标是一致的，不支持“target 定义和 reference 本身不一致”这个怀疑。

## mode 2 到 mode 5

| Mode | Theory `k^2` | Initial complex error | Fitted decay rate | Final amplitude ratio `pred/ref` |
| --- | --- | --- | --- | --- |
| `2` | `0.09869604401089357` | `0.08862849198973761` | `3.5989331673397515` | `3.143102681296083e-07` |
| `3` | `0.2220660990245106` | `0.20668558793175362` | `2.2217683411521385` | `3.853772422362117e-06` |
| `4` | `0.3947841760435743` | `3.548827110863633` | `2.7450335978173916e-18` | `6.228159932058819` |
| `5` | `0.6168502750680849` | `5.950734967445513` | `1.191614011986646e-16` | `15.828348364144922` |

这四个低频 mode 不是同一种失败模式。

### mode 2 与 mode 3

mode `2` 和 mode `3` 的初值并没有严重跑偏：

- mode `2` 初值频谱误差约 `0.0886`，参考初值幅值约 `35.77`
- mode `3` 初值频谱误差约 `0.2067`，参考初值幅值约 `31.25`

但它们学到的衰减速度远大于理论：

- mode `2` 理论 `k^2` 约 `0.0987`，拟合衰减率约 `3.5989`
- mode `3` 理论 `k^2` 约 `0.2221`，拟合衰减率约 `2.2218`

末态因此几乎塌到零：

- mode `2` 末态参考幅值约 `21.8364`，预测约 `6.86e-06`
- mode `3` 末态参考幅值约 `10.2961`，预测约 `3.97e-05`

mode `2` 与 mode `3` 的主要问题是预测衰减率错得过快，末态幅值塌缩；初值小偏差不是主因。

### mode 4 与 mode 5

mode `4` 和 mode `5` 在初值就已经有明显偏差：

- mode `4` 初值频谱误差约 `3.5488`
- mode `5` 初值频谱误差约 `5.9507`

更关键的是，它们几乎没有学到理论衰减：

- mode `4` 理论 `k^2` 约 `0.3948`，拟合衰减率约 `0`
- mode `5` 理论 `k^2` 约 `0.6169`，拟合衰减率约 `0`

末态因此被放大保留：

- mode `4` 末态参考幅值约 `3.6560`，预测约 `22.7701`
- mode `5` 末态参考幅值约 `0.9880`，预测约 `15.6391`

mode `4` 与 mode `5` 的末态误差来自初值错和衰减几乎没学到叠加，不是单纯末态重构误差。

## 对照 mode 6 与 mode 7

相邻的 mode `6` 与 mode `7` 说明网络和闭式目标并不是对所有低频 mode 都失败：

| Mode | Theory `k^2` | Fitted decay rate | Final amplitude ratio `pred/ref` |
| --- | --- | --- | --- |
| `6` | `0.8882643960980424` | `0.8883019904635612` | `1.0005324083361025` |
| `7` | `1.2090265391334463` | `1.207467435950163` | `0.9655517075608072` |

这两个 mode 的初值误差和末态误差都很小，预测衰减率也贴近理论。由此看，当前异常不是统一的频谱系数尺度错误，也不是单一的 rFFT 闭式目标实现错误。

## 当前判断

`summary.json` 的结论 flags 为：

- `reference_matches_closed_form = true`
- `low_modes_have_initial_mismatch = true`
- `low_modes_have_wrong_decay_rate = true`
- `low_modes_have_final_amplitude_error = true`

基于这些结果，当前更像是 **若干单 mode 的训练目标没有被优化到正确轨迹**：

1. target 闭式定义本身与 reference 一致，先排除 reference/ODE target 不一致。
2. mode `2` 和 mode `3` 初值已基本接近，但衰减率严重过快，说明单 mode 时间演化约束没有被正确学住。
3. mode `4` 和 mode `5` 既有初值偏差，又近似保持常幅值，说明初值项与残差项都没有把这些 mode 拉回闭式轨迹。
4. mode `6` 和 mode `7` 又能很好拟合，暂时不支持“网络表达力对所有 mode 都不够”这个更强判断。

所以当前优先怀疑的是 loss 尺度与优化不足在特定 mode 上的作用，尤其是单 mode 初始幅值和残差约束如何共同塑造 mode `2` 到 mode `5` 的优化路径；还没有证据把问题归因为闭式 target 定义错误。

## 下一步建议

下一步建议继续保持单 mode 层面的核查：

1. 对 mode `2` 到 mode `5` 单独导出训练残差和闭式轨迹点对点差异，确认残差低但轨迹错，还是残差本身没有压住。
2. 针对 mode `2` 到 mode `5` 直接做不改网络结构的单 mode 受控训练诊断，和解析闭式轨迹逐 step 对比。
3. 在确认单 mode loss 尺度问题前，不继续扩展 sequential schedule，也不把问题改写成 reference baseline 或 rFFT 闭式目标错误。
