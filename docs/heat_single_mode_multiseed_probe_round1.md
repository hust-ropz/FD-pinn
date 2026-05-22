# Heat Single Mode Multi-Seed Probe Round 1

## 诊断目的

上一轮 isolated probe 已经把 mode `2` 到 mode `5` 的异常定位到单 mode 训练轨迹层面。本轮不改 loss、不改 optimizer、不改网络结构，只在同一 isolated probe 上换 seed，检查失败是否高度依赖随机初始化。

正式 multi-seed probe 运行命令：

```powershell
uv run python -m scripts.probe_heat_single_mode_training --modes 2,3,4,5,6,7 --seeds 0,1,2,3,4 --steps 500 --num-residual 200 --output-dir outputs/heat_single_mode_probe_m2_7_multiseed
```

输出文件：

- `outputs/heat_single_mode_probe_m2_7_multiseed/multi_seed_probe_summary.csv`
- `outputs/heat_single_mode_probe_m2_7_multiseed/multi_seed_probe_summary.json`

success 判据为同时满足：

```text
relative_complex_error_over_time < 1e-2
abs(decay_rate_error) / max(k_squared, 1e-12) < 0.1
```

## Success Rate

| Mode | Theory `k^2` | Success rate | Best seed | Best relative complex error |
| --- | --- | --- | --- | --- |
| `2` | `0.09869604401089357` | `0/5` | `4` | `0.28921942867826517` |
| `3` | `0.2220660990245106` | `0/5` | `1` | `0.5315791984955458` |
| `4` | `0.3947841760435743` | `2/5` | `3` | `6.157903898787147e-05` |
| `5` | `0.6168502750680849` | `1/5` | `3` | `0.0002250169052503812` |
| `6` | `0.8882643960980424` | `1/5` | `2` | `0.0002620815665778929` |
| `7` | `1.2090265391334463` | `3/5` | `3` | `0.00010868354441487618` |

mode `2` 到 mode `5` 不是同一种 seed 行为：

- mode `2` 的五个 seed 全部失败，best error 仍为 `0.28921942867826517`。
- mode `3` 的五个 seed 全部失败，best error 仍为 `0.5315791984955458`。
- mode `4` 在 seed `3` 和 seed `4` 成功。
- mode `5` 只在 seed `3` 成功。

因此，“mode `2` 到 mode `5` 全部只是碰到了一个坏 seed”不成立；但 mode `4` 和 mode `5` 又确实表现出明显初始化敏感性。

## Best Seed 衰减率

| Mode | Best seed | Theory `k^2` | Best fitted decay rate | Best decay rate error |
| --- | --- | --- | --- | --- |
| `2` | `4` | `0.09869604401089357` | `4.25275400665133e-09` | `-0.09869603975813956` |
| `3` | `1` | `0.2220660990245106` | `3.8325827691152283` | `3.6105166700907176` |
| `4` | `3` | `0.3947841760435743` | `0.39480163491591486` | `1.7458872340558873e-05` |
| `5` | `3` | `0.6168502750680849` | `0.6167595605556634` | `-9.071451242148587e-05` |

mode `4` 与 mode `5` 在最优 seed 下已经能学到接近理论的衰减率。mode `2` 的最优 seed 仍近似不衰减，mode `3` 的最优 seed 仍严重过衰减；它们没有因为这五个初始化切换而进入正确轨迹。

## 正对照 mode 6 与 mode 7

mode `6` 和 mode `7` 在上一轮 seed `1234` probe 中是正对照，但本轮跨 seed 并不稳：

- mode `6` 只有 seed `2` 成功，success rate 为 `1/5`。
- mode `7` 在 seed `0`、`3`、`4` 成功，success rate 为 `3/5`。

这说明当前小网络加 `L-BFGS` 的 isolated single-mode 优化本身对 seed 有明显分岔，并不只影响先前出问题的 mode `2` 到 mode `5`。

## Summary Flags

`multi_seed_probe_summary.json` 给出的结论 flags 为：

- `any_seed_succeeds_for_modes_2_to_5 = true`
- `all_seeds_fail_for_modes_2_to_5 = false`
- `modes_6_to_7_robust_across_seeds = false`
- `likely_initialization_sensitivity = true`
- `likely_loss_scaling_or_objective_issue = false`

这些 flags 说明 seed 能改变部分 mode 的成败，尤其是 mode `4`、`5`。不过它们不应被读成“loss/objective 已经没问题”：mode `2` 和 mode `3` 在当前五个 seed 下仍全部失败，而且正对照 mode `6` 也只有一个 seed 成功，说明优化轨迹对初始状态和 mode 尺度都很脆弱。

## 当前判断

当前更像是 **seed-sensitive 的单 mode 优化问题中夹着 mode 相关难点**：

1. mode `4`、`5` 存在可成功 seed，初始化显著影响是否进入正确衰减轨迹。
2. mode `2`、`3` 在本轮五个 seed 下都没有成功，不能只靠换 seed 解释。
3. mode `6`、`7` 也不跨 seed 稳定，说明 best-of-seeds 会挑到好轨迹，但不能让训练口径本身变稳。

因此本轮结果不足以支持把下一步主线改成 best-of-seeds。best-of-seeds 能展示存在可达解，尤其对 mode `4`、`5` 有用；但它绕不过 mode `2`、`3` 的持续失败，也不能解释同样配置下 mode `6` 的 seed 分岔。

## 下一步建议

下一步更值得转向归一化和 loss/objective 尺度诊断，而不是继续扩大 seed 搜索：

1. 先在单 mode 层面核查初始频谱幅值、residual loss 和闭式轨迹误差的尺度关系。
2. 以 mode `2`、`3` 与 mode `4`、`5` 的不同失败形态为对照，判断归一化目标是否能让相同 success 判据跨 mode 更稳定。
3. 保留 best seed 结果作为“目标可达”的旁证，不把它当成正式训练修复。
