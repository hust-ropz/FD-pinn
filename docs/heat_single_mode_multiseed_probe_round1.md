# Heat Single Mode Multi-Seed Probe Round 1

## 诊断目的

本轮只做 isolated single-mode multi-seed 诊断，不修改主训练脚本、不改 loss、不改优化器、不改网络结构。目标是回答一个窄问题：mode `2` 到 mode `5` 的失败是否主要由特定随机初始化导致。

运行命令：

```powershell
uv run python -m scripts.probe_heat_single_mode_multiseed --modes 2 3 4 5 6 7 --seeds 0 1 2 3 4 --output-dir outputs/heat_single_mode_multiseed_probe_round1
```

输出文件：

- `outputs/heat_single_mode_multiseed_probe_round1/per_seed_results.csv`
- `outputs/heat_single_mode_multiseed_probe_round1/per_mode_summary.csv`
- `outputs/heat_single_mode_multiseed_probe_round1/probe_summary.json`
- `outputs/heat_single_mode_multiseed_probe_round1/report.md`

## Success 判据

脚本顶部常量给出的默认 success 判据为：

- `initial_coeff_abs_error < 0.01`
- `final_coeff_abs_error < 0.01`
- `decay_rate_relative_error < 0.1`

三个条件同时满足时，`success_flag = true`。这里的 `decay_rate_relative_error` 定义为：

```text
abs(fitted_decay_rate - theoretical_decay_rate) / max(theoretical_decay_rate, 1e-12)
```

## 聚合结果

| Mode | Success rate | Best seed | Best final coeff abs error | Median final coeff abs error | Median decay rate relative error |
| --- | --- | --- | --- | --- | --- |
| `2` | `0 / 5` | `4` | `13.586695658231022` | `13.586696790317596` | `1.0000000000000002` |
| `3` | `0 / 5` | `1` | `10.29607881715655` | `19.487478372723622` | `1.000000000000001` |
| `4` | `2 / 5` | `3` | `0.0006448015140007333` | `19.11412620376243` | `1.0000000000000007` |
| `5` | `1 / 5` | `3` | `0.0034936885424859472` | `14.651033469773143` | `1.0000000000000002` |
| `6` | `1 / 5` | `2` | `0.0020050620749835647` | `9.520420766491684` | `1.0` |
| `7` | `3 / 5` | `4` | `0.00016270254340538983` | `0.0011715059027437972` | `0.0007399515420396605` |

`probe_summary.json` 的关键结论字段为：

- `modes_2_to_5_have_any_successful_seed = true`
- `modes_2_to_5_all_fail_across_seeds = false`
- `modes_6_to_7_remain_stable_across_seeds = false`
- `likely_seed_sensitivity = true`
- `recommended_next_step = 转向 loss/scale 诊断；不建议把 best-of-seeds 当作修复主线。`

## mode 2 到 mode 5 是否存在成功 seed

存在，但只发生在部分 mode：

- mode `2`: `0/5` 成功，seed `0..4` 全失败。
- mode `3`: `0/5` 成功，seed `0..4` 全失败。
- mode `4`: `2/5` 成功，seed `3` 和 seed `4` 成功。
- mode `5`: `1/5` 成功，只有 seed `3` 成功。

因此，mode `2` 到 mode `5` 不能被统一解释成“某个固定 seed 太差”。mode `4`、`5` 明显 seed-sensitive；mode `2`、`3` 在这五个 seed 下仍没有进入成功轨迹。

## 最优 seed 的衰减率

| Mode | Best seed | Theoretical decay rate | Fitted decay rate | Decay rate relative error |
| --- | --- | --- | --- | --- |
| `2` | `4` | `0.09869604401089357` | `4.25275400665133e-09` | `0.999999956910593` |
| `3` | `1` | `0.2220660990245106` | `3.8325827691152283` | `16.25874766995482` |
| `4` | `3` | `0.3947841760435743` | `0.39480163491591486` | `4.422384026514743e-05` |
| `5` | `3` | `0.6168502750680849` | `0.6167595605556634` | `0.00014706082835332` |

mode `4` 和 mode `5` 在最优 seed 下的衰减率已经接近理论值，说明正确轨迹是可达的。mode `2` 最优 seed 仍几乎不衰减；mode `3` 最优 seed 仍严重过衰减，说明这两个 mode 的问题不能靠当前五个 seed 的 best-of-seeds 解决。

## mode 6 到 mode 7 是否稳定

mode `6` 和 mode `7` 不是跨 seed 稳定成功：

- mode `6`: `1/5` 成功，只有 seed `2` 成功。
- mode `7`: `3/5` 成功，seed `0`、`3`、`4` 成功。

这说明当前小网络加 `L-BFGS(max_iter=3)` 的 isolated 单 mode 训练整体存在 seed 分岔。mode `6`、`7` 可以作为“存在成功轨迹”的正对照，但不能说明当前训练口径跨 seed 稳定。

## 当前判断

本轮结论是分化的：

1. mode `4`、`5` 对随机初始化敏感，某些 seed 可以成功。
2. mode `2`、`3` 在 seed `0..4` 下全部失败，不能只归因于单个坏初始化。
3. mode `6`、`7` 也没有跨 seed 稳定成功，说明 seed sensitivity 是整体优化现象，不只限于 mode `2..5`。

因此，当前不建议把下一步主线改成 best-of-seeds。best-of-seeds 能证明部分 mode 存在可达解，尤其是 mode `4`、`5`；但它不能解释 mode `2`、`3` 的持续失败，也不能让 mode `6`、`7` 稳定。

下一步应转向 loss/scale 诊断：先检查单 mode 的初值系数尺度、残差项尺度、闭式轨迹误差之间是否存在跨 mode 不均衡，再决定是否需要做归一化单 mode 目标。
