# Poseidon 压缩一致性检查（2026-10-14）

## 最新对齐：以 MySVD_fast/mysvd.py 为基准

已按用户指定修正两项：Poseidon Fisher 只附加一阶及以上空间导数损失；默认右侧白化改为 new/sum_equivalent/1e-6。默认 comb 压缩直接使用原始 Gram 求和加绝对 eps，与 MySVD_fast 保持同一 SVD 数值尺度。输入 profile 仍输出均值度量，使用 eps/N；rank selection 的 N 能量缩放保持原规则。

Fisher/rank 默认缓存路径加入 derivonly，避免沿用旧零阶项缓存。显式指定旧 Fisher/rank 文件仍会读取，需由调用者重新计算或确认来源。rank selector、采样与模型加载没有在此次修改中变更。

下面保留此前针对 SAFE_SVD 的检查记录；其中“本仓库 Poseidon 默认 origin”描述的是此次修改前的状态，不能作为 MySVD_fast 的对齐结论。新增 MySVD_fast 数值测试分别对照实际 loss 函数及 mysvd_comb 完整逐层压缩。

验证结果：93 项 CPU 回归测试通过。MySVD_fast loss 对照覆盖不可压缩流体/wave、L1/MSE、0/1/2 阶，损失值与梯度完全一致；固定 rank 小模型的逐层因子和输出通过 rtol=2e-5、atol=2e-6 对照。未运行真实 Poseidon-L GPU 压缩或评估。

用户报告当前入口指标为 `0.006826038472354412`，参考仓库约为 `0.0023–0.0024`。这两次运行的完整命令与指标名称没有保留，因此下面区分代码事实和仍未验证的原因；不能把某一项差异直接认定为全部指标差距的原因。

本次检查使用参考源码作本地测试 oracle。生产入口不导入或依赖参考仓库。

## 1. rank 的对应关系需要纠正

当前参考 `poseidon/SAFE_SVD/rank_selector.py` 的 `get_global_greedy_rank_dict_v2` 是动态预算算法：每层先购买 rank=1，随后购买下一奇异值或转为 full-rank。
它对应本仓库的 **safe**，不是 legacy。`SAFE_SVD` 的 new whitening 版本还乘每层输入行数 N，当前入口的 safe/new 已实现相同规则。

参考目录另外还有 `MySVD_fast`、`MySVD_origin`、`MySVD` 等不同方法。`MySVD_fast/helper/rank_selector.py` 的旧全局排序算法才对应本仓库的 legacy。
之前笼统地说“复现旧代码选 legacy”不够准确；必须先确定具体方法目录。

现有对话中的失败运行明确加载了路径包含 `_legacy_y/rank_0.2.json` 的 rank 缓存。这只能证明那次运行用了 legacy 文件，不能证明后来得到 0.006826 的运行没有改过参数。

CPU 数值对照：两个 8×8 Linear，第一层能量显著大于第二层，ratio=0.5 时：

| 算法 | 第一层 | 第二层 | 权重成本 |
| --- | --- | --- | --- |
| 参考 SAFE_SVD | rank 3 | rank 1 | 64 |
| 本仓库 safe | rank 3 | rank 1 | 64 |
| 本仓库 legacy | full-rank | rank 1 | 80 |

因此相同 ratio 不保证相同预算实际使用量，更不保证相同各层 rank。应优先比较两次的 rank JSON，而不是只比较 ratio。

## 2. origin/new 与 raw Gram/mean 不是同一个配置

| 入口 | rank 输入统计 | compression 右 whitening |
| --- | --- | --- |
| 参考 SAFE_SVD/safe_svd.py 默认 | mean covariance，origin | Cholesky + 约 1% 对角均值 jitter |
| 参考 SAFE_SVD/safe_svd.sh 当前实验列表 | new / absolute eps=1e-12、1e-15、1e-18 | 谱分解 + 对应 epsilon |
| 参考 MySVD_fast | raw Gram sum，谱分解 eps=1e-6 | raw Gram sum 的谱分解 |
| 本仓库 Poseidon 默认 | mean covariance，origin | Cholesky + 约 1% jitter |

两边直接运行各自 shell 脚本，并不一定是 origin 对 origin。origin 不使用 whitening_eps，把 eps 改成 1e-12 但仍指定 origin 不会复制谱分解配置。

例如 mean(diagonal)=1 时，origin 的 jitter 约为 0.01，而 new/absolute/1e-12 是 1e-12。正则量级和方向都不同，这会改变重建目标。

若比较的是 MySVD_fast 的 raw-Gram 算法，对应配置为：

```text
--rank_selector legacy --whitening_mode new
--whitening_eps_mode sum_equivalent --whitening_eps 1e-6
```

这是算法尺度对应关系，不代表旧的 0.0023/0.0024 一定使用了 MySVD_fast。
因为 `G_mean=G_sum/N`，用 eps/N 可以保持相同的正则化矩阵尺度；new rank 的能量乘 N 恢复原来的跨层能量比例。控制实验仍应共享相同 checkpoint、校准数据、Fisher 和 rank 文件。

## 3. Fisher 的实际启用状态不同

参考 `SAFE_SVD/safe_svd.py` 虽然导入了 collect_fisher_dict_linear_only，但没有调用它：只在指定 Fisher 缓存存在且未 overwrite 时读取，否则 left_metrics=None。
所以这个入口可能在没有 Fisher 的情况下完成 rank/compression。

本仓库 `use_fisher=y` 默认会生成缺失的 Fisher 缓存并使用它。参考 MySVD_fast 则会主动采集 Fisher，其 batch 固定为 8；本仓库 CLI 默认 16，可显式设为 8。

不能用“同样传了 order=2/l1”判断两边都使用相同 Fisher。要对比实际加载的 Ldict、缓存来源和非缺失层。
当前日志中的 Loaded Fisher cache 表示没有重新采集；修改 Fisher batch 不会改变已显式指定的缓存内容。

本次修复：Poseidon 默认 Fisher/rank 缓存名称加入 Fisher batch 标签，避免 8 和 16 静默共用同一自动缓存路径。已有文件不会被删除。

## 4. 已修正的重建精度差异

参考 origin 在 Cholesky 后将因子转成 float32，并在 float32 中求逆；原本当前 compression 在 float64 中保持因子并求三角逆。
参考 SVD 后还把截断因子转成 float32，再做左右逆变换；原本当前 compression 保留 float64 到最后复制权重。

现在已在本仓库实现相同的 Poseidon 精度策略。MPP/VICON 的截断因子策略不变，模型原始权重 dtype 仍保留。
一个 8×8 正定测试中，修复前 origin 因子最大绝对差约为 3.87e-8；修复后与源码 oracle 的因子/逆精度完全一致。这个小矩阵结果不能证明精度差异导致了实际模型近三倍的指标变化。

## 5. 采集与逐层重建的对照结果

已比较完整轨迹采样、输入 profile、Fisher 输出梯度统计及 helper 的重算路径。
轨迹选择 helper 与参考文件一致；Input/Fisher 有源码 oracle 对照；QKV 缓存从 upstream 参数重算当前 student，而不是复用过时的下游输出。

新增测试直接执行参考 compress 函数，在相同数据与固定 ranks 上与本仓库比较：

- origin 与 new 两种完整逐层压缩；
- attention query/key/value/output 全部压缩；
- query 保留时的无缓存回退；
- value 保留时的 output 回退。

这些 CPU 小模型输出通过 rtol=2e-5、atol=2e-6 的对照，并不替代真实 Poseidon-L 全数据集评估。CPU/GPU 缓存位置不会改变定义的统计目标；batch 分组与浮点计算仍可能造成数值差异。

## 6. 评估与模型身份仍未确认

本仓库 evaluation/poseidon/physics.py 的 base_loss 来自 rollout 的 `_loss`；它不等于 mean_relative_l1_error、Sobolev loss 等其他字段。
参考 physical_loss.py 如果发现已有结果 CSV，会先读缓存；因此历史打印值不一定来自刚生成的模型。
这两次的 checkpoint 权重、测试 split、final_time、ongt/oncomp、指标字段、模型代码 revision 和缓存来源目前都没有运行记录证明一致。

官方 PiecewiseConstants 数据路径使用 constants.mean/std 和所有允许的时间对；本次核对的官方代码与参考数据采集约定一致，但服务器具体安装 revision 没有核实。
来源：[官方 incompressible 数据实现](https://raw.githubusercontent.com/camlab-ethz/poseidon/main/scOT/problems/fluids/incompressible.py)、[官方时间数据实现](https://raw.githubusercontent.com/camlab-ethz/poseidon/main/scOT/problems/base.py)。

## 7. 后续对照应固定什么

先使用同一 checkpoint、同一校准轨迹，显式指定同一 rank JSON 和同一 Fisher 文件，再仅比较 compression；随后才比较各自自动生成 rank 的整条流程。
两份压缩模型应使用同一个 evaluation 入口、同一个 test dataset 和指标字段。新的输出目录可避免旧最终模型或旧评估结果被误当成新结果；不需要删除历史实验。

CLI 现在在输出目录写 run_config.json，保留命令、所有解析参数、路径、缓存存在状态、Torch 版本、checkpoint config 哈希，成功后补充 rank JSON 哈希。
它不能恢复历史记录，也不能通过 config 哈希证明权重文件相同，但可以避免下一次对照再次丢失基本配置。
