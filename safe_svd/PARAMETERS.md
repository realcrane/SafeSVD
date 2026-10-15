# main.py 参数说明

统一入口：`python -m safe_svd.main --model mpp|vicon|poseidon ...`。
运行顺序是模型/数据加载 → Input Profiling → Fisher Collection → Rank Selection → Compression → 保存。
所有压缩和采集 helper 都在本仓库；模型定义与基础数据读取使用本仓库内安装的官方源码。

## 1. 压缩与 rank selection

| CLI 参数 | 默认值 | 定义和影响 |
| --- | --- | --- |
| `--model` / `--model_type` | 必填 | `mpp`、`vicon`、`poseidon`，选择模型加载、输入预处理、损失和采集路由。 |
| `--rank_ratio` | MPP `0.4`；VICON `0.8`；Poseidon `0.2` | 参与压缩的全部 Linear **权重**的保留参数预算比例，范围 `(0,1]`。预算为 `floor(ratio × Σ out_features × in_features)`；rank=k 的低秩层成本为 `k × (in_features + out_features)`。不包括 bias、Conv、Norm 等参数，也不是每层统一的 rank 比例。 |
| `--rank_selector` | MPP/VICON `legacy`；Poseidon `safe` | `legacy` 保留原贪心分配规则，补最低 rank 时可能超预算；`safe` 预先计入最低 rank 成本，并用动态贪心分配保证预算。输出 `None` 表示保留原层。 |
| `--alpha` | `0.7` | `comb` 中当前 student 输入修正的权重，范围 `[0,1]`。输入度量是 `(1-alpha) HᵀH + alpha HpᵀHp`，交叉项是 `(1-alpha) HᵀH + alpha HᵀHp`；H/Hp 是对应 teacher/student 的输入。`0` 使用 teacher 输入，增大后更重视对当前 student 输入的适配。`clean_input` 中不参与计算。 |
| `--compress_mode` | `comb` | `comb` 逐层采集 teacher/当前 student 的对应输入，再做低秩替换；`clean_input` 使用 Step 1 的原始模型输入度量，不重新采集 student 输入。两者都使用所选 rank 和可选 Fisher 左度量。 |
| `--use_fisher` | `y` | `y` 收集/加载 Fisher 左度量 L，在 rank selection 和 compression 中使用；`n` 跳过 Fisher，用单位左度量。这里的 Fisher 是 **Linear 输出梯度的协方差**，不是完整参数 Fisher 矩阵。 |
| `--use_trace_balance` / `--use_trace_align` | MPP/VICON `y`；Poseidon `n` | 对类别协方差按 trace 对齐后取类别均值，减弱类别量级差异。影响 Input、Fisher 和 `comb` 的统计；VICON Fisher 按 Sobolev 阶数分组对齐。关闭时 MPP/Poseidon 按展开后的特征行数加权，VICON 按类别等权。 |

## 2. Fisher 损失

| CLI 参数 | 默认值 | 定义和影响 |
| --- | --- | --- |
| `--fisher_loss` / `--gradient_loss` | `l1` | 物理/Sobolev 项的误差函数：`l1`、`mse`、`smooth_l1`。MPP 只支持前两项。它不替换基础模型损失：MPP 使用相对 MSE，VICON 使用 masked MSE，Poseidon 使用模型返回的 `outputs.loss`。 |
| `--balance_ratio` | `10.0` | 物理项强度。MPP/Poseidon 对每阶项使用 `min(base_loss / physical_loss × balance_ratio, 1000)` 的动态归一化系数；VICON 直接用 `base_loss + balance_ratio × derivative_loss`。`0` 可去掉物理项，通常使用非负值。 |
| `--decay_alpha` | `1.0` | MPP/Poseidon 对第 k 阶空间导数项乘 `decay_alpha ** (k-1)`。`1` 不衰减；小于 1 降低高阶项权重。当前 VICON 路由不使用它。 |
| `--order_control` | `2` | **Poseidon** 最大空间差分阶数，非负整数。与 MySVD_fast 一致，只附加一阶及以上空间差分损失；`0` 只保留模型 base loss。MPP/VICON 根据数据类别确定阶数，不使用此 CLI 值：MPP incompNS/diffre2d 为 2、compNS/swe 为 1；VICON NS2D 为 2、COMPRESSIBLE2D/EULER2D 为 1。 |

当前 Poseidon Fisher 没有传入 `prev_pred/prev_labels`，因此只附加空间差分物理项，不额外计算时间差分损失。
`use_fisher=n` 时，上述 Fisher 损失参数都不参与计算。

## 3. Whitening

| CLI 参数 | 默认值 | 定义和影响 |
| --- | --- | --- |
| `--whitening_mode` | `new` | `origin` 使用 Cholesky 和固定相对 jitter，失败时用特征分解；`new` 使用谱分解和显式 epsilon。影响输入度量和 compression 的右度量。Poseidon 默认与 MySVD_fast 的谱白化对齐；显式 origin 是另一种配置。Fisher 左度量始终使用独立的 Cholesky/eigen 例程。Poseidon 的 `new` rank selection 还将每层奇异值能量乘以输入特征行数，以保持原求和统计的相对尺度。 |
| `--whitening_eps` / `--eps_whiten` | `1e-6` | `new` 中特征值的正则量，必须为正。**对 `origin` 和 Fisher 左度量不生效**；它们使用约 `0.01 × mean(diagonal)` 的 jitter，零尺度时使用固定下限。 |
| `--whitening_eps_mode` / `--eps_whiten_mode` | Poseidon new：`sum_equivalent`；其他：`absolute` | `absolute` 使用 eps；`relative` 使用 `eps × mean(diagonal)`，尺度有数值下限；`sum_equivalent` 在均值统计上使用 `eps / N`，N 是该层展开后的输入特征行数。Poseidon comb 在默认 sample_mean/no-trace 配置下直接累加原始 Gram 求和，并加绝对 eps，以保持 MySVD_fast 的 SVD 和截断因子数值尺度。只对 `new` 生效。 |

## 4. 采样、batch 和显存

| CLI 参数 | 默认值 | 定义和影响 |
| --- | --- | --- |
| `--calibration_num` | MPP `16`；VICON `2048` | MPP 每个文件、VICON 每个类别的目标采样数。MPP 每文件最多 `ceil(calibration_num / batch_size)` 个完整 batch，DataLoader 设置 drop_last=True；VICON 每类别最多 `floor(calibration_num / batch_size)` 个完整 batch，要求至少一个 batch。不是 Poseidon 的采样参数。 |
| `--traj_num` | `64` | Poseidon 完整训练轨迹数，不是样本数或 batch 大小。一条轨迹可能生成多个时间对，最终 dataset 长度通常大于 traj_num。 |
| `--calibration_seed` | MPP `42`；VICON `2026`；Poseidon `default` | MPP 每个文件 shuffle 的种子。VICON 在 random_calibration=y 时用于随机校准。Poseidon `default` 取训练集前 traj_num 条；非负整数（包括 `0`）表示无放回随机选择完整轨迹。 |
| `--random_calibration` | `n` | VICON 专用，是否启用按 calibration_seed 随机选择；默认保持原校准流程。固定流程的数据构造使用 cfg.train_seed。 |
| `--final_time` | `20` | Poseidon 非椭圆数据的 `max_num_time_steps`，并设置 time_step_size=1；不是训练轮数，也不保证等于物理时间单位。椭圆数据不传该参数。 |
| `--batch_size` | MPP：YAML；VICON `128`；Poseidon `256` | MPP/VICON 的实际 DataLoader batch 大小，影响各采集阶段。Poseidon 是 Trainer 的 profiling/rollout batch 大小，**不控制 Fisher backward**。Poseidon 压缩时一个 capture slice 还可能被 Trainer 拆成更小 batch。 |
| `--fisher_batch_size` | `16` | Poseidon Fisher 前向/反向每批样本数。MPP/VICON 使用自己的 batch_size，不使用此项。 |
| `--capture_batch_size` | `64` | Poseidon compression 每次传给采集 rollout 的数据切片大小，按 dataset **样本**计，不是轨迹。实际一次前向通常不超过它与 Trainer batch_size 的较小值。MPP/VICON 不使用此项。 |
| `--dataset_workers` | VICON：cfg.dataset_workers | VICON 校准 DataLoader 工作进程数，非负整数。MPP 使用 params.num_data_workers；当前 Poseidon profiling、Fisher 和 capture 使用 0 worker。 |
| `--DEV` / `--device` | `cuda` | 计算设备，例如 cuda、cuda:0、cpu。选择模型、统计和 SVD 的执行设备。 |
| `--compression_backend` | `GPU` | `GPU` 的 **compression** 统计和分解使用 float64；`low-gpu` 的 compression 使用 float32。**两种模式的 Poseidon 捕获参数和复用缓存都默认放 CPU**，与 safesvd 的采集策略一致。teacher/student 模型仍在指定设备；不会把整个算法改成 CPU。此参数不改变模型权重 dtype、Input/Fisher 精度设置，也不改变 rank selection 的 float64 计算。 |

显存排查时先降低 `batch_size`、`capture_batch_size`，使用 `compression_backend=low-gpu`。
只有 Fisher 阶段 OOM 才需要先降低 fisher_batch_size。不要把减少 traj_num 当成减少单次前向 batch 的同义操作。
low-gpu 会改变 compression 的数值精度，并增加 CPU 内存与传输开销；不能保证任意模型都能装进指定 GPU。

## 5. 模型、数据和配置路径

用户显式传入的相对路径按命令执行目录解析。模型源码只定位本仓库内的官方 checkout；没有 source_root 参数。

| CLI 参数 | 默认值 | 定义 |
| --- | --- | --- |
| `--ckpt_path` / `--model_path` | MPP：models/MPP/MPP_AViT_L；VICON：models/VICON/model.pth；Poseidon：models/poseidon/数据集名称 | MPP/VICON 为 checkpoint 文件；Poseidon 为包含 config.json 与权重的具体目录。所有默认路径均相对于本仓库。 |
| `--data_root` / `--data_path` | MPP：data/official/mpp；VICON：data/VICON；Poseidon：poseidon/camlab-ethz/down_streams | 数据根目录。MPP YAML 的相对数据条目拼在此目录下，绝对条目（含展开后的 ~）保持自身位置。VICON 自动重定位 folder/path/data_path；显式 config_override 的目录优先。Poseidon 传给官方 get_dataset。 |
| `--yaml_config` | 优先 data/official/configs/mpp_avit_L.yaml，否则官方 MPP config/mpp_avit_L_config.yaml | MPP 架构和数据路径配置。 |
| `--config` | `basic_config` | MPP YAML 中读取的配置段名。 |
| `--config_dir` | 官方 VICON checkout 的 configs/ | VICON Hydra 配置目录。 |
| `--config_name` | `default` | VICON Hydra 主配置名。 |
| `--config_override` | 空列表 | 可重复传入 Hydra KEY=VALUE，例如 datasets.types.NS2D.folder=/data/NS。模型架构的覆盖值必须与 checkpoint 一致。 |
| `--dataset` | `fluids.incompressible.PiecewiseConstants` | Poseidon 官方数据集注册名称，同时用于物理通道规则及缓存目录。须与数据和 checkpoint 的任务一致。 |
| `--model_name` | `Poseidon-L` | Poseidon **缓存目录标签**。实际模型结构来自 checkpoint 的 config.json，不由这个参数创建。 |
| `--use_phyloss` | `no_phyloss` | 原命令兼容参数；当前入口不读取它，不改变加载的 checkpoint、不启用/关闭 Fisher 物理损失。要选择权重用 model_path；要关闭物理项用 balance_ratio=0。 |
| `--rank_type` | `MySVD_Rank`（唯一选项） | VICON 缓存名称兼容标签，不选择算法。实际算法由 rank_selector 选择。 |

## 6. 阶段、缓存和输出

| CLI 参数 | 默认值 | 定义 |
| --- | --- | --- |
| `--pipeline_stage` | `all` | all 执行四阶段并保存；rank_selection 执行 Input、Fisher、Rank 后停止；compression 加载已有 Input/Fisher/Rank 缓存，只重新执行压缩。compression 仍会加载模型与校准数据；use_fisher=n 时不要求 Fisher 缓存。 |
| `--profile_cache` | cache_dir/input/设置标签/profiling_mat.pt | Input 的 Rdict 和 sample_counts 缓存。存在时读取，否则采集并保存。 |
| `--fisher_cache` | cache_dir/fisher/设置标签/Ldict.pt | Fisher 的 (L, L⁻¹) 缓存。 |
| `--rank_cache` / `--rank_path` | rank_root/设置标签/rank_比例.json | 每层 rank 字典缓存，None 在 JSON 中为 null。 |
| `--cache_dir` | 本仓库 artifacts/cache/数据集或模型/模型标签 | 自动生成的 Input/Fisher 缓存根目录；显式 profile_cache/fisher_cache 优先。 |
| `--rank_root` | 本仓库 artifacts/ranks/数据集或模型/模型标签 | 自动生成的 rank 缓存根目录；显式 rank_cache 优先。 |
| `--save_path` | 本仓库 artifacts/compressed/数据集或模型/模型标签/比例/设置标签 | 输出目录。MPP/Poseidon 保存完整 nn.Module 到 compressed_model.pt；VICON 保存 state_dict 到 compressed_model.pth；三者都保存 rank_dict.json。 |
| `--checkpoint_percent` | `0.1` | 压缩每处理 `max(int(Linear层总数 × 比例), 1)` 层保存一次 compressing_model.pt；比例范围 `(0,1]`，最后不足一个间隔也保存。815 层时默认每 81 层保存。 |
| `--overwrite_cache` | 关闭 | 强制重新采集/分配缓存，忽略压缩断点，并允许替换已有最终模型。compression-only 不允许此选项。默认遇到已有最终模型会报错，避免覆盖。 |
| `--gpu_profile` | 关闭 | 在 save_path/gpu_profile.csv 记录四阶段耗时与内存指标。 |
| `--dry_run` | 关闭 | 打印解析后的配置和路径后退出，不加载模型、数据，也不创建缓存。不验证文件存在或 GPU 能否实际运行。 |
| `--help` | — | 显示 CLI 选项定义。 |

改变 batch_size 会改变默认缓存路径中的 bs 标签。若压缩 OOM 后仅降低前向批大小，想复用已经完成的 Input/Fisher/Rank，必须显式指定原来的三个缓存文件。
Poseidon 自动生成的 Fisher/rank 缓存路径包含 `_fbs8` 等 Fisher batch 标签，避免调整批大小后误读另一批大小生成的结果；这会使历史默认路径不再自动命中。历史缓存保留原文件，确认兼容后可以显式指定。
Poseidon Fisher/rank 路径还包含 `_derivonly`，隔离旧零阶损失产生的缓存。旧入口生成的 Fisher 和由其生成的 rank 应重新计算；显式指定缓存路径时不会检查其损失定义。
缓存路径没有完整 checkpoint/配置内容哈希；调用者必须保证复用的是同一模型、数据和采样设置的兼容缓存。

CLI 在输出目录保存 `run_config.json`，记录完整命令参数、解析后的路径、缓存在运行前是否存在、Torch 版本及 checkpoint config 的 SHA-256；成功后补充 rank JSON 的 SHA-256。config 哈希不能证明权重文件相同，缓存存在记录也不能证明缓存生成时的参数；这些信息不能替代一致性检查。

压缩断点保存已完成层名、rank 和压缩设置，以及当前 student。MPP/Poseidon 使用完整模型；VICON 保存 state_dict，恢复时根据已完成层及 rank 重建低秩结构。恢复不保存激活缓存，后续层需要的输入会按当前 student 重新采集。模型权重、数据和采样必须与原运行一致；断点会检查已记录的设置与 rank，但不校验全部权重或数据文件内容。可降低 batch_size/capture_batch_size，改变压缩 dtype/alpha/whitening 等设置会报错。

断点和最终模型均先写临时文件再替换正式文件；保存失败保留上一份完整断点。最终模型和 rank_dict.json 保存成功后删除 compressing_model.pt。Input/Fisher/Rank 仍按阶段缓存，不提供采集到一半的 batch 级恢复。

Poseidon reconstruction 还遵循参考实现的 float32 精度策略：`origin` 的 Cholesky 因子/逆先以 float32 构造，随后转换成 compute_dtype；SVD 截断后的因子在逆变换前转为 float32。GPU 模式的协方差和 SVD 仍使用 float64。

## 7. Python main() 接口

`main(model, calibration_data, *, ...)` 接受已经加载好的模型与数据。以上同名算法参数仍适用；它不负责通过 traj_num 等 CLI 参数重新选择数据。
它直接返回 CPU 上的 compressed nn.Module；pipeline_stage=rank_selection 时返回 None。

| Python 参数 | 默认值 | 定义/与 CLI 的区别 |
| --- | --- | --- |
| `model` | 必填 | 已加载权重的 nn.Module。 |
| `calibration_data` | 必填 | MPP train_dataset；VICON 类别到 DataLoader 的字典；Poseidon Dataset 或已有 Fisher DataLoader。 |
| `model_type`、`rank_ratio` | 必填 | 对应 CLI 的 model、rank_ratio。 |
| `device` | `cuda` | 对应 DEV。 |
| `params` | `None` | MPP 必需的模型/数据参数对象，包含 batch_size、num_data_workers 等。 |
| `trainer` | `None` | VICON/Poseidon 必需的模型 Trainer；采集时临时替换其 model 并恢复。 |
| `rollout_fn` | `None` | Poseidon 可注入的 rollout 函数；未传时延迟导入官方 scOT.inference.rollout。 |
| `dataset_name` | `None` | Poseidon 物理规则与统计标签，CLI 对应 dataset；未传时使用 poseidon 标签。 |
| `use_fisher` | `y` | 同 CLI，也接受 bool。 |
| `rank_selector` | `safe` | Python API 三模型都默认 safe，CLI 的 MPP/VICON 默认 legacy。 |
| `alpha`、`compress_mode` | `0.7`、`comb` | 同 CLI。 |
| `gradient_loss` | `None` | None 时 MPP/VICON 使用 l1，Poseidon 使用 mse。**CLI 三模型默认 l1**。 |
| `balance_ratio`、`decay_alpha` | `10.0`、`1.0` | 同 CLI。 |
| `order_control` | `0` | Python Poseidon 默认 0；CLI 默认 2。MPP/VICON 忽略这个值。 |
| `calibration_seed` | `42` | MPP shuffle 种子。Python API 的 VICON/Poseidon 校准数据已经准备好，此参数不重新抽样。 |
| `batches_per_dset` | `16` | MPP 每文件的 batch 上限，可为正数或 None（全部）；CLI 设置为 calibration_num/params.batch_size。 |
| `max_calib_batches` | `None` | VICON 每类别 batch 上限；CLI 设置为 floor(calibration_num/batch_size)。当前 main 的 Poseidon route 不传此限制。 |
| `train_seed` | `None` | VICON Input Profiling 前恢复原训练随机种子；CLI 使用 cfg.train_seed。 |
| `fisher_batch_size`、`capture_batch_size` | `8`、`64` | Python main 的 Fisher 默认 8，CLI 默认 16。若传已有 Poseidon DataLoader，Fisher 保留其 batch 设置而不是重新构造。 |
| `use_trace_balance` | `None` | 自动取 MPP/VICON True、Poseidon False；支持 y/n 或 bool。 |
| `whitening_mode` | `None` | 自动取 new。 |
| `eps_whiten`、`eps_whiten_mode` | `1e-6`、`None` | Poseidon new 自动取 sum_equivalent；其他自动取 absolute。对应 whitening_eps、whitening_eps_mode。 |
| `input_accumulation_dtype` | `None` | 仅 Input 统计累积 dtype：MPP/VICON 默认 float64，Poseidon 默认 float32；输入 Gram 乘积仍默认 float32。 |
| `compute_dtype` | `torch.float64` | 仅 compression 的统计/矩阵/SVD dtype，可设 float32。不修改 rank selection 的 float64。CLI low-gpu 设为 float32。 |
| `offload_cache` | `True` | Poseidon capture 参数与复用缓存默认放 CPU；模型仍在 device。CLI 的 GPU 和 low-gpu 都为 True，缓存位置不再与统计精度绑定。Python API 可以显式 False 使用设备缓存，但显存会随校准样本数量累计。 |
| `fisher_loss_fn` | `None` | 自定义可微标量损失回调 `(model, CalibrationBatch) -> Tensor`，覆盖默认 Fisher 损失路由。无 CLI 选项。 |
| `profile_cache`、`fisher_cache`、`rank_cache` | 均 `None` | 显式缓存文件；API 默认不读写缓存，CLI 自动生成路径。 |
| `checkpoint_percent` | `0.1` | 与 CLI 相同的逐层保存与自动续跑。只有提供 save_path 时写断点；overwrite_cache=True 忽略旧断点。 |
| `save_path` | `None` | API 默认只返回模型、不保存；CLI 默认生成输出目录。 |
| `overwrite_cache` | `False` | 同 CLI。 |
| `verbose` | `True` | 阶段/缓存日志及 Fisher、compression 的 tqdm 进度；False 关闭这两个进度条，Poseidon 官方 Input rollout 的进度由 Trainer 控制。 |
| `pipeline_stage` | `all` | 同 CLI。 |
| `perf_csv_path` | `None` | 显式阶段性能 CSV 文件，CLI gpu_profile 自动设为 save_path/gpu_profile.csv。 |

## 8. 进度条单位

Fisher 的第一个进度条按完成的 forward/backward **batch** 更新，显示数据类别与损失。无长度的 stream 数据不显示虚构的总数。
Compression 按原模型的目标 **Linear 层**更新，显示当前层名及 rank；保留、跳过和已完成层也计入总层数。缓存命中的 Fisher 不执行采集，因此只打印缓存路径。

## 9. 本次 Poseidon OOM 的恢复示例

日志停在 Step 4，前面已经写出的缓存可以保留。使用同一 checkpoint、dataset、轨迹选择和 rank 设置，显式填入原来的三个缓存文件：

```bash
python -m safe_svd.main --model poseidon \
  --model_path /path/to/checkpoint \
  --data_path /path/to/data \
  --dataset fluids.incompressible.PiecewiseConstants \
  --rank_ratio 0.2 --traj_num 64 --calibration_seed default --final_time 20 \
  --batch_size 8 --capture_batch_size 8 \
  --compression_backend low-gpu \
  --pipeline_stage compression \
  --profile_cache /path/to/original/profiling_mat.pt \
  --fisher_cache /path/to/original/Ldict.pt \
  --rank_cache /path/to/original/rank_0.2.json \
  --save_path /path/to/original/output
```

这些是示例值，必须替换为原运行的路径与采样/压缩参数。此方式跳过前三阶段；save_path 应指向原运行的输出目录，存在 compressing_model.pt 时自动跳过已保存的层。没有断点时从第一层开始。恢复必须保持原 compression_backend；示例 low-gpu 只适用于原运行也采用 low-gpu 的情况。

## 10. 与 safesvd/ 的 Poseidon 采集配置对比

以下以本仓库当前实际代码为准。`safesvd/adapters/poseidon_runtime.py` 的常量为 Fisher batch=8、capture chunk=64；部分旧注释仍写 512，那不是当前常量值。

| 项目 | safesvd/ | safe_svd/ 修正后 |
| --- | --- | --- |
| Fisher batch | 固定 8 | CLI 默认 16，可用 fisher_batch_size=8 对齐；缓存命中时不执行 Fisher。 |
| Input profiling / rollout batch | Trainer 固定 256 | batch_size，默认 256；fisher_batch_size、capture_batch_size 不改变 Step 1 的 batch。 |
| Compression capture slice | 当前常量 64 | capture_batch_size，默认 64，可设 8。只限制每批样本数，不减少总校准数据。 |
| MLP、QKV、patch 复用缓存 | 始终 CPU | 默认 CPU，两种 compression_backend 都一致；修正前 GPU 模式会把整份校准缓存累计在 GPU。 |
| 重算出的 MLP/attention/patch 特征 | 送回 CPU 后统计 | 默认也送回 CPU；先释放 teacher 重算的 GPU 输出，再执行 student 重算。 |
| GPU 统计与 SVD | float64 | GPU 模式 float64；low-gpu float32。矩阵大小由特征维数决定，不随 batch 缩小。 |
| 压缩阶段边界 | 模型临时搬 CPU、gc、empty_cache | Step 4 前同样清理；这是释放未使用存储，不能清除仍被缓存引用的活跃张量。 |
| 层间矩阵存储 | 随使用结束释放 | 及时消费协方差，提前删除 weight/left/right 工作矩阵，层间释放未使用 CUDA 缓存。 |

CUDA 显存主要由两份模型、当前前向/特征、复用缓存、协方差与分解工作空间组成。仅缩小 capture_batch_size，只能减少当前批特征；若复用缓存仍在 GPU，遍历完整校准集时总占用仍会不断增加。
Step 4 中 OOM 时，调整 Fisher batch 没有效果；若日志写 Loaded Fisher cache，它连 Step 2 都没有重新运行。
