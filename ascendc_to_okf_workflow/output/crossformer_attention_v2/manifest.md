# P0 材料清单 — CrossformerAttention（v2 重提取，服务 level4 算子）

> 生成时间：2026-10-10｜服务目标：`src/flash_attention_score/level4_target/CrossformerAttention/`
> 参考源：主 `src/flash_attention_score/`（FAS 前向，pse_shift/atten_mask/scale 语义 + FlashSoftmax 算法），
> 次 `src/swin_attention_score_quant/`（窗口化结构，量化版，仅取窗口分块组织方式）。
> 两者均克隆自 `src/ops-transformer`（gitcode.com/cann/ops-transformer @ d28d8f99a，CANN Open Software License 2.0）。
>
> **版本关系**：v2 与 `output/crossformer_attention/`（2026-09-21 六件套）**材料同源、方法论不同**。
> v2 按工作流 **v1.5「CANN 提取价值方法论」**（2026-10-09 新增）重跑：**先 host tiling 阶梯，后 kernel 流水**，
> L0~L3 归责、排除法≠穷举、pip 门。核心增量 = 取 FAS host `*_tiling_basic.cpp` 的 **D/S1/S2 档位阶梯**
> （原提取只取了 tiling 任务轴与 S1/S2 语义，漏了档位阶梯——恰是 v1.5 定义的 attention 族**第一顺位**价值）。

**kernel 类型判定：CV 融合（Cube+Vector）**——注意力主体是 Cube 矩阵乘（QKᵀ、softmax、·V），
CANN 参考 `flash_attention_score.cpp` 走 FlashAttention（FlashSoftmaxV2 在线 softmax）。**非壳算子**（调度+计算在仓）。

**🔺 低价值预判：非低价值。**
- ❌ 不可达项 ≥2：① Cube-Vector 异构流水（SkvSplit 在线 softmax 在 Triton 无原生对应，需 tl.dot+loop-carried
  maximum/exp 复刻）；② 窗口分组重排是 host 侧 layout 变换，kernel 无对应。
- 注意力是 Cube 高价值算子族，卡片价值密度高。
- 判走 **P2.5 降级①编译门+②正确性门**（非 P4 豁免）。

## v1.5 CANN 价值归责（L0~L3，跨算子横切）

| 层 | CANN 对 CrossformerAttention 的贡献 | 本地能否替代 | 本例落点 |
|---|---|---|---|
| **L0 架构事实/根因** | 910b 生产路径=arch35（DAV_3510）regbase `KERNEL_MIX_AIC_1_2`，Triton sync 对不上 CANN 的根因 | ❌ | arch35 `_tiling_basic.cpp` GetTilingKey/IsCapable(=DAV_3510) → 确认主档位来源 |
| **L1 结构性解锁（最高价值）** | CANN 默认档 **S1/S2=128/128**、D 档位阶梯 AlignUp(D,64)→{64,128,192,256}/768——激活「更大 tile 是可行目标配置」，而非在 64×128 打转 | ⚠️ profiling 走不到 | `CalcS1S2BasicBlock`/`CalcDBasicBlock`（arch35）→ 驱动 BLOCK_D 随 head_dim 缩放 |
| **L2 负面边界** | CANN 128/128 靠 L1 3-buffer + D-workspace，Triton 复刻会 **UB 溢出** → 64/128 是可移植近邻；num_stages 深流水不可移植（C-006） | ✅ 可本地试 | UB 预算硬算（skeleton 头部注释） |
| **L3 事后佐证** | flash 在线 softmax 块末 α 补偿、p 回 cast 输入 dtype → CANN FA设计介绍.md §1 步 2~4 背书 | ✅ 本地优先 | extraction §6.x |

**鲁棒结论**：CANN 对 crossformer 的价值 = L1（D-tier + 128/128 大档位解锁）+ L2（UB 边界），不在数值杠杆。
原提取已实现内核算法等价；v2 在此基础上补齐 host 档位阶梯这一最锋利项。

## 与 level4 目标的语义差（P0 动作 5：benchmark diff）

level4 定义见 `src/flash_attention_score/level4_target/CrossformerAttention/{definition.json,workload.jsonl}`。
golden = torch 实现：`to_heads → qh*=scale → attn=qh·khᵀ(+pos_bias)(+mask) → softmax(-1) → ·vh → permute back`。

| 维度 | level4 CrossformerAttention | CANN FAS 参考 | 差距定性 |
|---|---|---|---|
| 接口 | run(x, group_size, q/k/v(opt), num_heads, mask, scale, pos_bias, feature_shape) | 16 输入+多属性 | level4 是真子集；分组属 host layout |
| 分组语义 | feature_shape 时按 (gh,gw) 空间窗口分组；否则单窗口 | FAS 无分组 | **缺口**：窗口重排 CANN 无现成实现 |
| 加性偏置 | pos_bias | `pse`：score=scale*(pse+Q·Kᵀ) | 语义同构 |
| 掩码 | mask（加性数值相加） | `atten_mask`（布尔 select 遮蔽→−inf） | **差异**：以 level4 加性为准 |
| 缩放 | scale（缺省 head_dim^-0.5） | scale_value（默认 1.0） | 同构 |
| softmax | torch softmax(-1) fp32 重算 | FlashSoftmaxV2 在线重构 | 算法不同但数学等价 |
| dtype | f16/bf16/f32 | FAS 同+fp8 族 | 兼容 |
| head_dim | C/n_heads，可达 1024 | arch35 D-tier{64,128,192,256,768} | level4 大 D 需 Triton BLOCK_D 随行（v2 修原硬编码 128 缺陷） |
| 布局 | 2D (N,C) or 3D；feature (H·W,C) | ND↔BSH/SBH/BNSD/TND | level4 固定 ND |

## 材料归集表（按 v1.5 优先级=host tiling 阶梯第一）

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 承载详设章节 |
|---|---|---|---|---|
| **host 档位阶梯（v1.5 第一顺位）** | `src/flash_attention_score/op_host/arch35/flash_attention_score_tiling_basic.cpp` | ✅ | `CalcS1S2BasicBlock`(:79) 默认 128/128、D>256→64/128；`CalcDBasicBlock`(:51) AlignUp(D+rope,64)→{64,128,192,256}/768；`IsCapable`(:170)=DAV_3510 | §4.1、§4.1.3 |
| level4 目标定义 | `src/flash_attention_score/level4_target/CrossformerAttention/definition.json` | ✅ | `reference`=torch run()；50 case workload | §1.1、§2 |
| 参考公式 | `src/flash_attention_score/docs/aclnnFlashAttentionScore.md:23` | ✅ | `attention_out = Softmax(Mask(scale*(pse+Q·Kᵀ), atten_mask))·V` | §1.1 |
| 设计文档（flash 算法） | `docs/FA算子设计介绍.md` | ✅ | §1(1~4) flash 4 步；§3 Tiling(:30)；§4 KV 流水 | §5、§6 |
| 原型定义 | `op_host/flash_attention_score_def.cpp` | ✅ | pse/mask/scale_value(:157)/head_num(:161) | §2 |
| kernel 入口 | `op_kernel/flash_attention_score.cpp`、`_apt.cpp` | ✅ | 模板分支、AIC/AIV dispatch | §4.2 |
| kernel 主实现 | `op_kernel/arch32/*.h`（s1s2_bn2gs1 等） | ✅ | Cube MM+FlashSoftmaxV2 重构 | §5、§6 |
| 窗口/分组结构 | `src/swin_attention_score_quant/` | ⚠️ | 量化 NZ/Transpose，无清晰窗口算法层 | §1.1 |
| 测试标杆 | level4 `definition.json reference` | ✅ | golden 单标杆 | §测试设计 |

## 缺省项与注意点（P1/P2 必须处理）

1. **窗口分组 = host 侧 layout 变换，kernel 内不可达**：wrapper 用 `idx`/`inv` gather 把 x/q/k/v 重排为窗口连续
   （C-001 无 `%`，div/mod 用整除化解且仅 wrapper）；kernel 保持连续仿射 load。
2. **level4 mask 加性数值、FAS atten_mask 布尔 select**：以 level4 加性为准。
3. **feature+mask/pos_bias reference 形状不一致缺陷**：本卡以窗口局部坐标切片为契约（写卡 §2/§1.1）。
4. **Scale 施加位置**：level4 预乘 qh，FAS score 后乘；Triton 统一 QKᵀ 后乘（数学等价）。
5. **大窗（G 达 3721/7921）必走 flash 重构**（key 分块在线 softmax），禁止整窗 QKᵀ。
6. **precision**：fp32 输入 tl.dot 须 `input_precision="ieee"`；f16/bf16 attn 中间 fp32 累加。
7. **D 轴（v2 新增）**：head_dim 可 >128（单头大 C）；原骨架 BLOCK_D=128 硬编码会算错，
   v2 改 BLOCK_D=≥head_dim 最小 2 幂；大 D 触发 UB 溢出为固有边界（p25 注册）。
8. **网络约束**：gitcode fetch 不可达（代理 timeout 静默坑），材料源固定在本地克隆 @d28d8f99。