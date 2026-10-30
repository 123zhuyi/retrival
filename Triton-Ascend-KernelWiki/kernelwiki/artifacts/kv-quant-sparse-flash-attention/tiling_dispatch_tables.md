# kv_quant_sparse_flash_attention - tiling 与 dispatch 分析

来源：`attention/kv_quant_sparse_flash_attention/`（commit `d28d8f9…`）。KV Quant SFA 是 Sparse FlashAttention 的 KV 量化版本（Per-Token-Head-Tile-128 量化），仅推理场景（A2/A3 推理，无训练），三架构（arch32 + arch35 950）。

## A. 架构分发

`def.cpp` L92-93 / L151 注册 AICore config：

| 架构 | SocVersion | config | kernel 入口 | 说明 |
|---|---|---|---|---|
| Atlas A2 推理 | `ascend910b` L92 | `kv_quant_sparse_flash_attention` | `kv_quant_sparse_flash_attention.cpp`（arch32 | FP16/BF16+INT8 |
| Atlas A3 推理 | `ascend910_93` L93 | 同上 | 同上 | 同 A2 |
| Ascend 950PR/950DT | `ascend950` L151 | 同上（arch35 单独 config） | `kv_quant_sparse_flash_attention.cpp`（arch35 分支） | BF16+FP8_E4M3FN/HIFLOAT8 |

> 仅推理场景（A2/A3 推理），无训练支持。

## B. 模板参数与 dispatch

`template_tiling_key.h` L31-37 定义 4 个模板参数：

| 参数 | 类型 | 取值 | 说明 |
|---|---|---|---|
| FLASH_DECODE | bool | 0/1 | FlashDecode 使能 |
| LAYOUT_T | uint | 0:BSND / 1:TND | query layout |
| KV_LAYOUT_T | uint | 0:BSND / 1:TND / 2:PA_BSND | KV layout（PA 场景） |
| TEMPLATE_MODE | uint | 0:C_TEMPLATE / 1:V_TEMPLATE | Cube/Vector 模板选择 |

5 种合法组合（L42-55）：
- `FLASH_DECODE=0, LAYOUT=BSND, KV_LAYOUT=BSND/PA_BSND, TEMPLATE=V_TEMPLATE`
- `FLASH_DECODE=0, LAYOUT=TND, KV_LAYOUT=TND/PA_BSND, TEMPLATE=V_TEMPLATE`

kernel 入口（`kv_quant_sparse_flash_attention.cpp` L37-73）按 `__CCE_AICORE__` 宏分发：
- `__CCE_AICORE__==310`（arch35/950）：BF16+FP8_E4M3FN/HIFLOAT8（L49-60）
- 其他（arch32/A2）：FP16/BF16+INT8（L61-72）

## C. 核间切分

| 项 | 结论 | 来源 |
|---|---|---|
| 逻辑轴 | B、N2、S1、S2（sparse 取数） | 同 FA/IFA |
| 切分方式 | 沿 B/N2/S1 轴外切分核，按 aicNum | `tiling.cpp` |
| 稀疏离散取数 | sparse_indices 驱动，每行取前 sparse_size 个 block | README L42-43 |

## D. 核内处理

| 阶段 | 说明 |
|---|---|
| 1. sparse KV gather | sparse_indices 查表 → 离散 load KV |
| 2. KV dequant | Per-Token-Head-Tile-128 反量化（INT8/FP8 → FP32） |
| 3. Bmm1 QK | Q·K^T（MLA-absorb：D=576 nope+rope） |
| 4. mask + scale | sparse_mode rightDownCausal |
| 5. FlashSoftmax | 在线 max/sum/exp rescale |
| 6. Bmm2 PV | P·V（V 反量化后） |
| 7. rescale + writeback | ÷softmax_sum → output |

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| sparse_indices | `tl.load(sparse_indices)` → `tl.load(KV + idx * block_size, ...)` | 离散取数 |
| KV dequant per-tile | `cast(KV_int8, fp32) * dequant_scale` | tile 128 对齐 |
| MLA-absorb Q/K D=576/656 | `tl.dot([q_nope\|q_rope], [k_nope\|k_rope\|scale].T)` | 拼接 D 维 |
| 4 模板参数 | `if constexpr` / 独立 kernel | 不迁移 key 本身 |
| Cube/Vector 模板选择 | `@triton.jit` 统一 kernel | 按 arch 选 |
| C_TEMPLATE/V_TEMPLATE | 单 kernel 内分支 | 编译器调度 |
| sparse_mode=3 mask | `tl.where` rightDownCausal | 可迁移 |