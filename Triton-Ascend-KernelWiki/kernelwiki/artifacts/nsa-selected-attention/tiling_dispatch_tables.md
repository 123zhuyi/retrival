# nsa_selected_attention - tiling 与 dispatch 分析

来源：`attention/nsa_selected_attention/`（commit `d28d8f9...`）。NSA 训练前向的第二阶段：根据 topKIndices gather 选中 KV 块并计算 attention。

## A. 架构与模板分发

`def.cpp` L88-89 注册 `ascend910b` + `ascend910_93`。产品支持（docs L5-12）：**仅 A2 训练** √；A2 推理 ×、A3 训练+推理 ×、950PR/950DT ×。与 `nsa_compress_attention_infer`（A2/A3 训练+推理）不同，本算子仅 A2 训练。

任务类型：kernel 入口分 CUBE/VECTOR 路径（`nsa_selected_attention.cpp` L21-53），`__DAV_C220_CUBE__` / `__DAV_C220_VEC__`。

模板参数：`NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>`（`.h` L38）。`INPUT_T` = half/bfloat16_t（编译期 ORIG_DTYPE_QUERY），`ATTEN_ENABLE` = false/true（mask 开关）。

tiling key（`tiling_general.cpp` L565-572）：
- 0 = 无 mask（sparseMode=0）
- 1 = causal mask（sparseMode=2）
语义：sparseMode=0 时 NO_MASK（全计算），sparseMode=2 时 LEFT_UP_CAUSAL。
空输入路径（`tiling.cpp` L56-71）：90=FP16、92=FLOAT、94=BF16。

## B. 核内/核间切分

- 逻辑轴：S1（query 序列长）→ 按 `coreNum = min(aivNum, t1Size)` 切分（`tiling_general.cpp` L579-593）。
- `headCoreNum`：每核处理 `s1NumPerHeadCore = ceil(t1Size/coreNum)` 行；`tailCoreNum = s1NumPerHeadCore * coreNum - t1Size` 处理少一行。
- 每核内再循环 N2（kvHead）和 S1 行。
- GQA：`gSize = N1/N2`，`gSize <= 32`。Q 布局 `[S1, N2, G, D]`（`.h` L241-245）。
- workspace：per-core 存 selectedK/V、P（softmax 前分数）、softmax 残差（`tiling_general.cpp` L637-650）。

## C. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| core id（S1 切分） | `tl.program_id` + `grid` | 每核一段 query 行 |
| topKIndices gather | `tl.load(topkIndices)` → `tl.load(k_ptr + idx*...)` | 间接 gather 选中 KV 块 |
| BMM1（Q·selected_K^T） | `tl.dot(q, tl.trans(k))` | GEMM |
| HandleMask（causal） | `tl.where` causal mask | sparseMode=2 |
| 在线 softmax | `tl.max/sum` + `exp` rescale | 含 scale |
| BMM2（P·selected_V） | `tl.dot(p, v)` | GEMM |
| softmaxMax/Sum 输出 | `tl.store(max_ptr, m_i); tl.store(sum_ptr, l_i)` | dim=8 每 G 组 |
| GQA gSize 分组 | Q 布局 `[S1, N2, G, D]` | 广播 |
| V gather 与 BMM1 重叠 | 不迁移 | CANN 流水 |
| MTE DataCopy / ND↔NZ | 不迁移 | `tl.load/store`+layout |