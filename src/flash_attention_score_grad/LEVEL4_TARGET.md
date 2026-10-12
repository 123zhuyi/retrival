# LEVEL4 目标算子对照说明（flash_attention_score_grad）

> 本目录材料来源：`src/ops-transformer/attention/flash_attention_score_grad`（gitcode.com/cann/ops-transformer @ d28d8f99a，CANN Open Software License 2.0），与 CANN 9.1.0 安装包
> `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/flash_attention_score_grad/` 内 device 源码同源；本仓多带 host 侧 op_host/tiling、op_api、docs、tests。

## 服务的 level4 算子

### ScaledDotProductAttentionBackward（level4_target/ScaledDotProductAttentionBackward/）
- **对应度**：三个 level4 算子中最高，接口近似 1:1：
  - 输入 `grad_output/query/key/value/attn_output/logsumexp/is_causal` ↔ FASGrad 的 `dy/q/k/v/attention_in(fwd out)/softmax_logsumexp/atten_mask(causal)`；
  - 输出 `grad_query/grad_key/grad_value` ↔ `dq/dk/dv`；
  - level4 参考用 fp32 重算 + logsumexp 重放（`attn_probs = exp(S·scale − lse)`），与 flash attention 反向的标准重构路径一致。
- **形状风险**：workload 极小（如 B=1,H=4,S_q=16,S_k=64,D=64），S_q≠S_k 且 S_q 很小，属于 FASGrad 模板里的 corner 分支；P1 抽取以主路径（s1s2_bn2gs1s2）为准。
- **注意**：level4 定义里的 `logsumexp` 输入在 golden 中被重算覆盖（自洽化），FASGrad 则信任传入的 lse——对标精度时以重算语义为准。

## 材料清单（P0 五类）
| 类别 | 位置 |
|---|---|
| kernel 入口 | `op_kernel/flash_attention_score_grad.cpp`、`flash_attention_score_grad_apt.cpp` |
| kernel 模板 | `op_kernel/arch22/*.h`（主实现 s1s2_bn2gs1s2*、pre/post/sfmg 子阶段） |
| 流水模块 | `op_kernel/arch22/basic_modules/{cube_modules,vec_modules}/` |
| host tiling | `op_host/`（948K，安装包缺失部分，本仓补齐） |
| aclnn API / 文档 / 测试 | `op_api/`、`docs/`、`examples/`、`tests/` |
