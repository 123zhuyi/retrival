import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    concat_and_cast_mha_k —— DeepSeek 系 MLA 模型 MHA（非吸收）前向中的
    K 拼接算子（SGLang Triton kernel 的 golden reference）。

    算子出处：
      sglang-project/sglang
      python/sglang/kernels/ops/kvcache/cache_ops.py
        - concat_and_cast_mha_k_kernel（Triton kernel 本体）
        - concat_and_cast_mha_k_triton（launch 包装，含形状 assert）
      调用方：python/sglang/srt/models/deepseek_common/
              attention_forward_methods/forward_mha.py 的 _concat_and_cast_mha_k。

    数学语义（逐 token、逐 head，无任何算术运算，纯数据搬运）：
      输入  k_nope: [T, H, nope_dim]    —— kv_b_proj 出的 K 无位置分量
      输入  k_pe:   [T, 1, rope_dim]    —— RoPE 后的共享位置分量（所有 head 共用）
      输出  k:      [T, H, nope_dim + rope_dim]
        k[t, h, :nope_dim]  = k_nope[t, h, :]
        k[t, h, nope_dim:]  = k_pe[t, 0, :]        （沿 head 维广播）
      与 SGLang 非 kernel 回退路径逐位一致：
        k = k_nope.new_empty(T, H, nope_dim + rope_dim)
        k[..., :nope_dim] = k_nope
        k[..., nope_dim:] = k_pe

    dtype 约定（"cast" 的含义）：
      kernel 在 store 时做隐式 dtype 转换（load 源 dtype -> store 目标 dtype），
      等价于拼接完成后一次 .to(out_dtype)（同为 round-to-nearest-even）。
      - 主路径：out_dtype == 输入 dtype（bf16/fp16，恒等转换）；
      - fa3 后端 + FP8 KV cache 路径：out_dtype = float8_e4m3fn。
      因无累加/约减，不需要 fp32 中间计算级；唯一舍入发生在输出 cast。

    与 kernel 的已知差异点：
      1. 原 Triton kernel 用 tl.arange(0, H / nope_dim / rope_dim)，要求三者均为
         2 的幂，否则编译失败（GLM-5 nope_dim=192 即触发，sglang issue #25672）；
         此时 sglang 回退到 torch 切片赋值。本 golden 实现的是广义语义（torch
         回退路径），对任意 H/nope_dim/rope_dim 均成立，case 5 专门覆盖该差异。
      2. kernel 以 stride 传参支持非连续输入视图；golden 的切片赋值同样
         对任意内存布局成立，无需显式处理。
      3. workspace / enable_pdl / grid 调度参数属实现细节，不纳入语义。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, k_nope, k_pe, out_dtype=None):
        # ---- 形状自洽校验（对齐 concat_and_cast_mha_k_triton 的 assert）----
        assert k_nope.dim() == 3 and k_pe.dim() == 3, \
            f"k_nope/k_pe 应为 3D, got {k_nope.shape=}, {k_pe.shape=}"
        T, H, nope_dim = k_nope.shape
        T2, H_pe, rope_dim = k_pe.shape
        assert T == T2, f"token 数不一致: {k_nope.shape=}, {k_pe.shape=}"
        assert H_pe == 1, f"k_pe 的 head 维必须为 1（共享广播）, got {k_pe.shape=}"
        assert nope_dim > 0 and rope_dim > 0

        if out_dtype is None:
            # 主路径：与输入同 dtype（恒等 cast）
            out_dtype = k_nope.dtype

        # ---- 语义实现：与 sglang torch 回退路径逐位一致 ----
        k = torch.empty(T, H, nope_dim + rope_dim,
                        dtype=out_dtype, device=k_nope.device)
        k[..., :nope_dim] = k_nope
        k[..., nope_dim:] = k_pe  # [T,1,rope_dim] 沿 head 维广播
        return k


def get_input_groups():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "1_ConcatAndCastMhaK.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    def random_tensor(shape, dtype):
        # 标准分布：50% 均匀 U[-5,5] + 50% 正态 N(mu, sigma)，mu/sigma 先随机采样。
        # 本算子无物理约束特例：k_nope 是 kv_b_proj 输出、k_pe 是 RoPE 旋转
        # 后的有界向量，标准分布即可代表真实取值范围。
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 可复现性：每个 case 固定种子，两次调用本函数逐张量一致
        torch.manual_seed(3407 + case_idx)

        inputs = case["inputs"]
        nope_info = next(i for i in inputs if i["name"] == "k_nope")
        pe_info = next(i for i in inputs if i["name"] == "k_pe")
        out_info = next(i for i in inputs if i["name"] == "out_dtype")

        dtype = dtype_map[nope_info["dtype"]]
        k_nope = random_tensor(nope_info["shape"], dtype)
        k_pe = random_tensor(pe_info["shape"], dtype)

        out_dtype = None if out_info["value"] == "same" \
            else dtype_map[out_info["value"]]

        # JSON 形状与算子语义自洽校验
        T, H, nope_dim = nope_info["shape"]
        T2, H_pe, rope_dim = pe_info["shape"]
        assert T == T2 and H_pe == 1, f"case {case_idx} 形状不自洽"

        input_groups.append([k_nope, k_pe, out_dtype])
    return input_groups


def get_init_inputs():
    return []