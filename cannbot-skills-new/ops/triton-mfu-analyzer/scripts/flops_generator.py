#!/usr/bin/env python3
"""flops.py 半自动生成器（原型）。

设计思路
--------
1. 静态 AST 扫描 model.py，识别 forward 里的 GEMM/Attention 模式。
2. 按算子名字 + 输入名 + 检测到的模式做分类（attention/ffn/matmul/linear/lstm/...）。
3. 为每类生成对应的 flops.py 模板代码。
4. 对复杂算子（sparse/window/rope/cu_seqlens/block_table）插入已知处理逻辑。
5. 生成后用 check_flops.py 对 cases 做交叉校验，人工确认边界 case。

当前覆盖（原型阶段）：
- 标准 MHA / FlashAttention（含 4 个 projection）
- GQA / MQA（repeat_interleave）
- 单层 Attention（无 projection）
- FFN / MatMul / Linear
- Window / BlockSparse / SparseFlashAttention（sparse 语义）
- PagedAttention（block_table）
- varlen（cu_seqlens / actual_seq_lengths）

未覆盖（需后续扩展）：
- 复杂的 einsum / 自定义 scan / state-space（KDA/Mamba）
- 量化算子的反量化/量化辅助矩阵
- 反向算子（FlashAttentionBwd 等）
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# AST 工具
# ---------------------------------------------------------------------------

def find_class_method(tree: ast.AST, cls_name: str, method_name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def get_call_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def get_full_attr(node) -> str:
    """把 torch.matmul / F.linear / nn.Linear 等还原成字符串。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return get_full_attr(node.value) + "." + node.attr
    return ""


# ---------------------------------------------------------------------------
# 模式检测
# ---------------------------------------------------------------------------

class ForwardPattern:
    def __init__(self):
        self.has_matmul = False
        self.has_linear = False          # F.linear / nn.Linear
        self.has_einsum = False
        self.has_bmm = False
        self.has_softmax = False
        self.has_causal = False
        self.has_window_size = False
        self.has_block_size = False
        self.has_block_table = False
        self.has_cu_seqlens = False
        self.has_actual_seq_lengths = False
        self.has_sparse_indices = False
        self.has_rope = False
        self.has_repeat_interleave = False
        self.input_names: Set[str] = set()
        self.attr_names: Set[str] = set()


def detect_patterns(py_text: str, tree: ast.AST) -> ForwardPattern:
    p = ForwardPattern()
    forward = find_class_method(tree, "Model", "forward")
    if forward is None:
        return p

    for node in ast.walk(forward):
        if isinstance(node, ast.Call):
            name = get_call_name(node)
            full = get_full_attr(node.func)
            if name in ("matmul",):
                p.has_matmul = True
            if name in ("linear",) or "Linear" in full:
                p.has_linear = True
            if name == "einsum":
                p.has_einsum = True
            if name == "bmm":
                p.has_bmm = True
            if name == "softmax":
                p.has_softmax = True
            if name == "repeat_interleave":
                p.has_repeat_interleave = True
            if name == "masked_fill":
                p.has_causal = True

    # 文本级特征（比 AST 更鲁棒）
    low = py_text.lower()
    p.has_rope = "rope" in low
    p.has_window_size = "window_size" in py_text
    p.has_block_size = "block_size" in py_text and "sparse" in low
    p.has_block_table = "block_table" in py_text
    p.has_cu_seqlens = "cu_seqlens" in py_text
    p.has_actual_seq_lengths = "actual_seq_lengths" in py_text
    p.has_sparse_indices = "sparse_indices" in py_text

    # 从 forward 签名提取输入名
    if forward and forward.args and forward.args.args:
        for arg in forward.args.args:
            if isinstance(arg, ast.arg):
                p.input_names.add(arg.arg)

    return p


# ---------------------------------------------------------------------------
# 算子分类
# ---------------------------------------------------------------------------

def classify(op_name: str, p: ForwardPattern) -> str:
    low = op_name.lower()

    if "lstm" in low:
        return "lstm"

    if "ffn" in low or "swiglu" in low or "mlp" in low:
        return "ffn"

    # 判定是否 attention：名字是弱信号（prepare_op_dir 会把文件改名成 model.py，
    # 算子名信息丢失），因此同时看代码特征——softmax 作用在 matmul 结果上是
    # attention 的充分特征。
    looks_like_attention = (
        "attention" in low
        or "attn" in low
        or (p.has_softmax and (p.has_matmul or p.has_einsum or p.has_bmm))
    )

    if looks_like_attention:
        if p.has_sparse_indices or "sparseflash" in low:
            return "sparse_attention"
        if p.has_block_table or "paged" in low:
            return "paged_attention"
        if p.has_window_size or "window" in low:
            return "window_attention"
        if p.has_block_size or "blocksparse" in low:
            return "block_sparse_attention"
        if p.has_cu_seqlens or p.has_actual_seq_lengths or "varlen" in low:
            return "varlen_attention"
        if p.has_repeat_interleave or "grouped" in low or "multiquery" in low:
            return "grouped_attention"
        return "dense_attention"

    if p.has_matmul or p.has_bmm or p.has_einsum:
        return "matmul"

    if p.has_linear:
        return "linear"

    return "generic"


# ---------------------------------------------------------------------------
# flops.py 模板生成
# ---------------------------------------------------------------------------

def _specs_fn() -> str:
    return """def _specs(case):
    return {item["name"]: item for item in case["inputs"]}
"""


def _prod_fn() -> str:
    return """def _prod(shape):
    r = 1
    for d in shape:
        r *= d
    return r
"""


def template_dense_attention(op_name: str, p: ForwardPattern) -> str:
    """标准 MHA/FlashAttention：4 个 projection + attention + out projection。"""
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: dense_attention (q/k/v/out projection + attention).
Please review and adjust for causal/variable-length cases.
"""

{_specs_fn()}
{_prod_fn()}

def flops_parts(case) -> dict:
    s = _specs(case)
    # Common input names: x, query, q; n_heads, head_dim, causal
    x = s.get("x") or s.get("query") or s.get("q")
    if x is None:
        raise ValueError("Cannot find main input (x/query/q)")
    shape = x["shape"]
    *lead, seq_len, d_model = shape
    batch = _prod(lead)
    n_heads = s.get("n_heads", {{}}).get("value", s.get("num_heads", {{}}).get("value", 1))
    head_dim = d_model // n_heads
    causal = s.get("causal", {{}}).get("value", s.get("is_causal", {{}}).get("value", False))

    # 4 projections [B*S, D] @ [D, D]
    proj = 4.0 * 2.0 * batch * seq_len * d_model * d_model

    # attention: Q@K^T + P@V
    pairs = float(seq_len) * seq_len
    if causal:
        pairs = seq_len * (seq_len + 1) / 2.0
    attn = 2.0 * batch * n_heads * pairs * (head_dim + head_dim)

    return {{"proj": proj, "attn": attn}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))


KERNEL_PARTS = {{
    # "kernel_name": "proj",
    # "flash_attention_kernel": "attn",
}}
'''


def template_grouped_attention(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: grouped_attention (GQA/MQA with repeat_interleave).
"""

{_specs_fn()}
{_prod_fn()}

def flops_parts(case) -> dict:
    s = _specs(case)
    x = s.get("x") or s.get("query") or s.get("q")
    shape = x["shape"]
    *lead, seq_len, d_model = shape
    batch = _prod(lead)
    n_heads = s.get("n_heads", {{}}).get("value", 1)
    n_kv_heads = s.get("n_kv_heads", {{}}).get("value", n_heads)
    head_dim = d_model // n_heads
    g = n_heads // n_kv_heads
    causal = s.get("causal", {{}}).get("value", False)

    # q_proj: [B*S, D] -> [B*S, n_heads*head_dim]
    # k_proj/v_proj: [B*S, D] -> [B*S, n_kv_heads*head_dim]
    proj_q = 2.0 * batch * seq_len * d_model * (n_heads * head_dim)
    proj_kv = 2.0 * batch * seq_len * d_model * (n_kv_heads * head_dim) * 2.0
    out_proj = 2.0 * batch * seq_len * d_model * d_model
    proj = proj_q + proj_kv + out_proj

    pairs = float(seq_len) * seq_len
    if causal:
        pairs = seq_len * (seq_len + 1) / 2.0
    # after repeat_interleave, effective kv heads = n_heads
    attn = 2.0 * batch * n_heads * pairs * (head_dim + head_dim)

    return {{"proj": proj, "attn": attn}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))
'''


def template_sparse_attention(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: sparse_attention (SparseFlashAttention / sparse indices / actual_seq_lengths).
"""

{_specs_fn()}
{_prod_fn()}

def _cumsum(lengths, total):
    if lengths is None or len(lengths) == 0:
        return [total]
    if lengths[-1] >= total * 0.9:
        return list(lengths)
    return [sum(lengths[:i + 1]) for i in range(len(lengths))]


def _batch_of(cumsum_list, t_idx):
    b_idx = 0
    while b_idx < len(cumsum_list) and t_idx >= cumsum_list[b_idx]:
        b_idx += 1
    return min(b_idx, len(cumsum_list) - 1)


def flops_parts(case) -> dict:
    s = _specs(case)
    q = s["query"]
    k = s["key"]
    v = s["value"]
    si = s["sparse_indices"]
    qr = s.get("query_rope")
    kr = s.get("key_rope")

    layout_query = s.get("layout_query", {{}}).get("value", "BSND")
    layout_kv = s.get("layout_kv", {{}}).get("value", "BSND")
    block_size = s.get("sparse_block_size", {{}}).get("value", 1)

    if layout_query == "BSND":
        b, s1, n1, d_main = q["shape"]
        t1 = b * s1
        n2 = k["shape"][2] if layout_kv != "TND" else k["shape"][1]
    else:
        t1, n1, d_main = q["shape"]
        n2 = k["shape"][1] if layout_kv == "TND" else k["shape"][2]

    g = n1 // n2
    d_v = v["shape"][-1]
    topk = si["shape"][-1]
    has_rope = qr is not None and kr is not None
    d_rope = qr["shape"][-1] if has_rope else 0

    asq = s.get("actual_seq_lengths_query", {{}}).get("value") or [t1]
    if layout_kv == "PA_BSND":
        total_kv = k["shape"][0] * k["shape"][1]
    elif layout_kv == "BSND":
        total_kv = b * k["shape"][1]
    else:
        total_kv = k["shape"][0]
    ask = s.get("actual_seq_lengths_kv", {{}}).get("value") or [total_kv]

    asq_cumsum = _cumsum(asq, t1)
    ask_cumsum = _cumsum(ask, sum(ask) if ask else 0)

    total_sel = 0
    for i in range(t1):
        b_idx = _batch_of(asq_cumsum, i)
        cur_s2 = ask_cumsum[b_idx] - (ask_cumsum[b_idx - 1] if b_idx > 0 else 0)
        total_sel += min(topk * block_size, cur_s2)

    score_elems = total_sel * n2 * g
    flops = 2.0 * score_elems * (d_main + d_rope + d_v)
    return {{"attn": flops}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))
'''


def template_paged_attention(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: paged_attention (block_table / kv_cache / cache_seqlens).
"""

{_specs_fn()}
{_prod_fn()}

def flops_parts(case) -> dict:
    s = _specs(case)
    q = s["q"]
    kv_cache = s["kv_cache"]
    batch, query_length, n_heads, qk_dim = q["shape"]
    headdim_v = s.get("headdim_v", {{}}).get("value", qk_dim)
    page_size = kv_cache["shape"][1]
    num_pages = kv_cache["shape"][0]
    capacity = num_pages * page_size

    # cache_seqlens 可能是 attr（list/int）或 tensor（shape 已知，值未知）
    cs = s.get("cache_seqlens")
    if cs.get("type") == "attr":
        kv_lens = cs.get("value")
        if isinstance(kv_lens, int):
            kv_lens = [kv_lens] * batch
    else:
        # tensor: 无法从 shape 得知实际长度，保守取 capacity // batch
        kv_lens = [capacity // batch] * batch

    score_elems = sum(query_length * n_heads * min(kv_len, capacity // batch) for kv_len in kv_lens)
    flops = 2.0 * score_elems * (qk_dim + headdim_v)
    return {{"attn": flops}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))
'''


def template_window_attention(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: window_attention (window_size, local attention).
"""

{_specs_fn()}
{_prod_fn()}

def flops_parts(case) -> dict:
    s = _specs(case)
    x = s["x"]
    *lead, n, d_model = x["shape"]
    batch = _prod(lead)
    num_heads = s.get("num_heads", {{}}).get("value", 1)
    head_dim = d_model // num_heads
    window_size = s.get("window_size", {{}}).get("value", 7)

    # qkv projection + out projection
    proj_qkv = 2.0 * batch * n * d_model * (3 * d_model)
    out_proj = 2.0 * batch * n * d_model * d_model

    # window attention: each query token attends to ~window_size^2 tokens
    score_elems = batch * num_heads * n * (window_size * window_size)
    attn = 2.0 * score_elems * (head_dim + head_dim)

    return {{"proj": proj_qkv + out_proj, "attn": attn}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))
'''


def template_block_sparse_attention(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: block_sparse_attention (block_size, block mask).
"""

{_specs_fn()}
{_prod_fn()}

def flops_parts(case) -> dict:
    s = _specs(case)
    x = s.get("x")
    shape = x["shape"]
    *lead, seq_len, d_model = shape
    batch = _prod(lead)
    n_heads = s.get("n_heads", {{}}).get("value", 1)
    head_dim = d_model // n_heads
    block_size = s.get("block_size", {{}}).get("value", 1)

    # 4 projections
    proj = 4.0 * 2.0 * batch * seq_len * d_model * d_model

    # block sparse: only tokens in the same block attend to each other
    num_blocks = (seq_len + block_size - 1) // block_size
    # per block: block_size query tokens * block_size kv tokens
    score_elems = batch * n_heads * num_blocks * block_size * block_size
    attn = 2.0 * score_elems * (head_dim + head_dim)

    return {{"proj": proj, "attn": attn}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))
'''


def template_ffn(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: ffn (two matmul/linear with activation).
"""

{_specs_fn()}
{_prod_fn()}

def flops_parts(case) -> dict:
    s = _specs(case)
    x = s.get("x")
    shape = x["shape"]
    *lead, in_features = shape
    batch = _prod(lead)

    # Infer hidden dim from weight1 shape if available
    w1 = s.get("weight1") or s.get("w1")
    hidden = w1["shape"][-1] if w1 else in_features * 4

    # x @ W1
    gemm1 = 2.0 * batch * in_features * hidden
    # act(xW1) @ W2
    gemm2 = 2.0 * batch * hidden * in_features
    return {{"gemm": gemm1 + gemm2}}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))
'''


def template_matmul(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: matmul.
"""

{_specs_fn()}
{_prod_fn()}

def flops(case) -> float:
    s = _specs(case)
    a = s["x"] if "x" in s else s["a"] if "a" in s else s["input"]
    b = s["y"] if "y" in s else s["b"] if "b" in s else s["weight"]
    *batch_dims, m, k = a["shape"]
    n = b["shape"][-1]
    batch = _prod(batch_dims)
    return float(2.0 * batch * m * k * n)
'''


def template_linear(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: linear.
"""

{_specs_fn()}
{_prod_fn()}

def flops(case) -> float:
    s = _specs(case)
    x = s.get("x") or s.get("input")
    weight = s.get("weight")
    *batch_dims, in_features = x["shape"]
    out_features = weight["shape"][0]
    batch = _prod(batch_dims)
    return float(2.0 * batch * in_features * out_features)
'''


def template_lstm(op_name: str, p: ForwardPattern) -> str:
    return f'''"""Auto-generated flops.py for {op_name}.

Detected pattern: lstm.
"""

{_specs_fn()}
{_prod_fn()}

def flops(case) -> float:
    s = _specs(case)
    x = s.get("x")
    shape = x["shape"]
    # x shape: [seq_len, batch, input_size] or [batch, seq_len, input_size]
    batch_first = s.get("batch_first", {{}}).get("value", False)
    if batch_first:
        batch, seq_len, input_size = shape
    else:
        seq_len, batch, input_size = shape
    weight_ih = s.get("weight_ih_l0")
    four_hidden, _ = weight_ih["shape"]
    hidden_size = four_hidden // 4
    h0 = s.get("h_0")
    num_layers = h0["shape"][0]

    # per step: x @ W_ih^T + h @ W_hh^T
    # each has 4 * hidden_size * (input_size + hidden_size) MACs
    per_step = 4 * hidden_size * (input_size + hidden_size)
    return float(2.0 * batch * seq_len * per_step * num_layers)
'''


def template_generic(op_name: str, p: ForwardPattern) -> str:
    lines = [f'"""Auto-generated flops.py for {op_name}."""', ""]
    lines.append(_specs_fn())
    lines.append(_prod_fn())
    lines.append("def flops(case) -> float:")
    lines.append('    s = _specs(case)')
    lines.append('    # TODO: compute semantic FLOPs from model.py')
    lines.append('    return 0.0')
    return "\n".join(lines) + "\n"


TEMPLATE_MAP = {
    "dense_attention": template_dense_attention,
    "grouped_attention": template_grouped_attention,
    "sparse_attention": template_sparse_attention,
    "paged_attention": template_paged_attention,
    "window_attention": template_window_attention,
    "block_sparse_attention": template_block_sparse_attention,
    "varlen_attention": template_dense_attention,  # fallback
    "ffn": template_ffn,
    "matmul": template_matmul,
    "linear": template_linear,
    "lstm": template_lstm,
    "generic": template_generic,
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def generate_flops(model_path: Path, cases_path: Optional[Path] = None,
                   force_category: Optional[str] = None,
                   op_name: Optional[str] = None):
    py_text = model_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(py_text)
    except SyntaxError as e:
        sys.exit(f"[ERR] 无法解析 {model_path}: {e}")

    p = detect_patterns(py_text, tree)

    # 尝试用第一个 case 辅助判断输入名
    if cases_path and cases_path.is_file():
        try:
            with open(cases_path, encoding="utf-8-sig") as f:
                first = json.loads(next(l for l in f if l.strip()))
            names = {x["name"] for x in first.get("inputs", [])}
            p.input_names |= names
        except Exception:
            pass

    # 分类依据的名字：优先用显式 --op-name，其次用例文件名（prepare_op_dir 之后
    # model.py 已丢失算子名，用例文件名通常仍保留），最后才回退到 model 文件名。
    if op_name:
        name_hint = op_name
    elif cases_path is not None:
        name_hint = cases_path.stem.replace("_perf_cases", "")
    else:
        name_hint = model_path.stem

    category = force_category or classify(name_hint, p)
    template_fn = TEMPLATE_MAP.get(category, template_generic)
    return template_fn(name_hint, p), category


def main():
    ap = argparse.ArgumentParser(description="自动生成算子的 flops.py（原型）")
    ap.add_argument("--model", required=True, help="参考实现 model.py 路径")
    ap.add_argument("--cases", help="用例 json/jsonl 路径（可选，辅助输入名识别与分类）")
    ap.add_argument("--output", "-o", required=True, help="输出 flops.py 路径")
    ap.add_argument("--category", choices=sorted(TEMPLATE_MAP),
                    help="强制指定分类，覆盖自动检测")
    ap.add_argument("--op-name", help="算子名（用于分类提示，默认从用例/模型文件名推断）")
    args = ap.parse_args()

    model_path = Path(args.model)
    cases_path = Path(args.cases) if args.cases else None
    output_path = Path(args.output)

    code, category = generate_flops(model_path, cases_path,
                                    force_category=args.category,
                                    op_name=args.op_name)

    output_path.write_text(code, encoding="utf-8")
    print(f"[OK] 分类: {category}")
    print(f"[OK] 写入: {output_path}")
    print("[WARN] 这是自动生成的模板，请务必用 check_flops.py 对 cases 做校验并人工 review。")


if __name__ == "__main__":
    main()
