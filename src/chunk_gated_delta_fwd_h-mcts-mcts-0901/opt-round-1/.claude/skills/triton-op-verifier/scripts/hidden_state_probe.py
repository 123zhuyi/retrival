#!/usr/bin/env python3
"""隐藏状态对齐探针 —— 由 verify.py 在精度比对之前调用。

抓的是这一类失败：**参考实现的输出依赖不由输入张量决定的内部状态**
（随机初始化权重 / buffer / 跨调用缓存），**而生成实现没有逐位复刻该状态的产生过程**。
这类失败与 kernel 实现无关，单独修 kernel 永远修不好，且误差形态与"算法全错"
无法区分，因此必须在精度比对之前单独点名。

四层结构，严格区分硬闸门与软诊断（软诊断不判失败，避免误杀）：
  P0  触发判定  深度扫描后无隐藏状态 → SKIP（对无状态算子零开销、零风险）
  P1  同名对拍  两侧同名 parameter/buffer 必须逐位一致          —— 硬失败
  P2  集合对拍  fw 的每个隐藏张量需在 im 中找到逐位相等的对应项  —— 软诊断
  P3  确定性    同输入连跑两次，各自输出必须逐位一致            —— 软诊断

只有 P1 会抛异常。P1 比较的是"两侧都存在的同名项"，这类项按定义就应相等，
结构上不会误判；命名或存储布局不同是合法的实现自由，只能出诊断。

两条实现要点（否则探针会漏检或误判）：
  - 触发判定不能用全文文本匹配。benchmark 的结构是「Model 类 + get_input_groups()」，
    而输入构造函数里必然有 torch.randn，全文匹配会把无状态算子全部误判为有状态。
    scan_reference() 因此只解析 Model 类体。
  - named_parameters() 不足以发现隐藏状态。PyTorch 只登记 `self.x = Module` 这种
    直接赋值，不会递归进 dict/list，因此把 nn.Linear 存进普通 dict 的参考实现
    在 forward 之后仍返回 0 项。_deep_collect() 因此递归遍历实例 __dict__
    与生成侧的模块级全局（常见于把权重放在模块级 _WEIGHT_CACHE 的实现）。
"""
from __future__ import annotations

import ast
import sys


# ---------------------------------------------------------------- AST 预筛
RNG_MODULES = {
    "Linear", "Bilinear", "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "Embedding", "EmbeddingBag", "GRU", "LSTM", "RNN",
    "GRUCell", "LSTMCell", "RNNCell", "MultiheadAttention",
    "Transformer", "TransformerEncoderLayer", "TransformerDecoderLayer",
}
DET_MODULES = {
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "RMSNorm",
}
RNG_CALLS = {
    "randn", "rand", "randint", "randperm", "bernoulli", "multinomial",
    "normal", "randn_like", "rand_like", "randint_like", "dropout",
    "dropout1d", "dropout2d", "dropout3d",
}
SEED_CALLS = {"manual_seed", "set_rng_state", "Generator"}
INIT_PREFIXES = ("kaiming_", "xavier_", "normal_", "uniform_",
                 "trunc_normal_", "orthogonal_", "sparse_")


def _fname(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def scan_reference(path: str) -> dict:
    """AST 预筛：**只扫 Model 类体**，不看 get_input_groups 等输入构造函数。

    这一限定是必须的：全文匹配会把 `get_input_groups` 里的 `torch.randn`
    当成隐藏状态，把无状态算子全部误判为有状态。

    返回 {"tier": RNG|DET|NONE, "hits": {...}}。
    tier 只用于给 codegen 提示，**不作为 verify 判定依据**（判定用运行时深度扫描）。
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8-sig").read())
    except Exception as e:
        return {"tier": "ERR", "hits": {}, "error": str(e)}

    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "Model"), None)
    if cls is None:
        return {"tier": "ERR", "hits": {}, "error": "no Model class"}

    hits: dict[str, set] = {}

    def add(k, v):
        hits.setdefault(k, set()).add(v)

    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        fn = _fname(node.func)
        if fn in RNG_MODULES:
            add("rng_module", fn)
        elif fn in DET_MODULES:
            add("det_module", fn)
        elif fn in RNG_CALLS:
            add("rng_call", fn)
        elif fn in SEED_CALLS:
            add("seed_call", fn)
        elif fn == "Parameter":
            add("parameter", fn)
        elif fn == "register_buffer":
            add("buffer", fn)
        elif fn and fn.startswith(INIT_PREFIXES):
            add("init_fn", fn)

    hits = {k: sorted(v) for k, v in hits.items()}
    if hits.keys() & {"rng_module", "rng_call", "seed_call", "parameter", "init_fn"}:
        tier = "RNG"
    elif hits.keys() & {"det_module", "buffer"}:
        tier = "DET"
    else:
        tier = "NONE"
    return {"tier": tier, "hits": hits}


# ------------------------------------------------------------- 运行时探针
class HiddenStateMismatch(AssertionError):
    """隐藏状态（权重 / buffer）不一致 —— 与 kernel 实现无关的失败。"""


_MAX_DEPTH = 4
_MAX_ITEMS = 512


def _deep_collect(obj, prefix="", depth=0, seen=None, out=None, arities=None):
    """递归收集张量：nn.Module 的 parameters/buffers + 实例 __dict__ 里的容器。

    覆盖「把 nn.Linear 存进普通 dict / list」这类写法 ——
    PyTorch 不会把容器里的 Module 注册为子模块，这类状态不会出现在
    named_parameters() 里，即使 forward 之后也不会。

    arities: 可选的输出字典，记录 {序列容器前缀: 元素个数}。P1 用它判断两侧
    容器结构是否一致 —— 元数不同时位置下标不可比，必须降级到 P2 按值匹配。
    """
    import torch

    if out is None:
        out, seen = {}, set()
    if arities is None:
        arities = {}
    if depth > _MAX_DEPTH or len(out) >= _MAX_ITEMS or id(obj) in seen:
        return out
    seen.add(id(obj))

    if torch.is_tensor(obj):
        out[prefix or "<tensor>"] = obj
        return out

    if isinstance(obj, torch.nn.Module):
        for n, t in list(obj.named_parameters()) + list(obj.named_buffers()):
            out[f"{prefix}.{n}" if prefix else n] = t
        for k, v in vars(obj).items():
            if k.startswith("_") and k not in ("_cache",):
                continue                                # 跳过 PyTorch 内部簿记字段
            _deep_collect(v, f"{prefix}.{k}" if prefix else k, depth + 1, seen, out, arities)
        return out

    if isinstance(obj, dict):
        for k, v in list(obj.items())[:_MAX_ITEMS]:
            _deep_collect(v, f"{prefix}[{k!r}]", depth + 1, seen, out, arities)
    elif isinstance(obj, (list, tuple, set)):
        arities[prefix] = len(obj)
        for i, v in enumerate(list(obj)[:_MAX_ITEMS]):
            _deep_collect(v, f"{prefix}[{i}]", depth + 1, seen, out, arities)
    return out


def _positional_conflict_prefixes(fw_arities, im_arities):
    """找出两侧元数不同的序列容器前缀。

    这些容器下的位置下标在两侧指向不同语义的张量，不能按同名对拍。
    典型场景：baseline 把权重装成 (conv_q, conv_k, conv_v, gamma) 4 元组，
    实现因 AST 禁止 nn.Conv2d 而拆成 (w_q, b_q, w_k, b_k, w_v, b_v, gamma)
    7 元组 —— 此时 [3] 在 baseline 是 gamma(1,)、在实现是 b_k(c8,)，
    形状不等会被 P1 误判为隐藏状态不一致。
    """
    return {
        p for p in set(fw_arities) & set(im_arities)
        if fw_arities[p] != im_arities[p]
    }


def _under_conflicting_container(name, conflicts):
    """判断某个条目名是否落在元数冲突的容器内（即其位置下标不可比）。"""
    return any(name.startswith(p + "[") for p in conflicts)


def _module_globals(mod_obj, arities=None):
    """扫生成实现的模块级全局（很多实现把权重放在模块级 _WEIGHT_CACHE）。"""
    import torch

    out = {}
    m = sys.modules.get(getattr(type(mod_obj), "__module__", ""), None)
    if m is None:
        return out
    for k, v in list(vars(m).items()):
        if k.startswith("__") or callable(v) or isinstance(v, type):
            continue
        if torch.is_tensor(v) or isinstance(v, (dict, list, tuple, torch.nn.Module)):
            _deep_collect(v, f"<module>.{k}", 0, set(), out, arities)
    return out


def _equal_maybe_T(a, b):
    """逐位相等；2D 张量额外允许转置相等（生成侧可能按 kernel 布局预转置存权重）。"""
    import torch

    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape == b.shape and torch.equal(a, b):
        return True
    if a.dim() == 2 and b.dim() == 2 and a.shape == b.T.shape and torch.equal(a, b.T):
        return True
    return False


def probe(model, model_new, sample_inputs=None, run_fn=None, warmup=True,
          check_determinism=False):
    """返回诊断 dict；仅 P1 会抛 HiddenStateMismatch。

    model / model_new : 已实例化的 nn.Module
    sample_inputs     : 一组输入。用于 ① 物化懒构造的隐藏状态 ② P3 确定性检查
    run_fn            : 可选 run_fn(mod, inputs) -> 输出；默认 mod(*inputs)
    warmup            : 是否先各跑一次 forward（懒构造权重的算子必须为 True）
    check_determinism : 是否执行 P3（每侧额外 2 次 forward，默认关闭以控开销）

    注意：warmup 会吞掉 forward 抛出的异常 —— 这是刻意设计。生成实现编译失败时，
    探针仍应能对拍参考侧已物化的隐藏状态，从而在「kernel 都没跑起来」的情况下
    照样给出权重层面的归因。
    """
    import torch

    call = run_fn or (lambda m, x: m(*x))

    # ---- warmup：懒构造的权重要跑一次 forward 才存在
    if warmup and sample_inputs is not None:
        for mod in (model, model_new):
            try:
                call(mod, sample_inputs)
            except Exception:
                pass                                    # 跑不通交给主流程报错，探针不接管

    fw_arities, im_arities = {}, {}
    fw = _deep_collect(model, arities=fw_arities)
    im = _deep_collect(model_new, arities=im_arities)
    im.update(_module_globals(model_new, im_arities))

    # 两侧元数不同的序列容器：其位置下标不可比，P1 跳过，交给 P2 按值匹配
    conflicts = _positional_conflict_prefixes(fw_arities, im_arities)

    # ---- P0 触发判定
    if not fw:
        return {"status": "SKIP", "reason": "参考实现不持有任何隐藏状态张量",
                "fw_state": 0, "im_state": len(im)}

    diag = {"status": "PASS", "fw_state": len(fw), "im_state": len(im),
            "compared": 0, "matched": 0, "warnings": [],
            "skipped_positional": 0, "conflicting_containers": sorted(conflicts)}

    # ---- P1 同名逐位对拍（硬闸门）
    # 只比两侧都存在、且不落在"元数冲突容器"内的同名项。后者的位置下标在两侧
    # 指向不同语义的张量，按名对拍必然假阳（见 _positional_conflict_prefixes）。
    skipped_positional = 0
    for name in sorted(set(fw) & set(im)):
        if _under_conflicting_container(name, conflicts):
            skipped_positional += 1
            continue
        a, b = fw[name], im[name]
        if a.shape != b.shape:
            raise HiddenStateMismatch(
                f"隐藏状态 '{name}' 形状不一致: framework={tuple(a.shape)} impl={tuple(b.shape)}")
        if not _equal_maybe_T(a, b):
            md = (a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item()
            raise HiddenStateMismatch(_ADVICE.format(name=name, md=md))
        diag["compared"] += 1

    diag["skipped_positional"] = skipped_positional
    if skipped_positional:
        diag["warnings"].append(
            f"{skipped_positional} 项因容器元数不一致跳过按名对拍"
            f"（冲突容器: {sorted(conflicts)}），已降级到 P2 按值匹配"
        )

    # ---- P2 集合对拍（软诊断：按值找对应项，不要求同名）
    if diag["compared"] == 0:
        im_list = list(im.values())
        unmatched = []
        for name, a in fw.items():
            if any(_equal_maybe_T(a, b) for b in im_list
                   if b.shape == a.shape or (a.dim() == 2 and b.dim() == 2)):
                diag["matched"] += 1
            else:
                unmatched.append((name, tuple(a.shape)))
        if unmatched:
            diag["status"] = "WARN"
            diag["warnings"].append(
                "HIDDEN_STATE_VALUE_MISMATCH: 参考实现的 %d/%d 个隐藏状态张量"
                "在生成实现中找不到逐位相等的对应项 —— 首要怀疑权重复现方式。"
                "未匹配项(前5): %s" % (len(unmatched), len(fw), unmatched[:5]))
        if not im:
            diag["warnings"].append(
                "HIDDEN_STATE_UNEXPOSED: 生成实现未暴露任何隐藏状态张量，无法对拍。"
                "建议生成侧用 register_parameter/register_buffer 暴露复刻出的权重。")

    # ---- P3 确定性（同输入两次调用必须一致）
    if sample_inputs is not None and check_determinism:
        for tag, mod in (("framework", model), ("impl", model_new)):
            try:
                o1, o2 = call(mod, sample_inputs), call(mod, sample_inputs)
            except Exception as e:
                diag["warnings"].append(f"NONDET_CHECK_SKIPPED[{tag}]: {type(e).__name__}")
                continue
            o1 = o1 if isinstance(o1, (tuple, list)) else (o1,)
            o2 = o2 if isinstance(o2, (tuple, list)) else (o2,)
            for i, (x, y) in enumerate(zip(o1, o2)):
                if torch.is_tensor(x) and not torch.equal(x.cpu(), y.cpu()):
                    diag["status"] = "WARN"
                    diag["warnings"].append(
                        f"NONDETERMINISTIC[{tag}] 输出[{i}] 两次调用不一致 —— "
                        f"存在未 seed 的随机或跨调用状态，验证结果不可复现。")
    return diag


_ADVICE = (
    "隐藏状态 '{name}' 数值不一致 (max|diff|={md:.6e})。\n"
    "  这是与 kernel 实现无关的失败，先修它再看精度。常见原因：\n"
    "    ① 构造 device 不同 —— CPU 与 NPU 是两条独立 RNG 流，同 seed 结果完全不相关\n"
    "    ② 采样 dtype 不同 —— 先 fp32 采样再 .to(fp16) ≠ 直接在 fp16 上采样\n"
    "    ③ 构造顺序/次数不同 —— uniform_ 消耗 numel() 个随机数，错一步后面全错位\n"
    "    ④ 未做 RNG save/restore —— 污染同进程后续用例\n"
    "  修法：直接复刻参考实现的构造语句（连 .to(device=,dtype=) 在第几步都要一致），\n"
    "        禁止用等价数学公式重新实现初始化。"
)
