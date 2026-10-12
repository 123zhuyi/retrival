#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测四种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 有 @triton.jit kernel 定义但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
         （**含跨函数**：从 forward() 出发沿模块内函数调用做可达性分析，
           堵住"把计算挪到 forward 外的辅助函数里"的绕过路径）
  Type 4: 只有占位 kernel —— 被调用的 kernel 只做 load/store 搬运、无任何计算，
          纯粹为了让 AST 检测到"有 kernel 被调用"

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import logging
import os
import sys


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402

logger = logging.getLogger("triton_op_verifier.validate_triton_impl")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

# rng 状态操作：不产生任何数值计算，只影响后续随机初始化的取值序列。
# 部分任务的 baseline 在 forward 内用固定种子现场初始化权重、权重不从入参传入，
# 实现要对齐数值就必须复刻同样的 seed 设置。检查器自身在 MODULE_WIDE_FORBIDDEN_RNG_ATTRS
# 的注释里也认可这类 API 有合法用途（见文件上方）。真正的作弊由
# "是否借 nn.XXX/F.xxx 完成计算" 这一族规则拦截，而不是靠禁 rng。
_ALLOWED_TORCH_RNG_FUNCS = {
    "manual_seed", "seed", "initial_seed",
    "get_rng_state", "set_rng_state",
    "Generator",
}

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    "item", "tolist",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关（一般通过 __getitem__ 而非方法，但以防万一）
    "index_select",
    # 安全方法（不触发计算）
    "fill_", "copy_",
}

ALLOWED_TRITON_ATTRS = {
    "cdiv", "next_power_of_2",
}

# host 侧循环体内允许的无副作用 Python 内置 / math 调用
_ALLOWED_BUILTIN_CALLS = (
    "list", "tuple", "range", "len", "min", "max",
    "int", "float", "str", "enumerate", "zip",
)
_ALLOWED_MATH_CALLS = ("prod", "ceil", "floor", "log2", "pow")

# Host-side math fallback 检查常量
# 禁止在 host 侧（含 helper / __init__ / 全局初始化）用下列库计算三角/超越函数并生成 tensor。
# math.cos / math.sin 作为标量常量初始化例外允许；kernel 内部的 tl.* intrinsic 例外允许。
_HOST_FORBIDDEN_MATH_ROOTS = {
    "np": {"cos", "sin", "tan", "exp", "log", "log2", "log10", "sqrt", "pow"},
    "torch": {"cos", "sin", "tan", "exp", "log", "log2", "log10", "sqrt", "pow"},
    # math 模块仅保留 cos/sin 作为标量常量使用入口
    "math": {"tan", "exp", "log", "log2", "log10", "sqrt", "pow"},
}

_HOST_ALLOWED_MATH_FUNCS = {"cos", "sin"}

# 动态属性规避检测（getattr）相关常量
# torch 命名空间根：在这些对象上用 getattr 动态取属性，是绕过 nn.XXX / F.xxx
# 静态黑名单的典型手法（实测样本 16_EMSA：
#   _F = getattr(torch, 'nn').functional; _linear = getattr(_F, 'linear')）
_TORCH_NAMESPACE_ROOTS = {
    "torch", "nn", "F", "functional",
    # torch_npu 暴露 npu_fusion_attention / npu_* 等 CANN 融合算子入口，
    # 用 getattr 动态取到就等于把整个算子外包给现成 kernel，危害高于 F.xxx
    "torch_npu", "npu",
}

# 允许在 torch 上动态取的属性：dtype / device 等纯元数据，不是算子入口
# torch_npu 的命名约定：
#   - CANN 融合算子直接挂在顶层，统一以 npu_ 前缀暴露
#     （npu_fusion_attention / npu_rms_norm / npu_lightning_indexer ...）
#   - 设备管理与配置查询在 npu 子模块下
#     （torch_npu.npu.current_device / torch_npu.npu.npu_config.get_device_limit ...）
# 因此只需拦截"torch_npu 根之后第一级属性以 npu_ 开头"的调用，
# 既堵住把算子外包给现成 kernel，又不误伤 tiling 所需的硬件参数查询。
_TORCH_NPU_FUSED_OP_PREFIX = "npu_"

_GETATTR_ALLOWED_TORCH_ATTRS = {
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "bool",
    "half", "float", "double", "long", "short",
    "device", "dtype", "Tensor", "Size",
    "__version__", "version",
}

_HOST_FORBIDDEN_TENSOR_MATH_METHODS = {
    "cos", "sin", "tan", "exp", "log", "log2", "log10", "sqrt", "pow",
}

FORBIDDEN_TENSOR_METHODS = {
    # 计算操作
    "sum", "mean", "max", "min", "softmax", "log_softmax",
    "matmul", "mm", "bmm", "addmm", "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
}

# ---------------------------------------------------------------------------
# 模块级禁止（不只 forward 子树，扫整个文件）
# ---------------------------------------------------------------------------

# torch.nn 下所有"会构造带参数 Module / 调用算子"的入口。
# 出现这些调用基本等同于借 PyTorch 现成实现作弊。
MODULE_WIDE_FORBIDDEN_NN_ATTRS = {
    # 卷积 / 线性
    "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d",
    "ConvTranspose3d", "Linear", "Bilinear",
    # 池化
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d",
    "AvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "LPPool1d", "LPPool2d", "LPPool3d",
    # 归一化
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "LocalResponseNorm",
    # 激活
    "ReLU", "LeakyReLU", "PReLU", "ELU", "CELU", "SELU", "GELU", "SiLU", "GLU",
    "Hardswish", "Hardtanh", "Softplus", "Softsign", "Tanh", "Sigmoid",
    "LogSigmoid", "Hardsigmoid", "MultiheadAttention",
    # RNN / Embedding
    "RNN", "LSTM", "GRU", "Embedding",
    # Loss / Dropout
    "MSELoss", "L1Loss", "CrossEntropyLoss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss",
    "SmoothL1Loss", "HuberLoss", "TripletMarginLoss", "Dropout", "Dropout2d", "Dropout3d",
    # 量化 / 其它
    "Quantize", "DeQuantize", "Flatten", "Unflatten",
}

# 任何带 nn./nn.Module 子类的限定符都视为可疑 Module 构造路径。
# （FUNCTIONAL_QUALIFIERS 在下面定义，用函数动态合并避免顺序耦合）
MODULE_WIDE_NN_QUALIFIERS = {
    "nn", "torch.nn", "torch.nn.functional",
}

# rng state 操纵 API 不在模块级禁止范围——
# torch.manual_seed / get_rng_state / set_rng_state 等虽然在作弊代码里
# 经常和 nn.XXX 一起出现（用于复刻 baseline 随机权重），但它们也有
# 合法用途（确定性测试、含随机性的算子如 dropout/sampling）。
# 模块级禁用 nn.XXX 已经堵住了"复刻 nn 层权重"这一主要作弊路径。
MODULE_WIDE_FORBIDDEN_RNG_ATTRS = set()

# 模块级禁止的 torch 属性（不止计算，所有"借 PyTorch 现成算子"的入口都禁）
MODULE_WIDE_FORBIDDEN_TORCH_ATTRS = {
    # 直接调算子（绕过 F./nn. 形式）
    "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "multi_head_attention_forward",
    # 量化
    "quantize_per_tensor", "quantize_per_channel", "dequantize",
}

FUNCTIONAL_QUALIFIERS = {
    "F", "functional", "torch.nn.functional", "nn.functional",
}

# forward() 中禁止的 Python 控制流和结构
FORBIDDEN_PYTHON_STMTS = {
    "for": "Python for 循环",
    "while": "Python while 循环",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _decorator_is_triton_jit(decorator):
    """判断装饰器节点是否为 triton.jit 或 @jit（从 triton 导入）。"""
    # @triton.jit
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"):
            return True
    # @jit（直接导入）
    if isinstance(decorator, ast.Name) and decorator.id == "jit":
        return True
    # @triton.jit 作为 Call（如 @triton.jit 带参数，虽然少见）
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_jit(decorator.func)
    return False


def _decorator_is_triton_autotune(decorator):
    """判断装饰器是否为 triton.autotune。"""
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "autotune"):
            return True
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_autotune(decorator.func)
    return False


def _has_triton_decorator(func_node):
    """检查函数是否有 @triton.jit（可能与 @triton.autotune 组合）。"""
    for dec in func_node.decorator_list:
        if _decorator_is_triton_jit(dec):
            return True
    return False


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          my_func    -> (None, 'my_func')
          self.conv  -> ('self', 'conv')
          kernel[g]  -> 返回 None（kernel launch 通过 Subscript）
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _get_subscript_value_name(node):
    """从 kernel[grid](...) 的 Subscript 节点提取 kernel 名称。"""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return f"{node.value.value.id}.{node.value.attr}"
    return None


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def _kernel_uses_tl_api(func_node) -> bool:
    """判断 kernel 函数体内是否出现 tl.* 属性访问。"""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name) and child.value.id == "tl":
                return True
    return False


def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            kernels[node.name] = {
                "has_tl_usage": _kernel_uses_tl_api(node),
                "line": node.lineno,
            }
    return kernels


def _collect_torch_namespace_aliases(tree):
    """收集所有指向 torch 命名空间的变量名（含 getattr 链传递）。

    覆盖：
      import torch.nn.functional as XX          -> XX
      XX = torch.nn.functional / nn.functional  -> XX
      XX = getattr(torch, 'nn')                 -> XX
      XX = getattr(torch, 'nn').functional      -> XX
    用不动点迭代传递别名，避免依赖赋值顺序。
    """
    aliases = set(_TORCH_NAMESPACE_ROOTS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in ("torch", "torch_npu") and a.asname:
                    aliases.add(a.asname)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in ("torch", "torch_npu"):
                for a in node.names:
                    aliases.add(a.asname or a.name)

    def _root_of(expr):
        """取表达式的根标识符；getattr(X, ...) 视作 X 的根。"""
        cur = expr
        while True:
            if isinstance(cur, ast.Attribute):
                cur = cur.value
            elif isinstance(cur, ast.Call):
                f = cur.func
                if isinstance(f, ast.Name) and f.id == "getattr" and cur.args:
                    cur = cur.args[0]
                else:
                    return None
            elif isinstance(cur, ast.Name):
                return cur.id
            else:
                return None

    for _ in range(4):                       # 不动点：4 轮足够覆盖常见链长
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            root = _root_of(node.value)
            if root is None or root not in aliases:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id not in aliases:
                    aliases.add(tgt.id)
                    changed = True
        if not changed:
            break
    return aliases


def _collect_torch_npu_aliases(tree):
    """收集指向 torch_npu 的名字（含 import torch_npu as _tn 这类别名）。"""
    names = {"torch_npu"}
    if tree is None:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "torch_npu" and a.asname:
                    names.add(a.asname)
        elif isinstance(node, ast.Assign):
            v = node.value
            if isinstance(v, ast.Name) and v.id in names:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
    return names


def _attr_chain(node):
    """把 a.b.c 形式的 Attribute 拆成 (根名, [属性名...])；非此形式返回 (None, [])。"""
    attrs = []
    cur = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None, []
    return cur.id, list(reversed(attrs))


def _check_torch_npu_direct_calls(tree):
    """检测直接调用 torch_npu 顶层的 npu_* 融合算子。

    torch_npu 顶层的 npu_* 系列（npu_fusion_attention / npu_rms_norm /
    npu_lightning_indexer ...）是 CANN 已实现的高性能融合算子。生成代码若直接
    调用，等于把整个算子外包给现成 kernel，本质与 F.xxx 退化同类且危害更大
    —— 单次调用即可替代整个 attention。

    不误伤设备管理与硬件参数查询：这些在 npu 子模块下
    （torch_npu.npu.current_device()、
      torch_npu.npu.npu_config.get_device_limit(0) 等），
    是 tiling 决策所必需，根之后第一级属性为 "npu"，不以 npu_ 开头，天然放行。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    violations = []
    if tree is None:
        return violations

    aliases = _collect_torch_npu_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        root, attrs = _attr_chain(node.func)
        if root is None or root not in aliases or not attrs:
            continue
        first = attrs[0]
        if not first.startswith(_TORCH_NPU_FUSED_OP_PREFIX):
            continue
        violations.append({
            "line": node.lineno,
            "call": f"{root}.{'.'.join(attrs)}(...)",
            "reason": (
                f"直接调用 {root}.{first} —— 这是 CANN 已实现的融合算子，"
                "调用它等于把计算外包给现成 kernel。所有核心计算必须在 "
                "@triton.jit kernel 内用 tl.* 完成。"
            ),
        })
    return violations


def _check_dynamic_attr_evasion(source_code):
    """检测用 getattr 动态访问 torch/nn/F 命名空间来绕过静态黑名单的写法。

    命中条件（两条同时成立）：
      1. getattr 的第一个实参解析到 torch 命名空间根或其别名
      2. 第二个实参是字符串常量，且不在纯元数据白名单内（dtype/device 等）

    不误伤在普通对象上探属性的写法，如
      getattr(a, "dtype", None)               —— a 是 tensor 变量
      getattr(props, 'vector_core_num', None) —— props 是设备属性对象
    这些的第一个实参不属于 torch 命名空间，不会命中。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    aliases = _collect_torch_namespace_aliases(tree)
    violations = list(_check_torch_npu_direct_calls(tree))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue

        target = node.args[0]
        # 解析第一个实参的根标识符
        cur, ok = target, True
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Call):
            f = cur.func
            if isinstance(f, ast.Name) and f.id == "getattr" and cur.args:
                inner = cur.args[0]
                while isinstance(inner, ast.Attribute):
                    inner = inner.value
                cur = inner
            else:
                ok = False
        if not ok or not isinstance(cur, ast.Name) or cur.id not in aliases:
            continue

        attr_node = node.args[1]
        if not (isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str)):
            # 属性名非字面量 —— 更可疑，直接判违规
            attr_name = "<非常量>"
        else:
            attr_name = attr_node.value
            # 元数据白名单仅适用于 torch 根（dtype/device 等）。
            # torch_npu / npu 下没有需要动态取的纯元数据，一律不放行。
            if cur.id == "torch" and attr_name in _GETATTR_ALLOWED_TORCH_ATTRS:
                continue

        violations.append({
            "line": node.lineno,
            "call": f"getattr({cur.id}, {attr_name!r})",
            "reason": (
                f"用 getattr 动态访问 {cur.id}.{attr_name}，绕过 nn.XXX / F.xxx 静态黑名单。"
                "torch/nn/F 命名空间下的算子必须直接书写以便审计；"
                "所有计算必须在 @triton.jit kernel 内完成。"
            ),
        })
    return violations


def _check_host_math_fallback(source_code):
    """检查 host 侧是否用 np/torch/tensor 方法计算三角/超越函数并生成 tensor。

    禁止模式：
      - np.cos / np.sin / np.exp / ... 后接 torch.from_numpy(...).to(device)
      - torch.cos / torch.sin / torch.exp / ...
      - tensor.cos() / tensor.sin() / tensor.exp() / ...
      - math.tan / math.exp / math.log / ...（math 模块仅 cos/sin 允许）

    允许模式：
      - math.cos / math.sin 用于标量常量初始化
      - tl.cos / tl.sin / tl.exp 等 triton.language intrinsic 在 @triton.jit kernel 内部

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    kernel_funcs = {
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node)
    }

    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def _is_inside_triton_kernel(node):
        cur = parents.get(node)
        while cur is not None:
            if isinstance(cur, ast.FunctionDef) and cur in kernel_funcs:
                return True
            cur = parents.get(cur)
        return False

    def _is_table_building_context(node):
        """判断 math.* 调用的结果是否被逐元素收集成序列/数组。

        math 模块只接受标量，单次调用产出一个 Python float。只有当这个结果
        被逐个收集起来时，才可能拼出 kernel 输入表，例如：
            vals = [math.exp(i) for i in range(n)]; t = torch.tensor(vals)
            for i in range(n): arr.append(math.sin(i))
            for i in range(n): buf[i] = math.cos(i)

        以下写法不算填表，必须放行（实测大量算子在用）：
            bound = 1.0 / math.sqrt(fan_in)                      # 标量初始化边界
            w = torch.empty(a, b).uniform_(-bound, bound)        # 张量整体生成
            for c_out, c_in in shapes:                           # 循环遍历"层"而非元素
                bound = 1.0 / math.sqrt(c_in)
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))    # 标量超参
            sm_scale = 1.0 / math.sqrt(head_dim)                 # attention 缩放因子

        判据是"结果去了哪里"，而不是"出现在什么语句里"。
        """
        cur = node
        parent = parents.get(cur)
        while parent is not None:
            # ① 推导式的元素表达式里 —— 每次迭代产出一个值并被收集
            if isinstance(parent, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                if cur is parent.elt:
                    return True
            elif isinstance(parent, ast.DictComp):
                if cur is parent.key or cur is parent.value:
                    return True
            # ② 作为 append/extend/insert 的实参 —— 逐个塞进列表
            elif isinstance(parent, ast.Call):
                f = parent.func
                if isinstance(f, ast.Attribute) and f.attr in ("append", "extend", "insert"):
                    if any(cur is a for a in parent.args):
                        return True
            # ③ 赋给下标 —— buf[i] = math.xxx(...)，逐位置填充
            elif isinstance(parent, ast.Assign):
                if cur is parent.value and any(
                        isinstance(t, ast.Subscript) for t in parent.targets):
                    return True
            cur, parent = parent, parents.get(parent)
        return False

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        # 1) 函数调用形式：np.cos / torch.cos / math.cos / tl.cos
        if isinstance(func, (ast.Name, ast.Attribute)):
            resolved = _resolve_call_name(node)
            if resolved is not None:
                qual, attr = resolved
                root = qual.split(".")[0] if qual else None

                # kernel 内部的 triton.language intrinsic 允许
                if root in ("tl", "triton") and _is_inside_triton_kernel(node):
                    continue

                if root in _HOST_FORBIDDEN_MATH_ROOTS and attr in _HOST_FORBIDDEN_MATH_ROOTS[root]:
                    # math.cos / math.sin 作为标量常量初始化允许
                    if root == "math" and attr in _HOST_ALLOWED_MATH_FUNCS:
                        continue
                    # math.* 只接受标量，单次调用产出 Python float，无法生成 tensor
                    # 输入表（如 sm_scale = 1.0 / math.sqrt(head_dim)、
                    # bound = 1.0 / math.sqrt(d_model) 这类缩放因子/初始化边界）。
                    # 只有放进推导式或循环体逐元素填表时才真正构成 host 侧超越计算。
                    if root == "math" and not _is_table_building_context(node):
                        continue
                    call_str = f"{qual}.{attr}(...)" if qual else f"{attr}(...)"
                    violations.append({
                        "line": node.lineno,
                        "call": call_str,
                        "reason": (
                            f"host 侧 {call_str} 生成 tensor 输入表，"
                            "必须移入 @triton.jit kernel 用 tl.* intrinsic 实现"
                        ),
                    })
                    continue

        # 2) tensor 方法调用形式：x.cos() / x.sin() / x.exp()
        if isinstance(func, ast.Attribute) and func.attr in _HOST_FORBIDDEN_TENSOR_MATH_METHODS:
            # 排除已在上文处理的模块函数调用（如 math.cos / torch.cos）
            if _resolve_call_name(node) is not None:
                continue
            violations.append({
                "line": node.lineno,
                "call": f"<tensor>.{func.attr}(...)",
                "reason": (
                    f"host 侧 tensor.{func.attr}() 生成 tensor 输入表，"
                    "必须移入 @triton.jit kernel 用 tl.* intrinsic 实现"
                ),
            })

    return violations


def _find_forward_in_class(class_node):
    """从 ModelNew 类节点中找到 forward 方法。"""
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            return item
    return None


def find_model_new_forward(tree):
    """找到 ModelNew 类的 forward 方法节点。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            forward = _find_forward_in_class(node)
            if forward is not None:
                return forward
    return None


def _call_invokes_kernel(call_node, kernel_names) -> bool:
    """判断单个 Call 节点是否启动了 triton kernel（直接或通过 Subscript）。"""
    if isinstance(call_node.func, ast.Subscript):
        return _get_subscript_value_name(call_node.func) in kernel_names
    resolved = _resolve_call_name(call_node)
    return bool(resolved and resolved[0] is None and resolved[1] in kernel_names)


def _func_calls_kernel(func_node, kernel_names) -> bool:
    """判断函数体内是否存在 kernel 启动调用。"""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call) and _call_invokes_kernel(child, kernel_names):
            return True
    return False


def find_wrapper_functions(tree, kernel_names):
    """找到模块级别或类级别的辅助函数，这些函数内部调用了 triton kernel。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            if _func_calls_kernel(node, kernel_names):
                wrappers.add(node.name)
    return wrappers


def _called_from_call_node(call_node, kernel_names, wrapper_names):
    """从单个 Call 节点中提取被调用的 kernel/wrapper 名称，找不到返回 None。"""
    if isinstance(call_node.func, ast.Subscript):
        name = _get_subscript_value_name(call_node.func)
        if name in kernel_names:
            return name
    resolved = _resolve_call_name(call_node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual is None and attr in kernel_names:
        return attr
    if qual is None and attr in wrapper_names:
        return attr
    if qual == "self" and attr in wrapper_names:
        return attr
    return None


def check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names):
    """检查 forward 中是否调用了 triton kernel（直接或通过 wrapper）。

    返回被调用的 kernel/wrapper 名称集合。
    """
    called = set()
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call):
            name = _called_from_call_node(node, kernel_names, wrapper_names)
            if name is not None:
                called.add(name)
    return called


def _count_kernel_launches_in_forward(forward_node):
    """统计 forward() 中 kernel 启动调用（kernel[grid](...)）的次数。"""
    count = 0
    if forward_node is None:
        return count
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            count += 1
    return count


# --- check_forbidden_torch_ops 拆分出的辅助规则 ---

def _violation_for_loop(node):
    """循环禁用规则：返回违规字典或 None。"""
    if isinstance(node, ast.For):
        return {
            "line": node.lineno,
            "call": "for 循环",
            "reason": "forward() 中禁止 Python for 循环，核心计算必须在单个 Triton kernel 内完成",
        }
    if isinstance(node, ast.While):
        return {
            "line": node.lineno,
            "call": "while 循环",
            "reason": (
                "forward() 中禁止 Python while 循环，"
                "核心计算必须在单个 Triton kernel 内完成"
            ),
        }
    return None


def _violation_matmul_op(node):
    """检测 @ 运算符。"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return {
            "line": node.lineno,
            "call": "@",
            "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
        }
    return None


def _violation_list_append(node):
    """检测 list.append 形式调用。"""
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if attr == "append" and qual is not None:
        return {
            "line": node.lineno,
            "call": f"{qual}.append(...)",
            "reason": (
                "forward() 中禁止 list.append，"
                "动态状态维护必须在 Triton kernel 内完成"
            ),
        }
    return None


def _violation_for_torch_qual(node, qual, attr):
    """处理 qual == 'torch' 的调用。"""
    if attr in ALLOWED_TORCH_FUNCS:
        return None
    # rng 状态操作不是计算操作（复刻 baseline 随机权重初始化所必需）
    if attr in _ALLOWED_TORCH_RNG_FUNCS:
        return None
    return {
        "line": node.lineno,
        "call": f"torch.{attr}",
        "reason": f"torch.{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_functional_qual(node, qual, attr):
    """处理 F./functional. 形式调用。"""
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}",
        "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_tensor_method(node, qual, attr):
    """处理被禁止的 tensor 方法调用。

    ⚠️ 只对**属性调用**（`x.min()`）成立。裸调用（`min(a, b)`）是 Python 内建或
    模块内自定义函数，不是 tensor 方法：host 侧的 `min(128, ceil16(S_K))`、
    `grid = min(NUM_CORES, total_tasks)` 等标准 tiling 写法必须放行。
    （自定义函数里的主链计算由 check_main_chain_compute 的跨函数扫描覆盖。）
    """
    if qual is None:
        return None
    if attr not in FORBIDDEN_TENSOR_METHODS:
        return None
    # 排除已知安全的 qual（torch/F/triton 已在上面处理）
    skip_quals = {"torch", "F", "triton"} | FUNCTIONAL_QUALIFIERS
    if qual in skip_quals:
        return None
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}()" if qual else f"{attr}()",
        "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


_CURRENT_SELF_METHODS = set()


def _collect_self_methods(tree):
    """收集文件中所有 class 里定义的方法名（供 self.xxx() 规则放行本类方法）。"""
    names = set()
    if tree is None:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(item.name)
    return names


def _violation_for_self_call(node, qual, attr, self_methods=None):
    """处理 self.xxx(...) 形式调用。

    只有当 self.xxx 是 __init__ 里挂上去的 nn.Module 属性时，self.xxx(...) 才是
    "借 module 前向计算"。若 xxx 是本类里定义的普通方法（如 _layers() 返回缓存
    权重、_get_weights() 造权重），调用它并不构成 PyTorch 计算——方法体内部
    是否有违规计算，由跨函数扫描单独覆盖。
    """
    if qual != "self" or attr == "forward":
        return None
    if self_methods and attr in self_methods:
        return None
    return {
        "line": node.lineno,
        "call": f"self.{attr}(...)",
        "reason": (
            f"self.{attr}() 疑似 nn.Module 前向调用，"
            "核心计算必须在 Triton kernel 中实现"
        ),
    }


def _violation_for_call(node):
    """对 Call 节点应用所有调用相关规则，返回首个命中或 None。"""
    v = _violation_list_append(node)
    if v is not None:
        return v

    # --- kernel launch: kernel[grid](...) —— 允许 ---
    if isinstance(node.func, ast.Subscript):
        return None

    resolved = _resolve_call_name(node)
    if resolved is None:
        return None

    qual, attr = resolved
    if qual == "torch":
        return _violation_for_torch_qual(node, qual, attr)
    if qual in FUNCTIONAL_QUALIFIERS:
        return _violation_for_functional_qual(node, qual, attr)
    # --- triton.cdiv 等 —— 允许 ---
    if qual == "triton" and attr in ALLOWED_TRITON_ATTRS:
        return None
    v = _violation_for_tensor_method(node, qual, attr)
    if v is not None:
        return v
    return _violation_for_self_call(node, qual, attr, _CURRENT_SELF_METHODS)


def _violation_for_node(node):
    """对任意 AST 节点应用所有规则，返回首个命中或 None。"""
    v = _violation_for_loop(node)
    if v is not None:
        return v
    v = _violation_matmul_op(node)
    if v is not None:
        return v
    if isinstance(node, ast.Call):
        return _violation_for_call(node)
    return None


def _resolved_call_allowed(resolved):
    """判断已解析的调用 (qual, attr) 是否属于 loop 体内允许的 host 侧调用。"""
    if not resolved:
        return False
    qual, attr = resolved
    # torch.empty / torch.empty_like 等 buffer 分配允许
    if qual == "torch" and attr in ALLOWED_TORCH_FUNCS:
        return True
    # 允许的 tensor 方法
    if attr in ALLOWED_TENSOR_METHODS and qual is not None:
        return True
    # list / tuple / range / len 等 Python 内置允许
    if qual is None and attr in _ALLOWED_BUILTIN_CALLS:
        return True
    # math.prod / math.ceil 等 math 模块函数允许
    if qual == "math" and attr in _ALLOWED_MATH_CALLS:
        return True
    # self.xxx(...) 放宽，由后续 torch/F 检测兜底
    return qual == "self"


def _is_loop_pure_kernel_launch(loop_node, kernel_names, wrapper_names):
    """检查循环体是否仅包含 kernel 启动和允许的 host 侧操作。
    允许的语句：kernel[grid](...)、赋值、条件判断（if/else）、
    torch.empty/empty_like、属性访问、方法调用（如 .contiguous/.numel 等）。
    禁止的语句：任何非 kernel 的 torch/F 计算操作、nn.Module 调用等。
    """

    def _check_stmt(stmt):
        if isinstance(stmt, ast.If):
            for s in stmt.body + stmt.orelse:
                if not _check_stmt(s):
                    return False
            return True
        if isinstance(stmt, ast.For):
            return False  # 嵌套循环禁止
        if isinstance(stmt, ast.While):
            return False
        if isinstance(stmt, ast.With):
            return False
        if isinstance(stmt, ast.Try):
            return False
        if isinstance(stmt, ast.Expr):
            return _check_expr(stmt.value)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if not _check_expr(target):
                    return False
            return _check_expr(stmt.value)
        if isinstance(stmt, ast.AugAssign):
            return _check_expr(stmt.target) and _check_expr(stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            return _check_expr(stmt.target) and (stmt.value is None or _check_expr(stmt.value))
        if isinstance(stmt, ast.Return):
            return stmt.value is None or _check_expr(stmt.value)
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, (ast.Continue, ast.Break)):
            return True
        return False

    def _check_call_expr(expr):
        # kernel[grid](...) 允许
        if isinstance(expr.func, ast.Subscript):
            name = _get_subscript_value_name(expr.func)
            if name in kernel_names or name in wrapper_names:
                return True
        if _resolved_call_allowed(_resolve_call_name(expr)):
            return True
        # 递归检查参数
        for kw in expr.keywords:
            if not _check_expr(kw.value):
                return False
        for arg in expr.args:
            if not _check_expr(arg):
                return False
        return True

    def _check_expr_collection(expr):
        if isinstance(expr, ast.Compare):
            return all(_check_expr(e) for e in [expr.left] + expr.comparators)
        if isinstance(expr, ast.BoolOp):
            return all(_check_expr(v) for v in expr.values)
        if isinstance(expr, (ast.Tuple, ast.List)):
            return all(_check_expr(e) for e in expr.elts)
        if isinstance(expr, ast.Dict):
            return all(_check_expr(k) for k in expr.keys) and all(_check_expr(v) for v in expr.values)
        if isinstance(expr, ast.JoinedStr):
            return all(_check_expr(v) for v in expr.values)
        if isinstance(expr, ast.IfExp):
            return _check_expr(expr.test) and _check_expr(expr.body) and _check_expr(expr.orelse)
        # 其余（Lambda / 推导式 / Await / Yield 等）默认不允许
        return False

    def _check_expr(expr):
        if isinstance(expr, ast.Call):
            return _check_call_expr(expr)
        if isinstance(expr, (ast.Name, ast.Constant)):
            return True
        if isinstance(expr, ast.Attribute):
            return _check_expr(expr.value)
        if isinstance(expr, ast.Subscript):
            return _check_expr(expr.value) and _check_expr(expr.slice)
        if isinstance(expr, (ast.Starred, ast.FormattedValue)):
            return _check_expr(expr.value)
        if isinstance(expr, ast.UnaryOp):
            return _check_expr(expr.operand)
        if isinstance(expr, ast.BinOp):
            return _check_expr(expr.left) and _check_expr(expr.right)
        return _check_expr_collection(expr)

    for stmt in loop_node.body:
        if not _check_stmt(stmt):
            return False
    return True


def _call_leaf_name(call_node):
    """取 Call 节点的叶子名：foo() -> foo；self.foo() -> foo；a.b.foo() -> foo。"""
    f = call_node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


# 主链计算算子：出现在 kernel 之外即等同于"没用 Triton 实现"。
# 刻意不含 host 侧合法操作（rng 操纵、math.*、torch.empty/zeros、.view/.t()/.contiguous 等）。
MAIN_CHAIN_TORCH_FUNCS = {
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "tensordot", "inner", "outer",
    "softmax", "log_softmax", "sigmoid", "tanh", "relu", "gelu", "silu", "elu",
    "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "layer_norm", "batch_norm", "group_norm", "scaled_dot_product_attention",
    "max_pool1d", "max_pool2d", "avg_pool1d", "avg_pool2d", "linear", "cumsum", "cumprod",
}
MAIN_CHAIN_TENSOR_METHODS = {
    "matmul", "mm", "bmm", "softmax", "log_softmax", "sigmoid", "tanh", "relu",
    "masked_fill", "masked_fill_", "cumsum", "cumprod",
}


def check_main_chain_compute(func_node, func_label):
    """在非 kernel 函数体内查找**主链计算算子**（比 forward 的规则窄，避免误伤 host 侧准备工作）。"""
    out = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        qual = f.value.id if isinstance(f.value, ast.Name) else None
        if qual in ("torch", "F", "functional", "nn") and f.attr in MAIN_CHAIN_TORCH_FUNCS:
            out.append({"line": node.lineno,
                        "call": f"{func_label} -> {qual}.{f.attr}",
                        "reason": f"{qual}.{f.attr} 是主链计算算子，必须在 @triton.jit kernel 中实现"})
        elif qual not in ("torch", "F", "functional", "nn", "math", "np", "numpy") \
                and f.attr in MAIN_CHAIN_TENSOR_METHODS:
            out.append({"line": node.lineno,
                        "call": f"{func_label} -> .{f.attr}()",
                        "reason": f"tensor 方法 .{f.attr}() 是主链计算算子（用方法调用绕过 F.{f.attr} 名单同样不允许）"})
    return out


def collect_reachable_funcs(tree, forward_node, kernel_names):
    """从 forward() 出发沿模块内函数调用做可达性分析。

    返回 [(函数名, FunctionDef), ...]（不含 forward 自身与 triton kernel）。
    用途：作弊代码常把主链计算挪到 forward() 之外的模块级辅助函数里，
    只在 forward() 里留一句调用 + 一个占位 kernel 启动，从而绕过只扫
    forward 子树的 Type 3 检查。
    """
    module_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            module_funcs.setdefault(node.name, node)

    seen = set()
    out = []
    stack = [forward_node]
    while stack:
        cur = stack.pop()
        for node in ast.walk(cur):
            if not isinstance(node, ast.Call):
                continue
            name = _call_leaf_name(node)
            if name and name in module_funcs and name not in seen:
                seen.add(name)
                fn = module_funcs[name]
                if fn is not forward_node:
                    out.append((name, fn))
                    stack.append(fn)
    return out


# kernel 内只做搬运不算计算的 tl.* API
_TL_NON_COMPUTE = {
    "load", "store", "arange", "program_id", "num_programs", "cdiv",
    "constexpr", "multiple_of", "max_contiguous", "static_assert",
    "make_block_ptr", "advance", "zeros", "full", "device_print",
}


def kernel_is_placeholder(func_node) -> bool:
    """判断 kernel 是否是"占位 kernel"：把 load 进来的值原样 store 出去，无任何变换。

    判据（两条同时满足才判占位，避免误伤只用 Python 运算符做计算的 kernel）：
      1. 没有使用任何**计算类** tl.* / libdevice.* API；
      2. 每一处 tl.store 的值参数都是一个变量名，且该变量的唯一赋值就是 tl.load(...)。
    """
    used = set()
    for child in ast.walk(func_node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id in ("tl", "libdevice"):
                used.add(child.attr)
    if used - _TL_NON_COMPUTE:
        return False

    # 收集「变量 -> 是否直接由 tl.load 赋值」
    load_vars, other_vars = set(), set()
    for child in ast.walk(func_node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
            name = child.targets[0].id
            v = child.value
            is_load = (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                       and v.func.attr == "load"
                       and isinstance(v.func.value, ast.Name) and v.func.value.id == "tl")
            (load_vars if is_load else other_vars).add(name)

    stores = [c for c in ast.walk(func_node)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
              and c.func.attr == "store"
              and isinstance(c.func.value, ast.Name) and c.func.value.id == "tl"]
    if not stores:
        return False
    for st in stores:
        if len(st.args) < 2:
            return False
        val = st.args[1]
        if not (isinstance(val, ast.Name) and val.id in load_vars and val.id not in other_vars):
            return False
    return True


def kernel_has_dot(func_node) -> bool:
    """kernel 内是否出现 tl.dot。"""
    for child in ast.walk(func_node):
        if (isinstance(child, ast.Attribute) and child.attr == "dot"
                and isinstance(child.value, ast.Name) and child.value.id == "tl"):
            return True
    return False


def check_forbidden_torch_ops(forward_node, kernel_names=None, wrapper_names=None):
    """检查 forward 中是否使用了禁止的 torch 计算操作或 Python 控制流。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        v = _violation_for_node(node)
        if v is not None:
            violations.append(v)

    # --- 规则 B: 检查 Host 侧循环 ---
    # 如果循环体仅包含 kernel 启动和允许的 host 侧操作，则允许
    # 否则视为 Type3 退化
    if kernel_names is None:
        kernel_names = set()
    if wrapper_names is None:
        wrapper_names = set()
    for node in ast.walk(forward_node):
        if isinstance(node, (ast.For, ast.While)):
            if not _is_loop_pure_kernel_launch(node, kernel_names, wrapper_names):
                loop_type = "for" if isinstance(node, ast.For) else "while"
                violations.append({
                    "line": node.lineno,
                    "call": f"{loop_type} 循环",
                    "reason": (
                        f"forward() 中 {loop_type} 循环体包含非 kernel 的计算操作，"
                        "核心计算必须在 Triton kernel 内完成"
                    ),
                })

    return violations


# ---------------------------------------------------------------------------
# 模块级禁止检查（不只 forward AST，扫整个文件）
# ---------------------------------------------------------------------------

def _nn_modules_are_only_weight_holders(tree):
    """判断文件里构造的 nn.XXX 是否只被当作"权重容器"使用。

    背景：部分任务的 baseline 在 forward 内用固定种子现场 nn.XXX 初始化权重、
    权重不从入参传入（如 17_MultiHeadAttention、44_CrissCrossAttention），
    实现要对齐数值就必须复刻同样的初始化。这类实现只读 .weight/.bias 张量
    喂给自己的 @triton.jit kernel，从不调用 module 的前向。

    判据（两条同时成立才放行）：
      1. 文件里至少有一处 .weight / .bias 属性读取（提取权重的正向信号）
      2. 没有任何"既被读 .weight/.bias、又被当函数调用"的对象
         （既读权重又调用 = 真拿 module 算前向，属作弊）
    """
    weight_read = set()      # 被读过 .weight/.bias 的对象源码形态
    called = set()           # 被当函数调用过的对象源码形态

    def _key(n):
        """把 Name / Attribute 归一成可比较的字符串键。"""
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            base = _key(n.value)
            return f"{base}.{n.attr}" if base else None
        return None

    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in ("weight", "bias"):
            k = _key(n.value)
            if k:
                weight_read.add(k)
        elif isinstance(n, ast.Call) and not isinstance(n.func, ast.Subscript):
            k = _key(n.func)
            if k:
                called.add(k)

    if not weight_read:
        return False
    # 既读权重又被调用的对象 => module 参与了前向计算
    return not (weight_read & called)


def _violation_module_wide_for_call(node, weight_holder_only=False):
    """对整个文件范围内的 ast.Call 节点应用模块级禁止规则。

    覆盖 forward() AST 扫描抓不到的间接作弊：
    - helper 函数 / __init__ 里 nn.ConvTranspose2d(...) 复刻 baseline 权重
    - F.relu 等 functional 调用绕过算子
    """
    # kernel launch: kernel[grid](...) 允许
    if isinstance(node.func, ast.Subscript):
        return None

    resolved = _resolve_call_name(node)
    if resolved is None:
        return None
    qual, attr = resolved

    # nn.XXX / torch.nn.functional.XXX / F.XXX 构造或调用
    nn_qualifiers = MODULE_WIDE_NN_QUALIFIERS | FUNCTIONAL_QUALIFIERS
    if qual in nn_qualifiers:
        if attr in MODULE_WIDE_FORBIDDEN_NN_ATTRS:
            # 仅用于复刻 baseline 权重（只读 .weight/.bias、从不调用前向）时放行
            if weight_holder_only:
                return None
            return {
                "line": node.lineno,
                "call": f"{qual}.{attr}(...)",
                "reason": (
                    f"模块级禁止：{qual}.{attr} 是 PyTorch 现成算子/Module，"
                    "在生成代码任何位置（含 helper、__init__、forward）调用"
                    "都视为作弊。所有计算必须在 @triton.jit kernel 内完成；"
                    "权重应当从外部传入或用常量初始化，不得借 nn.XXX 复刻 baseline 权重。"
                ),
            }
        # nn.functional.* 任意调用都禁（这一类全是 PyTorch 算子入口）
        if qual in FUNCTIONAL_QUALIFIERS:
            return {
                "line": node.lineno,
                "call": f"{qual}.{attr}(...)",
                "reason": (
                    f"模块级禁止：{qual}.{attr} 属于 torch.nn.functional，"
                    "是 PyTorch 算子入口，禁止在生成代码任何位置调用。"
                ),
            }

    # torch.manual_seed / get_rng_state / set_rng_state 等 rng 操纵
    if qual == "torch" and attr in MODULE_WIDE_FORBIDDEN_RNG_ATTRS:
        return {
            "line": node.lineno,
            "call": f"{qual}.{attr}(...)",
            "reason": (
                f"模块级禁止：torch.{attr} 操纵随机数状态。"
                "正常 Triton kernel 不需要操纵 rng；出现此调用通常意味着"
                "Agent 在复刻 baseline 的随机权重或随机输入以骗过精度校验。"
            ),
        }

    # torch.<其它 forbidden attr>（如 torch.conv_transpose2d 直接调用）
    if qual == "torch" and attr in MODULE_WIDE_FORBIDDEN_TORCH_ATTRS:
        return {
            "line": node.lineno,
            "call": f"{qual}.{attr}(...)",
            "reason": (
                f"模块级禁止：torch.{attr} 是 PyTorch 现成算子入口，"
                "禁止在生成代码任何位置调用。"
            ),
        }

    return None


def check_forbidden_module_wide_ops(tree):
    """扫整个 AST（不只 forward 子树）找禁止的 PyTorch 接口调用。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    设计目标：堵住 forward() AST 扫描抓不到的间接作弊路径，
    如 nn.ConvTranspose2d 在 helper 函数里被构造、F.relu 在 wrapper
    里被调用等。
    """
    violations = []
    if tree is None:
        return violations
    weight_holder_only = _nn_modules_are_only_weight_holders(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        v = _violation_module_wide_for_call(node, weight_holder_only=weight_holder_only)
        if v is not None:
            violations.append(v)
    return violations


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def _empty_result(filepath):
    return {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "triton_kernel_exists": {"passed": False, "kernels": [], "error": None},
            "kernel_called_from_forward": {"passed": False, "called": [], "error": None},
            "no_forbidden_module_wide_ops": {"passed": False, "violations": [], "error": None},
            "kernel_not_placeholder": {"passed": False, "placeholders": [], "error": None},
            "no_dynamic_attr_evasion": {"passed": False, "violations": [], "error": None},
            "no_host_math_fallback": {"passed": False, "violations": [], "error": None},
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
        },
        "regression_type": None,
        "suggestion": "",
    }


def _check_kernel_exists(result, tree):
    """填充 triton_kernel_exists 检查；返回 (passed, kernel_names)。"""
    kernels = find_triton_kernels(tree)
    kernel_names = set(kernels.keys())
    result["checks"]["triton_kernel_exists"]["kernels"] = [
        {"name": k, "line": v["line"], "has_tl_usage": v["has_tl_usage"]}
        for k, v in kernels.items()
    ]

    if not kernel_names:
        result["checks"]["triton_kernel_exists"]["error"] = "未找到任何 @triton.jit 装饰的 kernel 函数"
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有 Triton kernel。必须创建至少一个 @triton.jit 装饰的函数，"
            "在其中使用 tl.load/tl.store 实现核心计算逻辑。"
        )
        return False, kernel_names

    kernels_without_tl = [k for k, v in kernels.items() if not v["has_tl_usage"]]
    if len(kernels_without_tl) == len(kernels):
        result["checks"]["triton_kernel_exists"]["error"] = (
            f"kernel 函数 {kernels_without_tl} 未使用任何 tl.* API，"
            "可能是空壳 kernel"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "虽然存在 @triton.jit 装饰的函数，但没有使用 triton.language (tl) API。"
            "kernel 必须使用 tl.load/tl.store 等进行显式内存操作和计算。"
        )
        return False, kernel_names

    result["checks"]["triton_kernel_exists"]["passed"] = True
    return True, kernel_names


def _check_forward_calls_kernel(result, tree, kernel_names):
    """填充 kernel_called_from_forward 检查；返回 (passed, forward_node, wrapper_names)。"""
    forward_node = find_model_new_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew 类或 forward 方法。"
        return False, None, set()

    wrapper_names = find_wrapper_functions(tree, kernel_names)
    called = check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names)
    result["checks"]["kernel_called_from_forward"]["called"] = list(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"@triton.jit kernel {list(kernel_names)} 已定义但 forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        wrapper_hint = (
            f"也存在 wrapper 函数 {list(wrapper_names)} 但 forward 也未调用它们。"
            if wrapper_names else ""
        )
        result["suggestion"] = (
            f"已定义 kernel {list(kernel_names)} 但 ModelNew.forward() 中未调用。"
            "forward() 必须通过 kernel_name[grid](...) 形式启动 kernel。"
            f"{wrapper_hint}"
        )
        return False, forward_node, wrapper_names

    result["checks"]["kernel_called_from_forward"]["passed"] = True
    return True, forward_node, wrapper_names


def _check_no_forbidden_ops(result, forward_node, kernel_names, wrapper_names, tree=None):
    """填充 no_forbidden_torch_ops 检查；返回 passed。

    除 forward() 子树外，**还扫描从 forward() 可达的模块内辅助函数**——
    把主链计算挪到 forward 外的函数里是常见的绕过手法。
    """
    violations = check_forbidden_torch_ops(forward_node, kernel_names, wrapper_names)
    if tree is not None:
        for fname, fnode in collect_reachable_funcs(tree, forward_node, kernel_names):
            violations.extend(check_main_chain_compute(fnode, f"{fname}()"))
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if not violations:
        result["checks"]["no_forbidden_torch_ops"]["passed"] = True
        return True

    result["checks"]["no_forbidden_torch_ops"]["error"] = (
        f"forward() 及其可达辅助函数中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"forward() 调用了 Triton kernel 但仍使用 PyTorch 进行部分计算: "
        f"{violation_details}。"
        "所有核心计算必须在 @triton.jit kernel 中完成，"
        "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
    )
    return False


def _check_no_forbidden_module_wide_ops(result, tree):
    """填充 no_forbidden_module_wide_ops 检查；返回 passed。

    扫整个文件（不只 forward AST）找 nn.XXX 构造、rng 操纵、F.xxx 调用等
    间接作弊路径。
    """
    violations = check_forbidden_module_wide_ops(tree)
    result["checks"]["no_forbidden_module_wide_ops"]["violations"] = violations

    if not violations:
        result["checks"]["no_forbidden_module_wide_ops"]["passed"] = True
        return True

    result["checks"]["no_forbidden_module_wide_ops"]["error"] = (
        f"模块级发现 {len(violations)} 处禁止的 PyTorch 接口调用（不只 forward）"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"生成代码借 PyTorch 接口作弊: {violation_details}。"
        "禁用 nn.XXX 构造、F.xxx 调用；"
        "所有计算必须在 @triton.jit kernel 内完成，"
        "权重应从外部传入或用常量初始化，不得借 nn.XXX 复刻 baseline 权重。"
    )
    return False


def _check_kernel_not_placeholder(result, tree, kernel_names, called_names):
    """Type 4/5：被调用的 kernel 是否只是占位；矩阵乘类算子是否有 tl.dot。"""
    kernels = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in kernel_names:
            kernels[node.name] = node

    called = [n for n in (called_names or kernel_names) if n in kernels] or list(kernels)
    placeholders = [n for n in called if kernel_is_placeholder(kernels[n])]
    result["checks"]["kernel_not_placeholder"]["placeholders"] = placeholders

    if placeholders and len(placeholders) == len(called):
        result["checks"]["kernel_not_placeholder"]["error"] = (
            f"被调用的 kernel 全部是占位 kernel（只做 load/store 搬运，无任何计算）: {placeholders}"
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"检测到占位 kernel {placeholders}：它只把数据从输入拷到输出，不参与任何计算，"
            "真实计算仍在 PyTorch 侧。这属于为通过 AST 检测而伪造 Triton 调用。"
            "所有核心计算必须在 @triton.jit kernel 内用 tl.* 完成。"
        )
        return False

    result["checks"]["kernel_not_placeholder"]["passed"] = True
    return True


def _check_no_dynamic_attr_evasion(result, code):
    """填充 no_dynamic_attr_evasion 检查；返回 passed。

    扫整个文件（不只 forward）检测用 getattr 动态访问 torch/nn/F 命名空间
    以规避 nn.XXX / F.xxx 静态黑名单的作弊写法。
    """
    violations = _check_dynamic_attr_evasion(code)
    result["checks"]["no_dynamic_attr_evasion"]["violations"] = violations

    if not violations:
        result["checks"]["no_dynamic_attr_evasion"]["passed"] = True
        return True

    result["checks"]["no_dynamic_attr_evasion"]["error"] = (
        f"检测到 {len(violations)} 处绕过 Triton 实现的 PyTorch/CANN 算子入口"
    )
    details = "; ".join(f"第{v['line']}行 {v['call']}" for v in violations[:5])
    result["regression_type"] = 3
    result["suggestion"] = (
        f"生成代码绕过 Triton 实现直接使用 PyTorch/CANN 算子入口: {details}。"
        "包括用 getattr 动态访问 torch/nn/F/torch_npu 命名空间以规避静态黑名单，"
        "以及直接调用 torch_npu 的 npu_* 融合算子；"
        "所有核心计算必须在 @triton.jit kernel 内用 tl.* 完成。"
    )
    return False


def _check_no_host_math_fallback(result, code):
    """填充 no_host_math_fallback 检查；返回 passed。

    扫整个文件（不只 forward）检测 host 侧是否用 np/torch/tensor 方法
    计算三角/超越函数并作为 Triton kernel 输入表。
    """
    violations = _check_host_math_fallback(code)
    result["checks"]["no_host_math_fallback"]["violations"] = violations

    if not violations:
        result["checks"]["no_host_math_fallback"]["passed"] = True
        return True

    result["checks"]["no_host_math_fallback"]["error"] = (
        f"检测到 {len(violations)} 处 host 侧三角函数/超越函数调用"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"host 侧使用非 Triton 数学库生成 kernel 输入表: {violation_details}。"
        "所有 sin/cos/exp/log 等超越计算必须在 @triton.jit kernel 内用 tl.* intrinsic 实现；"
        "host 侧仅允许 math.cos/math.sin 标量常量初始化。"
    )
    return False


def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = _empty_result(filepath)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["triton_kernel_exists"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # 记录本文件定义的类方法名，供 self.xxx() 规则区分"本类方法"与"nn.Module 前向"
    global _CURRENT_SELF_METHODS
    _CURRENT_SELF_METHODS = _collect_self_methods(tree)

    ok, kernel_names = _check_kernel_exists(result, tree)
    if not ok:
        return result

    ok, forward_node, wrapper_names = _check_forward_calls_kernel(result, tree, kernel_names)
    if not ok:
        return result

    # 模块级检查放在 forward 级之前：模块级作弊（nn.XXX 复刻权重等）
    # 比单纯的 forward 内 PyTorch 退化更严重，应当优先暴露。
    if not _check_no_forbidden_module_wide_ops(result, tree):
        return result

    if not _check_kernel_not_placeholder(result, tree, kernel_names,
                                         result["checks"]["kernel_called_from_forward"].get("called")):
        return result

    # 动态属性规避检查：必须早于其它 PyTorch 入口检查——被 getattr 混淆过的
    # 调用在后续按 qual/attr 匹配的规则里全部解析不出来，会整片漏检。
    if not _check_no_dynamic_attr_evasion(result, code):
        return result

    # Host-side math fallback 检查：必须在 forward 级 PyTorch 退化检查之前
    # 拦截 np.cos / torch.cos / tensor.cos() 等 host 侧生成 kernel 输入表的行为。
    if not _check_no_host_math_fallback(result, code):
        return result

    if not _check_no_forbidden_ops(result, forward_node, kernel_names, wrapper_names, tree):
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_code(path, want_json):
    """读取代码文件；失败时返回 None，由调用方决定退出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        if want_json:
            logger.info("%s", json.dumps({"valid": False, "error": f"文件不存在: {path}"}))
        else:
            logger.error("[ERROR] 文件不存在: %s", path)
        return None


def _emit_pass(result):
    kernels = result["checks"]["triton_kernel_exists"]["kernels"]
    called = result["checks"]["kernel_called_from_forward"]["called"]
    logger.info("[PASS] Triton 实现验证通过")
    logger.info(
        "  - 发现 %d 个 @triton.jit kernel: %s",
        len(kernels),
        ", ".join(k["name"] for k in kernels),
    )
    logger.info("  - forward() 调用: %s", ", ".join(called))
    logger.info("  - forward() 中无禁止的 PyTorch 计算操作")
    logger.info("  - 无 host 侧三角函数/超越函数调用")


def _emit_fail(result):
    rtype = result["regression_type"]
    type_desc = {
        1: "完全无 Triton kernel（纯 PyTorch）",
        2: "有 Triton kernel 但 forward() 未调用",
        3: "部分计算使用 PyTorch（需全部移入 Triton kernel；含 forward 可达的辅助函数）",
        4: "只有占位 kernel（只做 load/store 搬运，为通过检测而伪造 Triton 调用）",
    }
    logger.info("[FAIL] 检测到 PyTorch 退化 — Type %s: %s", rtype, type_desc.get(rtype, "未知"))

    for check_name, check_result in result["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        logger.info("  [%s] %s", status, check_name)
        if check_result["error"]:
            logger.info("         %s", check_result["error"])

    if result["checks"]["no_forbidden_torch_ops"]["violations"] or \
       result["checks"]["no_forbidden_module_wide_ops"]["violations"] or \
       result["checks"]["no_host_math_fallback"]["violations"]:
        logger.info("  违规详情:")
        for check_name in ("no_forbidden_module_wide_ops", "no_host_math_fallback", "no_forbidden_torch_ops"):
            vs = result["checks"][check_name]["violations"]
            if not vs:
                continue
            logger.info("    [%s] 共 %d 处:", check_name, len(vs))
            for v in vs:
                logger.info("      第 %s 行: %s — %s", v["line"], v["call"], v["reason"])

    logger.info("\n  修复建议: %s", result["suggestion"])


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(
        description="检查生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    code = _load_code(args.file, args.json)
    if code is None:
        sys.exit(1)
    result = validate(code, filepath=args.file)

    if args.json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    elif result["valid"]:
        _emit_pass(result)
    else:
        _emit_fail(result)

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
