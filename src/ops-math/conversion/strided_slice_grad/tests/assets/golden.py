#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""StridedSliceGrad TestSpec golden for TTK (kernel / GEIR / ACLNN / E2E-TF).

真值一律由 TensorFlow 算子 ``tf.raw_ops.StridedSliceGrad`` 计算，与旧 golden
（``ttk/user_defined_modules/op/golden_funcs/strided_slice_grad.py``）语义一致，
仅适配到当前 TTK 的 TestSpec（``__spec__``）框架。不使用 numpy 重实现算法。

注意：``import tensorflow`` 采用惰性导入（放在函数内部），因此在未安装 TF 的环境里
本文件仍可被 loader 静态索引、被 CSV 校验加载；只有真正生成真值时才要求 TF 可用。

各通路 golden 入参/出参类型（见 ttk-how-write-plugin）：
  - kernel / GEIR : op_name  ``strided_slice_grad``   → 输入 numpy.ndarray，返回 numpy
  - ACLNN         : api_name ``aclnnStridedSliceGrad`` → 输入 torch.Tensor，返回 torch
  - E2E (tf)      : api_name ``tf.raw_ops.StridedSliceGrad`` → 输入 torch.Tensor，返回 torch

参数顺序：
  - kernel/GEIR/E2E 张量入参依 def.cpp / tf 签名：shape, begin, end, strides, dy；mask 走 kwargs
  - ACLNN 依 aclnn 头：shape/begin/end/strides 是 aclIntArray（经 attributes 传入），
    dy、out 是张量；mask（beginMask...）经 attributes 传入
"""

__spec__ = {
    # kernel 与 GEIR 共用 snake_case 注册名与同一 TestSpec。
    "strided_slice_grad": "StridedSliceGradKernelSpec",
    "aclnnStridedSliceGrad": "AclnnStridedSliceGradSpec",
    "tf.raw_ops.StridedSliceGrad": "TfStridedSliceGradSpec",
}

# 兼容旧仓库入口，迁移期保留。
__golden__ = {
    "kernel": {"strided_slice_grad": "strided_slice_grad_golden"},
}


_FLOAT_TOL = {
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
    "float64": {"standard": "stat_rel_err"},
    "complex32": {"standard": "stat_rel_err"},
    "complex64": {"standard": "stat_rel_err"},
    "complex128": {"standard": "stat_rel_err"},
}
# StridedSliceGrad 是纯 scatter/拷贝，整型结果按位精确。
_INT_TOL = {
    dt: {"standard": "binary_equal"}
    for dt in (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "bool",
    )
}
_TOLERANCE = {**_FLOAT_TOL, **_INT_TOL}


def _to_int_list(v):
    """把 shape/begin/end/strides 参数（numpy 数组 / torch 张量 / list / 标量）归一为 python int 列表。"""
    if v is None:
        return None
    if hasattr(v, "tolist"):  # numpy.ndarray / torch.Tensor
        v = v.tolist()
    if isinstance(v, (int, float)):
        return [int(v)]
    return [int(x) for x in v]


def _value_dtype_name(dy):
    """从 dy 张量取 dtype 名（'float16'/'bfloat16'/'complex64'/...）。numpy 与 torch 通用。"""
    dt = getattr(dy, "dtype", None)
    if dt is None:
        return "float32"
    # numpy: dt.name；torch: str(dt) 形如 'torch.float32'
    name = getattr(dt, "name", None) or str(dt)
    name = name.split(".")[-1].lower()
    aliases = {"double": "float64", "half": "float16"}
    return aliases.get(name, name)


def _tf_value_dtype(tf, name):
    """dtype 名 → tf dtype，用于 dy/输出。"""
    table = {
        "float16": tf.float16,
        "bfloat16": tf.bfloat16,
        "float32": tf.float32,
        "float64": tf.float64,
        "int8": tf.int8,
        "uint8": tf.uint8,
        "int16": tf.int16,
        "uint16": tf.uint16,
        "int32": tf.int32,
        "uint32": tf.uint32,
        "int64": tf.int64,
        "uint64": tf.uint64,
        # TF 无 complex32；StridedSliceGrad 是纯 scatter/拷贝，用 complex64 计算再由
        # _tf_result_to_torch 按 ref.dtype 转回 complex32，结果按位精确。
        "complex32": tf.complex64,
        "complex64": tf.complex64,
        "complex128": tf.complex128,
        "bool": tf.bool,
    }
    if name not in table:
        raise ValueError(
            f"StridedSliceGrad golden: unsupported dy dtype '{name}' for tf.raw_ops"
        )
    return table[name]


def _compute_tf(
    shape,
    begin,
    end,
    strides,
    dy,
    begin_mask,
    end_mask,
    ellipsis_mask,
    new_axis_mask,
    shrink_axis_mask,
):
    """核心：调用 tf.raw_ops.StridedSliceGrad 生成真值，返回 tf 张量（eager）。

    ``dy`` 可以是 numpy.ndarray 或 torch.Tensor；一律经 ``.tolist()`` 转 python 原生
    嵌套列表后交给 tf.constant（同时规避 numpy bfloat16/complex 的 dtype 兼容问题，
    且不引入 numpy 计算）。
    """
    import tensorflow as tf

    shape_l = _to_int_list(shape)
    begin_l = _to_int_list(begin)
    end_l = _to_int_list(end)
    strides_l = _to_int_list(strides)

    value_name = _value_dtype_name(dy)
    tf_value_dt = _tf_value_dtype(tf, value_name)

    # dy 转 python 嵌套 list 再建常量，保证 dtype 与 shape 精确重建。
    dy_list = dy.tolist() if hasattr(dy, "tolist") else dy
    dy_t = tf.constant(dy_list, dtype=tf_value_dt)
    # 空张量修正：dy 含 0 维且其后有非 0 维时（如 [0,5]/[3,0,4]），tolist()→tf.constant
    # 会丢失尾部维度（推断成 [0]/[3,0]），触发 StridedSliceGrad shape 校验失败。
    # dy 有已知 shape 时按其 reshape 回真实形状。
    dy_shape = getattr(dy, "shape", None)
    if dy_shape is not None:
        dy_shape = list(dy_shape)
        if list(dy_t.shape) != dy_shape:
            dy_t = tf.reshape(dy_t, dy_shape)

    # shape/begin/end/strides 为 Index 类型，统一 int64。
    idx_dt = tf.int64
    res = tf.raw_ops.StridedSliceGrad(
        shape=tf.constant(shape_l, dtype=idx_dt),
        begin=tf.constant(begin_l, dtype=idx_dt),
        end=tf.constant(end_l, dtype=idx_dt),
        strides=tf.constant(strides_l, dtype=idx_dt),
        dy=dy_t,
        begin_mask=int(begin_mask),
        end_mask=int(end_mask),
        ellipsis_mask=int(ellipsis_mask),
        new_axis_mask=int(new_axis_mask),
        shrink_axis_mask=int(shrink_axis_mask),
    )
    return res


# --------------------------------------------------------------------------- #
# Kernel / GEIR flow —— 输入 numpy.ndarray，返回 numpy。
# 入参顺序依 strided_slice_grad_def.cpp（仅输入）：shape, begin, end, strides, dy。
# 前四个是 ValueDepend 张量（值由 CSV attributes 指定），mask 走 kwargs。
# --------------------------------------------------------------------------- #
def strided_slice_grad_golden(
    shape,
    begin,
    end,
    strides,
    dy,
    *,
    begin_mask=0,
    end_mask=0,
    ellipsis_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
    **kwargs,
):
    """Kernel/GEIR golden：tf 计算真值，返回 [numpy.ndarray]。"""
    res = _compute_tf(
        shape,
        begin,
        end,
        strides,
        dy,
        begin_mask,
        end_mask,
        ellipsis_mask,
        new_axis_mask,
        shrink_axis_mask,
    )
    return [res.numpy()]


class StridedSliceGradKernelSpec:
    """kernel 与 GEIR 通路共用（numpy 真值，tf 计算）。"""

    @staticmethod
    def golden(
        shape,
        begin,
        end,
        strides,
        dy,
        *,
        begin_mask=0,
        end_mask=0,
        ellipsis_mask=0,
        new_axis_mask=0,
        shrink_axis_mask=0,
        **kwargs,
    ):
        return strided_slice_grad_golden(
            shape,
            begin,
            end,
            strides,
            dy,
            begin_mask=begin_mask,
            end_mask=end_mask,
            ellipsis_mask=ellipsis_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask,
            **kwargs,
        )

    tolerance = _TOLERANCE


def _tf_result_to_torch(res, ref):
    """把 tf 结果张量转成与 ref（torch.Tensor）同 device/dtype 的 torch.Tensor，不经 numpy 计算。"""
    import torch

    res_shape = list(res.shape)
    data = res.numpy().tolist()  # tf → python 原生嵌套 list（保持 shape）
    out = torch.tensor(data, dtype=ref.dtype, device=ref.device)
    # 空张量修正：结果含 0 维且其后有非 0 维时（如 [3,0,4]），tolist()→torch.tensor
    # 会丢失尾部维度（塌成 [3,0]）。按 tf 结果的已知 shape reshape 回真实秩。
    if list(out.shape) != res_shape:
        out = out.reshape(res_shape)
    return out


# --------------------------------------------------------------------------- #
# 竞品（XPU/GPU）执行入口 —— 经 AclnnStridedSliceGradSpec.third_party 映射。
# TTK 的 --xpu-perf 只送张量入参 dy（shape/begin/end/strides 是 ValueDepend，
# 走 attributes）+ attrs（shape/begin/end/strides + 驼峰 masks）。
# _invoke_function 阶段一以 `fn(**{attrs∪named})` 绑定，故本函数形参名须与
# CSV attributes 的键（驼峰 mask）及 dy 输入名完全一致。竞品用
# tf.raw_ops.StridedSliceGrad 在 GPU 上算真值，TTK 采集其 XPU 侧耗时。
# --------------------------------------------------------------------------- #
def _xpu_strided_slice_grad(
    dy,
    shape,
    begin,
    end,
    strides,
    beginMask=0,
    endMask=0,
    ellipsisMask=0,
    newAxisMask=0,
    shrinkAxisMask=0,
    **kwargs,
):
    return _compute_tf(
        shape,
        begin,
        end,
        strides,
        dy,
        beginMask,
        endMask,
        ellipsisMask,
        newAxisMask,
        shrinkAxisMask,
    )


# --------------------------------------------------------------------------- #
# ACLNN flow —— aclnnStridedSliceGradGetWorkspaceSize(shape, begin, end, strides,
# dy, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, out, ...)。
# shape/begin/end/strides 是 aclIntArray（经 attributes 传入），dy、out 是张量。
# golden 依 C 头顺序收参；张量为 torch.Tensor，返回 torch.Tensor。
# --------------------------------------------------------------------------- #
class AclnnStridedSliceGradSpec:
    """ACLNN 通路（torch 真值，tf 计算）。参数名/序与 aclnn 头一致。"""

    @staticmethod
    def golden(
        shape,
        begin,
        end,
        strides,
        dy,
        beginMask=0,
        endMask=0,
        ellipsisMask=0,
        newAxisMask=0,
        shrinkAxisMask=0,
        out=None,
        **kwargs,
    ):
        res = _compute_tf(
            shape,
            begin,
            end,
            strides,
            dy,
            beginMask,
            endMask,
            ellipsisMask,
            newAxisMask,
            shrinkAxisMask,
        )
        return [_tf_result_to_torch(res, dy)]

    tolerance = _TOLERANCE
    # 竞品性能对比（--xpu-perf）：映射到 tf 竞品实现（torch 无 strided_slice_grad）。
    third_party = {"tf": _xpu_strided_slice_grad}


# --------------------------------------------------------------------------- #
# E2E / TensorFlow flow —— tf.raw_ops.StridedSliceGrad(shape, begin, end,
# strides, dy, begin_mask=..., ...)。张量入参按 tf 签名位置传入（torch.Tensor），
# mask 走 attributes/kwargs。golden 返回 torch.Tensor。
# --------------------------------------------------------------------------- #
class TfStridedSliceGradSpec:
    """E2E 通路：tf.raw_ops.StridedSliceGrad（torch 入/出，tf 计算）。"""

    @staticmethod
    def golden(
        shape,
        begin,
        end,
        strides,
        dy,
        *,
        begin_mask=0,
        end_mask=0,
        ellipsis_mask=0,
        new_axis_mask=0,
        shrink_axis_mask=0,
        **kwargs,
    ):
        res = _compute_tf(
            shape,
            begin,
            end,
            strides,
            dy,
            begin_mask,
            end_mask,
            ellipsis_mask,
            new_axis_mask,
            shrink_axis_mask,
        )
        return [_tf_result_to_torch(res, dy)]

    tolerance = _TOLERANCE
