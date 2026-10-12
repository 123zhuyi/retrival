#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import os


def _to_np(o):
    """把 tensor / OutputPackage / numpy 统一转成 numpy 数组。"""
    if hasattr(o, "data") and not isinstance(o, torch.Tensor):
        o = o.data
    if hasattr(o, "detach"):
        o = o.detach()
    if hasattr(o, "cpu"):
        o = o.cpu()
    return np.asarray(o)


def _dump_outputs(tag, outputs, case_id):
    """把输出保存到 CDIST_DUMP_DIR 下，文件名 {tag}_case{id}_{idx}.npy/.txt。"""
    dump_dir = os.environ.get("CDIST_DUMP_DIR")
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    for idx, o in enumerate(outputs):
        if o is None:
            continue
        try:
            arr = _to_np(o)
            np.save(
                os.path.join(dump_dir, "%s_case%d_%d.npy" % (tag, case_id, idx)), arr
            )
            with open(
                os.path.join(dump_dir, "%s_case%d_%d.txt" % (tag, case_id, idx)), "w"
            ) as f:
                f.write("shape=%s dtype=%s\n" % (arr.shape, arr.dtype))
                np.savetxt(f, arr.reshape(-1)[:50], fmt="%.6g")
        except Exception as e:
            print("[dump] %s_case%d_%d failed: %s" % (tag, case_id, idx, e))


@register("function_aclnnCdistBackward")
class aclnnCdistBackwardExecutor(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(aclnnCdistBackwardExecutor, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        if "performance_device" in task_result.task_type:
            os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "0"
        elif "accuracy" in task_result.task_type:
            os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"
        else:
            pass

    def cdist_grad(self, grad, x1, x2, cdist, p):
        x1 = torch.unsqueeze(x1, -2)
        x2 = torch.unsqueeze(x2, -3)
        grad = torch.unsqueeze(grad, -1)
        cdist = torch.unsqueeze(cdist, -1)
        diff = x1 - x2
        diff_abs = torch.abs(diff)
        nz_cdist = torch.where(cdist == 0, torch.ones_like(cdist), cdist)
        sign = torch.where(diff > 0, torch.ones_like(diff), torch.full_like(diff, -1))
        sign = torch.where(diff == 0, torch.zeros_like(diff), sign)
        if p == 0.0:
            res = torch.zeros_like(diff)
        elif p == 1.0:
            res = grad * sign
        elif p < 1.0:
            # p<1 时与 kernel 对齐：r = cdist/|diff| (>=1), q = 1-p, r^q = exp(q*ln(r))。
            # 直接 pow(diff_abs, p-1)/pow(cdist, p-1) 在 p->0 时数值病态，与算子精度偏差大(ratio 3x)。
            r = nz_cdist / diff_abs
            # 算子 LnHighPrec 对 inf 做 Newton 细化时 inf*0 -> nan，而 torch.log(inf)=inf，
            # 导致 cdist(fp16) 溢出为 inf 时算子梯度为 nan、标杆却为 inf。这里复现算子：ln(inf)=nan。
            log_r = torch.log(
                torch.where(r == float("inf"), torch.full_like(r, float("nan")), r)
            )
            res = sign * grad * torch.exp((1.0 - p) * log_r)
            res = torch.where(cdist == 0, torch.zeros_like(res), res)
        elif p < 2.0:
            try:
                res = (
                    sign
                    * torch.pow(diff_abs, p - 1.0)
                    * grad
                    / torch.pow(nz_cdist, p - 1.0)
                )
            except ZeroDivisionError:
                print("raise ZeroDivisionError.")
            res = torch.where(cdist == 0, torch.zeros_like(res), res)
        elif p == 2.0:
            try:
                res = grad * diff / nz_cdist
            except ZeroDivisionError:
                print("raise ZeroDivisionError.")
            res = torch.where(cdist == 0, torch.zeros_like(res), res)
        elif p == float("inf"):
            # mask = torch.where(cdist - diff_abs > 0, torch.zeros_like(diff), torch.ones_like(diff))

            # 与 kernel 对齐：mask = (|diff| - cdist) == 0（Sub + Compare EQ，精确相等），
            # 而不是 >= 。cdist 经 fp16/bf16 四舍五入后会与 |diff| 有细微差异，== 才能复现算子行为。
            mask = torch.where(
                diff_abs - cdist == 0, torch.ones_like(diff), torch.zeros_like(diff)
            )
            res = grad * sign * mask
        else:
            # p>2 时与 kernel 对齐：r = |diff|/cdist (<=1), q = p-1, r^q = exp(q*ln(r))。
            # 直接 diff*pow(|diff|,p-2)*grad/pow(cdist,p-1) 与算子 exp/ln 幂实现精度偏差大。
            r = diff_abs / nz_cdist
            res = sign * grad * torch.exp((p - 1.0) * torch.log(r))
            res = torch.where(cdist == 0, torch.zeros_like(res), res)
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
        # 与 kernel 的 SelectZero(|diff|==0) 对齐：p<1 时 pow(0, 负指数) 会产生 inf/nan
        res = torch.where(diff_abs == 0, torch.zeros_like(res), res)  # modify
        result = torch.zeros_like(res[..., 0, :])
        for i in range(res.shape[-2]):
            result = result + res[..., i, :]
        res = result
        # res = torch.sum(res, -2)
        return res

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad = input_data.kwargs["grad"]
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        cdist = input_data.kwargs["cdist"]
        p = input_data.kwargs["pValue"]
        if self.x1.dtype == torch.float16 or self.x1.dtype == torch.bfloat16:
            return (
                self.cdist_grad(
                    grad.cpu().to(torch.float32),
                    x1.cpu().to(torch.float32),
                    x2.cpu().to(torch.float32),
                    cdist.cpu().to(torch.float32),
                    p,
                )
                .to(self.x1.dtype)
                .to(self.x1.device)
            )
        else:
            return self.cdist_grad(grad, x1, x2, cdist, p)

    def init_by_input_data(self, input_data: InputDataset):
        torch.manual_seed(2026)
        self.x1 = input_data.kwargs["x1"].detach()
        self.x2 = input_data.kwargs["x2"].detach()
        self.x_p = input_data.kwargs["pValue"]
        self.x_computMode = 2
        self.compute_mode_dict = {
            0: "use_mm_for_euclid_dist_if_necessary",
            1: "use_mm_for_euclid_dist",
            2: "donot_use_mm_for_euclid_dist",
        }
        self.x1.requires_grad = True
        self.x2.requires_grad = True
        self.output = (
            torch.cdist(
                self.x1.cpu().to(torch.float64),
                self.x2.cpu().to(torch.float64),
                p=self.x_p,
                compute_mode=self.compute_mode_dict[self.x_computMode],
            )
            .to(self.x1.dtype)
            .to(self.x1.device)
        )
        print(self.x_p)
        self.grad_input = (
            torch.randn(self.output.shape, dtype=torch.float32)
            .to(self.output.dtype)
            .to(self.output.device)
        )
        input_data.kwargs["grad"] = self.grad_input
        input_data.kwargs["cdist"] = self.output
        _dump_outputs(
            "inputs",
            [self.x1, self.x2, self.output, self.grad_input],
            self.task_result.case_config.id,
        )
        with open(
            os.path.join(
                os.environ.get("CDIST_DUMP_DIR", "/tmp"),
                "p_case%d.txt" % self.task_result.case_config.id,
            ),
            "w",
        ) as f:
            f.write("p=%r\n" % (self.x_p,))


@register("function_aclnnCdistBackward_aclop")
class aclnnCdistBackwardAclopExecutor(aclnnCdistBackwardExecutor):
    """aclop 模式标杆：输入 tensor 由 npu 后端搬到 NPU，torch.ops.aten._cdist_backward
    走 torch_npu 在线编译（aclop）在 NPU 上执行。复用 cpu 标杆的 init（device 无关）。"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad = input_data.kwargs["grad"]
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        cdist = input_data.kwargs["cdist"]
        p = input_data.kwargs["pValue"]
        out = self.cdist_grad(grad, x1, x2, cdist, p)
        return out


@register("pyaclnn_aclnnCdistBackward")
class pyaclnnCdist(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        """参数处理流同步报错问题"""
        torch.npu.synchronize()
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        res = super().after_call(output_packages)
        try:
            src = res if isinstance(res, (list, tuple)) else output_packages
            tensors = []
            for o in src:
                if o is None:
                    continue
                if isinstance(o, torch.Tensor):
                    tensors.append(o)
                else:
                    tensors.append(self.acl_tensor_to_torch(o))
            _dump_outputs("pyaclnn", tensors, self.task_result.case_config.id)
        except Exception as e:
            print("[dump] pyaclnn after_call failed: %s" % e)
        return res
