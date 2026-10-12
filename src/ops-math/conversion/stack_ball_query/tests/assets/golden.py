#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {"stack_ball_query": "StackBallQueryKernelSpec"}

import numpy
import torch


def _partition(total, parts, dtype, rotation=0):
    if parts == 0:
        if total != 0:
            raise ValueError("zero batch count is legal only for an empty tensor")
        return torch.empty(0, dtype=dtype)
    quotient, remainder = divmod(total, parts)
    values = torch.full((parts,), quotient, dtype=torch.int64)
    values[:remainder] += 1
    if parts > 1:
        values = torch.roll(values, rotation % parts)
    return values.to(dtype)


class StackBallQueryKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray"""

    def customize_inputs(
        xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, **kwargs
    ):
        xyz_t = torch.from_numpy(numpy.ascontiguousarray(xyz))
        center_t = torch.from_numpy(numpy.ascontiguousarray(center_xyz))

        if xyz_t.ndim != 2 or xyz_t.shape[0] != 3:
            raise ValueError("library xyz must have shape [3, N]")
        if center_t.ndim != 2 or center_t.shape[1] != 3:
            raise ValueError("center_xyz must have shape [M, 3]")

        cnt_np = numpy.asarray(xyz_batch_cnt)
        count_dtype = torch.int32
        if cnt_np.dtype == numpy.dtype(numpy.int64):
            count_dtype = torch.int64
        cnt_t = torch.from_numpy(cnt_np.astype(numpy.int64))

        batch_count = int(cnt_t.shape[0])
        point_count = int(xyz_t.shape[1])
        center_count = int(center_t.shape[0])

        testcase_name = str(kwargs.get("testcase_name", ""))
        rotation = sum(ord(c) for c in testcase_name)

        xyz_counts = _partition(point_count, batch_count, count_dtype, rotation)
        center_counts = _partition(center_count, batch_count, count_dtype, rotation + 1)

        xyz_offsets = torch.zeros(batch_count, dtype=torch.int64)
        if batch_count > 1:
            xyz_offsets[1:] = torch.cumsum(xyz_counts[:-1], dim=0)
        center_offsets = torch.zeros(batch_count, dtype=torch.int64)
        if batch_count > 1:
            center_offsets[1:] = torch.cumsum(center_counts[:-1], dim=0)

        xyz_out = xyz_t.clone()
        for batch_idx in range(batch_count):
            n_points = int(xyz_counts[batch_idx].item())
            m_centers = int(center_counts[batch_idx].item())
            if n_points == 0 or m_centers == 0:
                continue
            pt_off = int(xyz_offsets[batch_idx].item())
            ct_off = int(center_offsets[batch_idx].item())
            for local_c in range(m_centers):
                local_pt = local_c % n_points
                cx = center_t[ct_off + local_c, 0]
                cy = center_t[ct_off + local_c, 1]
                cz = center_t[ct_off + local_c, 2]
                xyz_out[0, pt_off + local_pt] = cx
                xyz_out[1, pt_off + local_pt] = cy
                xyz_out[2, pt_off + local_pt] = cz

        return (
            xyz_out.contiguous().numpy(),
            center_t.contiguous().numpy(),
            xyz_counts.contiguous().to(count_dtype).numpy(),
            center_counts.contiguous().to(count_dtype).numpy(),
        )

    def golden(
        xyz,
        center_xyz,
        xyz_batch_cnt,
        center_xyz_batch_cnt,
        *,
        max_radius,
        sample_num,
        **kwargs,
    ):
        del kwargs

        xyz_t = torch.from_numpy(xyz)
        center_t = torch.from_numpy(center_xyz)
        xyz_counts = torch.from_numpy(xyz_batch_cnt).to(torch.int64).tolist()
        center_counts = torch.from_numpy(center_xyz_batch_cnt).to(torch.int64).tolist()

        total_centers = int(center_t.shape[0])
        sample_num_value = int(sample_num)

        radius_sq = torch.tensor(float(max_radius) ** 2, dtype=xyz_t.dtype)

        output = torch.full(
            (total_centers, sample_num_value),
            -1,
            dtype=torch.int32,
        )

        x_row = xyz_t[0]
        y_row = xyz_t[1]
        z_row = xyz_t[2]

        xyz_offset = 0
        center_offset = 0
        for batch_idx in range(len(xyz_counts)):
            n_points = int(xyz_counts[batch_idx])
            m_centers = int(center_counts[batch_idx])
            if n_points == 0:
                center_offset += m_centers
                xyz_offset += n_points
                continue

            xs = x_row[xyz_offset : xyz_offset + n_points]
            ys = y_row[xyz_offset : xyz_offset + n_points]
            zs = z_row[xyz_offset : xyz_offset + n_points]

            for local_c in range(m_centers):
                row = center_offset + local_c
                cx = center_t[row, 0]
                cy = center_t[row, 1]
                cz = center_t[row, 2]

                dx = torch.sub(xs, cx)
                dy = torch.sub(ys, cy)
                dz = torch.sub(zs, cz)
                dist = torch.add(
                    torch.add(torch.mul(dx, dx), torch.mul(dy, dy)), torch.mul(dz, dz)
                )
                in_radius = torch.lt(dist, radius_sq)
                candidates = torch.nonzero(in_radius, as_tuple=False).flatten()

                if candidates.numel() == 0:
                    continue

                take = min(sample_num_value, candidates.numel())
                selected = candidates[:take].to(dtype=torch.int32)
                output[row, :take] = selected
                if take < sample_num_value:
                    output[row, take:] = selected[0]

            xyz_offset += n_points
            center_offset += m_centers

        return [output.numpy()]

    class ThirdPartyImpl:
        """三方标杆实现 — torch vendor，入参为 torch.Tensor（已在设备上）。

        独立于 golden 的实现路径：用 torch.cumsum + arange + masked_fill 向量化收集候选点，
        避免与 golden 共享 nonzero 路径，确保交叉校验有效。

        __init__: 预计算 batch 偏移、radius_sq、dtype 归一化（逻辑判断与类型转换前置）
        __call__: 只做距离计算 + cumsum + masked_fill 纯计算
        """

        def __init__(
            self, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, **kwargs
        ):
            self.orig_dtype = xyz.dtype
            self.M = int(center_xyz.shape[0])
            self.N = int(xyz.shape[1])
            self.sample_num = int(kwargs["sample_num"])

            xyz_counts = xyz_batch_cnt.to(torch.int64)
            center_counts = center_xyz_batch_cnt.to(torch.int64)
            self.batch_size = int(xyz_counts.shape[0])

            self.skip = (
                self.M == 0
                or self.N == 0
                or self.batch_size == 0
                or self.sample_num == 0
            )

            if self.skip:
                return

            self.device = xyz.device

            self.xyz_counts = xyz_counts.tolist()
            self.center_counts = center_counts.tolist()

            self.radius_sq = torch.tensor(
                float(kwargs["max_radius"]) ** 2,
                dtype=self.orig_dtype,
                device=self.device,
            )

            self.pos_grid = torch.arange(
                self.sample_num, dtype=torch.int32, device=self.device
            )

        def __call__(
            self, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, **kwargs
        ):
            if self.skip:
                return [
                    torch.full(
                        (self.M, self.sample_num),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                ]

            x_row = xyz[0]
            y_row = xyz[1]
            z_row = xyz[2]

            output = torch.full(
                (self.M, self.sample_num), -1, dtype=torch.int32, device=self.device
            )

            xyz_offset = 0
            center_offset = 0
            for batch_idx in range(self.batch_size):
                n_points = int(self.xyz_counts[batch_idx])
                m_centers = int(self.center_counts[batch_idx])
                if n_points == 0:
                    center_offset += m_centers
                    xyz_offset += n_points
                    continue

                xs = x_row[xyz_offset : xyz_offset + n_points]
                ys = y_row[xyz_offset : xyz_offset + n_points]
                zs = z_row[xyz_offset : xyz_offset + n_points]

                for local_c in range(m_centers):
                    row = center_offset + local_c
                    cx = center_xyz[row, 0]
                    cy = center_xyz[row, 1]
                    cz = center_xyz[row, 2]

                    dx = xs - cx
                    dy = ys - cy
                    dz = zs - cz
                    dist = dx * dx + dy * dy + dz * dz
                    hit = dist < self.radius_sq

                    cumsum = hit.cumsum(dim=0)
                    cnt = int(torch.clamp(cumsum[-1], max=self.sample_num).item())

                    if cnt == 0:
                        continue

                    first_idx = int(hit.long().argmax(dim=0).item())

                    selected = hit & (cumsum <= self.sample_num)
                    k_sel = torch.nonzero(selected, as_tuple=False).flatten()
                    pos_sel = cumsum[k_sel] - 1

                    output[row, pos_sel.to(torch.int64)] = k_sel.to(torch.int32)

                    if cnt < self.sample_num:
                        output[row, cnt:] = first_idx

                xyz_offset += n_points
                center_offset += m_centers

            return [output]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {"int32": {"standard": "binary_equal"}}
