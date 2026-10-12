/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mod_copy_impl.h
 * \brief Mod<T,ST,OT>'s TQue-based copy-in/out + generic-broadcast offset helper definitions
 *        (GetInput2Offset / GetInput2ContiguousCopyCount / CopyIn / CopyOut).
 *
 * 四个 TQue-path 辅助 (GetInput2Offset / GetInput2ContiguousCopyCount / CopyIn / CopyOut，ProcessBroadcast
 * 全平台用、ProcessContiguous 非 arch22 用) 自包含，拆出到此文件 (纯物理迁移，无语句/操作数/分支条件改动，
 * 含 GetInput2ContiguousCopyCount 的纯广播尾轴 tiling 修复)。同其它 _impl.h 惯例：不自带 namespace，从 mod.h
 * 的 `namespace ModNs` 内 #include -> 定义附着到 ModNs::Mod<T,ST,OT>。
 */
#ifndef MOD_COPY_IMPL_H
#define MOD_COPY_IMPL_H

template <typename T, typename ST, typename OT>
__aicore__ inline uint64_t Mod<T, ST, OT>::GetInput2Offset(const uint64_t outputOffset)
{
    uint64_t remaining = outputOffset;
    uint64_t input2Offset = 0;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        uint64_t coord = 0;
        uint64_t dimSize = input1Shape[i];
        if (dimSize > 0) {
            coord = remaining % dimSize;
            remaining = remaining / dimSize;
        }
        input2Offset += coord * input2Stride[i];
    }
    return input2Offset;
}

template <typename T, typename ST, typename OT>
__aicore__ inline uint32_t Mod<T, ST, OT>::GetInput2ContiguousCopyCount(const uint64_t outputOffset,
                                                                        const uint32_t remainingCount,
                                                                        bool& isConstantRun)
{
    // 纯广播尾轴 tiling 修复：原循环把 input2Stride[i]==0 的纯广播维 (input2Shape[i]==1) 当成硬非连续断点，
    //   导致尾维纯广播时 suffixSize 恒为 1、每个广播 tile 退化成单元素迭代 (极端 shape 下表现为超长耗时)。
    //   修复：在内层前缀内 (未遇到真实匹配 stride 维 !sawRealDim 时) 把纯广播尾维识别为可折叠 "常量段"——x2 在
    //   整段内是单一值，故 tile 可批处理整段 (至 remainingCount) 而非单元素。count = suffixSize - (outputOffset %
    //   suffixSize) (公式不变) 保证不越出当前常量块。一旦遇到真实变化维 (input2Stride[i]!=0) 即停止折叠——
    //   isConstantRun 告知 CopyIn 改用 Duplicate 填充该常量。纯 tiling 粒度改动，不改 K1/K2/数值结果；最内维
    //   非纯广播的 shape 走 sawRealDim 原路径不变 -> 零回归。
    isConstantRun = false;
    bool sawRealDim = false;
    uint64_t suffixSize = 1;
    uint64_t expectedStride = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        if (input1Shape[i] == 1) {
            continue;
        }
        if (input2Stride[i] == 0 && !sawRealDim) {
            suffixSize *= input1Shape[i];
            isConstantRun = true;
            continue;
        }
        if (isConstantRun) {
            break; // genuinely-varying dim reached after a folded broadcast prefix -> stop the constant run
        }
        if (input2Stride[i] != expectedStride) {
            break;
        }
        sawRealDim = true;
        suffixSize *= input1Shape[i];
        expectedStride *= input2Shape[i];
    }
    uint64_t count = suffixSize - (outputOffset % suffixSize);
    if (count > remainingCount) {
        count = remainingCount;
    }
    return static_cast<uint32_t>(count);
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::CopyIn(const uint64_t offset, const int32_t calCount, const bool isConstantX2)
{
    LocalTensor<ST> datax1Local = inputx1Queue.AllocTensor<ST>();
    LocalTensor<OT> datax2Local = inputx2Queue.AllocTensor<OT>();
    const int32_t alignedCalCount = CeilAlign(calCount, DATA_BLOCK);

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = calCount * sizeof(ST);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(datax1Local, inputx1GM[offset], copyParams, {false, 0, 0, 0});
    if (isInput2Scalar) {
        DataCopyParams scalarCopyParams;
        scalarCopyParams.blockCount = 1;
        scalarCopyParams.blockLen = sizeof(OT);
        scalarCopyParams.srcStride = 0;
        scalarCopyParams.dstStride = 0;
        DataCopyPad(datax2Local, inputx2GM[0], scalarCopyParams, {false, 0, 0, 0});
        OT scalarValue = datax2Local.GetValue(0);
        Duplicate(datax2Local, scalarValue, alignedCalCount);
    } else if (!isInput2SameShape && isConstantX2) {
        // GetInput2ContiguousCopyCount folded a trailing
        // pure-broadcast run (input2Stride[i]==0) into `calCount` -> x2 is a SINGLE value across this whole
        // tile. Read that ONE value + Duplicate (byte-identical pattern to the isInput2Scalar branch above,
        // just addressed via GetInput2Offset instead of a fixed [0]) instead of a matching-count DataCopyPad
        // (which would be wrong: x2's GM buffer has no such contiguous run). Same value, same result as the
        // old per-element (count==1) loop -- only the tile granularity changes.
        DataCopyParams constCopyParams;
        constCopyParams.blockCount = 1;
        constCopyParams.blockLen = sizeof(OT);
        constCopyParams.srcStride = 0;
        constCopyParams.dstStride = 0;
        DataCopyPad(datax2Local, inputx2GM[GetInput2Offset(offset)], constCopyParams, {false, 0, 0, 0});
        OT constValue = datax2Local.GetValue(0);
        Duplicate(datax2Local, constValue, alignedCalCount);
    } else if (!isInput2SameShape) {
        DataCopyParams input2CopyParams;
        input2CopyParams.blockCount = 1;
        input2CopyParams.blockLen = calCount * sizeof(OT);
        input2CopyParams.srcStride = 0;
        input2CopyParams.dstStride = 0;
        DataCopyPad(datax2Local, inputx2GM[GetInput2Offset(offset)], input2CopyParams, {false, 0, 0, 0});
    } else {
        DataCopyParams input2SameParams;
        input2SameParams.blockCount = 1;
        input2SameParams.blockLen = calCount * sizeof(OT);
        input2SameParams.srcStride = 0;
        input2SameParams.dstStride = 0;
        DataCopyPad(datax2Local, inputx2GM[offset], input2SameParams, {false, 0, 0, 0});
    }

    inputx1Queue.EnQue(datax1Local);
    inputx2Queue.EnQue(datax2Local);
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::CopyOut(const uint64_t offset, const int32_t calCount)
{
    LocalTensor<T> dstLocal = outputQueue.DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = calCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM[offset], dstLocal, copyParams);

    outputQueue.FreeTensor(dstLocal);
}

#endif // MOD_COPY_IMPL_H
