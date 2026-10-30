/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef CALINDEX_UNIFIED_H
#define CALINDEX_UNIFIED_H

#include <cstdint>
constexpr int32_t kMaxTensorDims = 8;
//---------------------------------------
// 计算 strides
//---------------------------------------
__aicore__ inline void ComputeStrides(const uint32_t shape[], uint32_t dim, uint32_t strides[])
{
    if (dim == 0)
        return;
    strides[dim - 1] = 1;
    for (int32_t i = static_cast<int32_t>(dim) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

//---------------------------------------
// 均分切分反算全局索引
//---------------------------------------
// srcShape: 原 tensor shape
// srcdim: 维度数
// axis_in: 切分轴
// sections: slice 数量
// localIndex: slice 内局部索引
// sliceIdx: 当前 slice 序号
// unit: axis 外元素乘积
__aicore__ inline bool NormalizeAxis(const uint32_t srcdim, const int64_t& axis_in, uint32_t& axis_out)
{
    if (srcdim == 0)
        return false;
    axis_out = static_cast<uint32_t>(axis_in);
    return axis_out < srcdim;
}

__aicore__ inline uint32_t CalIndexEven(
    const uint32_t srcShape[], uint32_t srcdim, int64_t axis_in, uint32_t sections, uint32_t localIndex,
    uint32_t sliceIdx, uint32_t unit)
{
    if (srcdim == 0 || sections == 0)
        return 0;
    if (unit == 0)
        return 0;
    if (axis_in < 0)
        return localIndex;

    uint32_t axis = 0;
    if (!NormalizeAxis(srcdim, axis_in, axis))
        return 0;

    uint32_t axisLen = srcShape[axis];
    if (axisLen == 0)
        return 0;

    if (axisLen % sections != 0)
        return 0;
    uint32_t sliceLen = axisLen / sections;

    if (sliceIdx >= sections)
        return 0;
    if (localIndex >= sliceLen * unit)
        return 0;

    const uint32_t sliceStart = sliceIdx * sliceLen;

    // 构造 slice shape
    uint32_t outShape[kMaxTensorDims];
    for (uint32_t i = 0; i < srcdim; ++i) {
        outShape[i] = (i == axis) ? sliceLen : srcShape[i];
    }

    uint32_t outStrides[kMaxTensorDims];
    ComputeStrides(outShape, srcdim, outStrides);

    uint32_t coords[kMaxTensorDims];
    uint32_t rem = localIndex;
    for (uint32_t i = 0; i < srcdim; ++i) {
        coords[i] = rem / outStrides[i];
        rem = rem % outStrides[i];
    }
    coords[axis] += sliceStart;

    uint32_t srcStrides[kMaxTensorDims];
    ComputeStrides(srcShape, srcdim, srcStrides);

    uint32_t globalIndex = 0;
    for (uint32_t i = 0; i < srcdim; ++i) {
        globalIndex += coords[i] * srcStrides[i];
    }

    return globalIndex;
}

//---------------------------------------
// 按索引数组切分反算全局索引
//---------------------------------------
// srcShape: 原 tensor shape
// srcdim: 维度数
// axis_in: 切分轴
// indices: 切分点数组，长度为 indices_len
// indices_len: 切分点个数
// localIndex: 当前 slice 内局部索引
// sliceIdx: slice 序号（0~indices_len）
// unit: axis 外元素乘积
__aicore__ inline uint32_t CalIndexByIndices(
    const uint32_t srcShape[], uint32_t srcdim, int64_t axis_in, const uint32_t indices[], uint32_t indices_len,
    uint32_t localIndex, uint32_t sliceIdx, uint32_t unit)
{
    if (srcdim == 0)
        return 0;
    if (sliceIdx > indices_len)
        return 0;
    if (unit == 0)
        return 0;
    if (axis_in < 0)
        return localIndex;

    uint32_t axis = 0;
    if (!NormalizeAxis(srcdim, axis_in, axis))
        return 0;

    uint32_t axisLen = srcShape[axis];
    if (axisLen == 0)
        return 0;

    // 计算 slice 长度
    uint32_t start = (sliceIdx == 0) ? 0 : indices[sliceIdx - 1];
    uint32_t end = (sliceIdx == indices_len) ? axisLen : indices[sliceIdx];
    uint32_t sliceLen = end - start;
    if (localIndex >= sliceLen * unit)
        return 0;

    // 构造 slice shape
    uint32_t outShape[kMaxTensorDims];
    for (uint32_t i = 0; i < srcdim; ++i) {
        outShape[i] = (i == axis) ? sliceLen : srcShape[i];
    }

    uint32_t outStrides[kMaxTensorDims];
    ComputeStrides(outShape, srcdim, outStrides);

    uint32_t coords[kMaxTensorDims];
    uint32_t rem = localIndex;
    for (uint32_t i = 0; i < srcdim; ++i) {
        coords[i] = rem / outStrides[i];
        rem = rem % outStrides[i];
    }
    coords[axis] += start;

    uint32_t srcStrides[kMaxTensorDims];
    ComputeStrides(srcShape, srcdim, srcStrides);

    uint32_t globalIndex = 0;
    for (uint32_t i = 0; i < srcdim; ++i) {
        globalIndex += coords[i] * srcStrides[i];
    }

    return globalIndex;
}

// 函数：从tensorlist地址获取第index个tensor的实际地址
// 参数：
// - index：要获取的tensor在tensorlist中的索引
// - tensorPtr：tensorlist本身的地址
template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

template <typename T>
__aicore__ inline void CopyOutRange(
    AscendC::LocalTensor<T> src, uint32_t srcOffset, AscendC::GlobalTensor<T> dst, uint32_t dstOffset, uint32_t count,
    const uint32_t blockSize)
{
    if (count == 0)
        return;
    const uint64_t totalByte = static_cast<uint64_t>(count) * sizeof(T);
    uint64_t st = reinterpret_cast<uint64_t>(src[srcOffset].GetPhyAddr());
    uint64_t en = st + totalByte;
    uint32_t pre = (blockSize - st % blockSize) % blockSize;
    uint32_t aft = en % blockSize;
    uint32_t prenum = pre / sizeof(T);
    uint32_t midnum = (totalByte - pre - aft) / sizeof(T);

    if (prenum != 0) {
        for (uint32_t i = 0; i < prenum; ++i) {
            dst.SetValue(dstOffset + i, src.GetValue(srcOffset + i));
        }
    }
    if (midnum != 0) {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(midnum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst[dstOffset + prenum], src[srcOffset + prenum], copyParams);
    }
    if (aft != 0) {
        uint32_t aftNum = aft / sizeof(T);
        for (uint32_t i = 0; i < aftNum; ++i) {
            dst.SetValue(dstOffset + prenum + midnum + i, src.GetValue(srcOffset + prenum + midnum + i));
        }
    }
}
#endif // CALINDEX_UNIFIED_H
