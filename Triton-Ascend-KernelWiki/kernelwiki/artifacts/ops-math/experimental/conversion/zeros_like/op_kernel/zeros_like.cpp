/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file zeros_like.cpp
 * \brief ascend910b (DAV_2201) 标准 AscendC AI Core kernel：向输出 GM 写全 0。
 *        退化 Elementwise（仅写常量 0，无 CopyIn）：UB 单块零缓冲 Duplicate(0) 一次后循环复用，
 *        主体 32B 对齐用 DataCopy、尾部非对齐用 DataCopyPad（UB→GM，A2 支持）。
 *        按字节宽度桶（1/2/4/8）选择 Duplicate 视图（uint16/uint32），全 0 二进制等价。
 *
 * experimental 自包含算子：仅 ascend910b（DAV_2201）AI Core 配置，AICPU/950 兜底由
 * op_api 层根据芯片选择（experimental 仅落地 910b AI Core）。
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "zeros_like_tiling_data.h"
#include "zeros_like_tiling_key.h"

using namespace AscendC;
using namespace ZerosLikeNs; // 含共享常量 ZL_BLOCK_BYTES（见 zeros_like_tiling_data.h）

// 计算本核负责的字节区间 [coreOffsetBytes, coreOffsetBytes + coreBytes)。
// 返回 false 表示本核无实际数据（多分配核 / 越界对齐块 / 空核），调用方应直接 return。
__aicore__ inline bool ZerosLikeComputeCoreRange(
    const ZerosLikeTilingData& t, uint32_t blockIdx, uint64_t& coreOffsetBytes, uint64_t& coreBytes)
{
    if (blockIdx >= t.usedCoreNum) {
        return false; // 多分配核保护（usedCore=min(core, totalBlock) 已避免，但稳妥起见）
    }

    // 本核字节范围：前 tailCoreNum 个核各多 1 个 32B 块（以对齐块描述）
    coreBytes = t.perCoreBytes;
    if (blockIdx < t.tailCoreNum) {
        coreBytes += ZL_BLOCK_BYTES;
        coreOffsetBytes = coreBytes * blockIdx;
    } else {
        coreOffsetBytes = (t.perCoreBytes + ZL_BLOCK_BYTES) * t.tailCoreNum +
                          t.perCoreBytes * (static_cast<uint64_t>(blockIdx) - t.tailCoreNum);
    }

    // 闭合到 totalBytes：各核以「整 32B 块」均分，最后一个有效核的尾部可能不足 32B，
    // 通过 clamp 把超出 totalBytes 的对齐补齐字节裁掉，非对齐尾部由 DataCopyPad 写出。
    if (coreOffsetBytes >= t.totalBytes) {
        return false; // 该核无实际数据（小 shape 多分配的对齐块）
    }
    if (coreOffsetBytes + coreBytes > t.totalBytes) {
        coreBytes = t.totalBytes - coreOffsetBytes;
    }
    return coreBytes != 0; // coreBytes==0：0 元素 / 空核保护
}

// 将本核字节区间 [coreOffsetBytes, +coreBytes) 写全 0：主体 32B 对齐用 DataCopy，
// 尾部非对齐用 DataCopyPad；zUb 为已 Duplicate(0) 的单块零缓冲。
template <typename ViewT>
__aicore__ inline void ZerosLikeWriteRange(
    GlobalTensor<ViewT>& yGm, const LocalTensor<ViewT>& zUb, uint32_t tileBytes, uint64_t coreOffsetBytes,
    uint64_t coreBytes)
{
    uint64_t remainBytes = coreBytes;
    uint64_t curByteOff = coreOffsetBytes;
    while (remainBytes > 0) {
        uint32_t chunkBytes = (remainBytes >= tileBytes) ? tileBytes : static_cast<uint32_t>(remainBytes);
        uint64_t elemOff = curByteOff / sizeof(ViewT); // curByteOff 恒为 32B 对齐，整除 sizeof(ViewT)
        if (chunkBytes % ZL_BLOCK_BYTES == 0) {
            // 32B 对齐主体：DataCopy（count 单位 = ViewT 元素个数）
            DataCopy(yGm[elemOff], zUb, chunkBytes / sizeof(ViewT));
        } else {
            // 非对齐尾部：DataCopyPad（blockLen 单位 = 字节）
            DataCopyExtParams padParams;
            padParams.blockCount = 1;
            padParams.blockLen = chunkBytes;
            padParams.srcStride = 0;
            padParams.dstStride = 0;
            padParams.rsv = 0;
            DataCopyPad(yGm[elemOff], zUb, padParams);
        }
        curByteOff += chunkBytes;
        remainBytes -= chunkBytes;
    }
}

// 统一写出实现：ViewT 为 Duplicate / 搬运视图（uint16_t 或 uint32_t）。
// 所有切分以字节为单位；本核负责 [coreOffsetBytes, coreOffsetBytes + coreBytes)。
template <typename ViewT>
__aicore__ inline void ZerosLikeWriteImpl(GM_ADDR y, const ZerosLikeTilingData& t)
{
    uint64_t coreOffsetBytes = 0;
    uint64_t coreBytes = 0;
    if (!ZerosLikeComputeCoreRange(t, GetBlockIdx(), coreOffsetBytes, coreBytes)) {
        return;
    }

    // 单块全 0 UB 缓冲：Duplicate 一次后循环复用
    TPipe pipe;
    TBuf<TPosition::VECCALC> zeroBuf;
    uint32_t tileBytes = static_cast<uint32_t>(t.tileBytes);
    pipe.InitBuffer(zeroBuf, tileBytes);
    LocalTensor<ViewT> zUb = zeroBuf.Get<ViewT>();
    Duplicate<ViewT>(zUb, static_cast<ViewT>(0), static_cast<int32_t>(tileBytes / sizeof(ViewT)));

    // 同步：Duplicate(Vector) 写完 UB 后再由 MTE3 搬出
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

    // 以 ViewT 视图寻址：所有核字节偏移、主体 chunk 长度均为 32B 对齐
    // （= sizeof(ViewT) 的整数倍）。
    GlobalTensor<ViewT> yGm;
    yGm.SetGlobalBuffer((__gm__ ViewT*)y);

    ZerosLikeWriteRange<ViewT>(yGm, zUb, tileBytes, coreOffsetBytes, coreBytes);
}

// kernel 入口：模板编程（template<int BYTE_KEY> + if constexpr），禁止 TILING_KEY_IS 宏。
// 入参 x 存在但不读取（仅依赖 shape/dtype 推导的 tiling，向 y 写全 0）。
template <int BYTE_KEY>
__global__ __aicore__ void zeros_like(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ZerosLikeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ZerosLikeTilingData, tilingData, tiling);

    if constexpr (BYTE_KEY == ZL_KEY_1B) {
        // int8/uint8/bool：UB 按 uint16 整块写 0，按字节长度搬出（奇数字节由 DataCopyPad）
        ZerosLikeWriteImpl<uint16_t>(y, tilingData);
    } else if constexpr (BYTE_KEY == ZL_KEY_2B) {
        // fp16/bf16
        ZerosLikeWriteImpl<uint16_t>(y, tilingData);
    } else if constexpr (BYTE_KEY == ZL_KEY_4B) {
        // fp32/int32
        ZerosLikeWriteImpl<uint32_t>(y, tilingData);
    } else { // ZL_KEY_8B
        // int64：拆成 2×uint32 写 0，二进制等价
        ZerosLikeWriteImpl<uint32_t>(y, tilingData);
    }
}
