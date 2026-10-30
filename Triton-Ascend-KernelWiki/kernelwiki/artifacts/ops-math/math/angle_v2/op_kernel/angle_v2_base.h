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
 * \file angle_v2_base.h
 * \brief
 */
#ifndef _ANGLE_V2_BASE_H_
#define _ANGLE_V2_BASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AngleV2N {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t COEFFICENT = 2;
constexpr uint8_t UINT8_MAX_VALUE = 255;
const double PI = 3.14159265358979323846;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#ifndef NAN
#define NAN (__builtin_nanf(""))
#endif

struct ConstData {
    const float const_pi = PI;
    const float const_pi_by_two = const_pi / 2;
    const float const_pi_by_four = const_pi / 4;
    const float const_pi_by_three_quarters = const_pi * 3 / 4;

    const float const_neg_pi = -const_pi;
    const float const_neg_pi_by_two = -const_pi_by_two;
    const float const_neg_pi_by_four = -const_pi_by_four;
    const float const_neg_pi_by_three_quarters = -const_pi_by_three_quarters;
};

template <typename yType>
class AngleV2Base
{
public:
    __aicore__ inline AngleV2Base()
    {}
    __aicore__ inline void BaseMemberDataInit(const AngleV2TilingData* __restrict tilingData)
    {
        totalLength = tilingData->totalLength;
        alignNum = tilingData->alignNum;
        formerNum = tilingData->formerNum;
        formerLength = tilingData->formerLength;
        tailNum = tilingData->tailNum;
        tailLength = tilingData->tailLength;
        totalLengthAligned = tilingData->totalLengthAligned;
        tileLength = tilingData->tileLength;
        mask = static_cast<uint64_t>(tilingData->dataPerRepeat);

        if (GetBlockIdx() < formerNum) {
            blockLength = formerLength;
            offset = blockLength * GetBlockIdx();
        } else {
            blockLength = tailLength;
            offset = formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum);
        }
        CalTileLength();
        tileNum = (blockLength - 1) / tileLength;
        lastTileLength = blockLength - tileNum * tileLength;
    }

    __aicore__ inline void CalTileLength()
    {
#if (__CCE_AICORE__ >= 200)
        // calculate tileLength for 910B
        if (blockLength <= tileLength) {
            tileLength = (blockLength + mask - 1) / mask * mask;
        }
#else
        tileLength = mask;
#endif
    }

    template <typename T>
    __aicore__ inline void DoSelect(
        LocalTensor<T>& dstLocal, LocalTensor<uint8_t>& selMask, LocalTensor<T>& src0Local, LocalTensor<T>& src1Local,
        uint64_t mask, uint8_t repeatTimes)
    {
#if (__CCE_AICORE__ >= 200)
        // Select for 910B mode 2
        Select(
            dstLocal, selMask, src0Local, src1Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeatTimes, repeatParams);
#else
        // Select for 910 mode 0
        Select(dstLocal, selMask, src0Local, src1Local, SELMODE::VSEL_CMPMASK_SPR, mask, repeatTimes, repeatParams);
#endif
    }

protected:
    int64_t totalLength;
    int64_t alignNum;
    int64_t totalLengthAligned;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t blockLength;
    int64_t offset;
    int64_t tileNum;
    int64_t lastTileLength;
    int64_t tileLength = 64;
    uint64_t mask = 64;
    BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
    UnaryRepeatParams CastDownParams = {1, 1, 4, 8};
    UnaryRepeatParams CastKeepParams = {1, 1, 8, 8};
    UnaryRepeatParams CastHighParams = {1, 1, 8, 4};
    uint16_t dupDstBlockStride = 1;
    uint8_t dupDstRepeatStride = 8;
};
} // namespace AngleV2N
#endif // _ANGLE_V2_BASE_H_
