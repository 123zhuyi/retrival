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
 * \file addr.cpp
 * \brief addr kernel
 */

#include "kernel_operator.h"
#include "arch35/addr_dag.h"
#include "arch35/addr_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace AddrOp;

template <uint64_t schMode>
__aicore__ inline void AddrWithoutAlphaProc(GM_ADDR x1, GM_ADDR y, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, int8_t>::value || std::is_same<DTYPE_X1, bool>::value) {
        using OpDag = AddrWithoutAlphaInt8::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, y);
    } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
        using OpDag = AddrWithoutAlphaUint8::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, y);
    } else if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value || std::is_same<DTYPE_X1, half>::value) {
        using OpDag = AddrWithoutAlphaBf16Fp16<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, y);
    } else {
        using OpDag = AddrWithoutAlphaCommon<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, y);
    }
}

template <uint64_t schMode>
__aicore__ inline void AddrWithoutBetaProc(GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, int8_t>::value || std::is_same<DTYPE_X1, bool>::value) {
        using OpDag = AddrWithoutBetaInt8::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x2, x3, y);
    } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
        using OpDag = AddrWithoutBetaUint8::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x2, x3, y);
    } else if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value || std::is_same<DTYPE_X1, half>::value) {
        using OpDag = AddrWithoutBetaBf16Fp16<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x2, x3, y);
    } else {
        using OpDag = AddrWithoutBetaCommon<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x2, x3, y);
    }
}

template <uint64_t schMode>
__aicore__ inline void AddrWithBetaWithAlphaProc(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, int8_t>::value || std::is_same<DTYPE_X1, bool>::value) {
        using OpDag = AddrWithBetaWithAlphaInt8::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, x3, y);
    } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
        using OpDag = AddrWithBetaWithAlphaUint8::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, x3, y);
    } else if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value || std::is_same<DTYPE_X1, half>::value) {
        using OpDag = AddrWithBetaWithAlphaBf16Fp16<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, x3, y);
    } else {
        using OpDag = AddrWithBetaWithAlphaCommon<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, x3, y);
    }
}

template <uint64_t schMode, uint64_t betaZeroOrNot, uint64_t alphaZeroOrNot>
__global__ __aicore__ void addr(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR beta, GM_ADDR alpha, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (static_cast<int8_t>(alphaZeroOrNot) == static_cast<int8_t>(0)) {
        AddrWithoutAlphaProc<schMode>(x1, y, tiling);
    } else if constexpr (static_cast<int8_t>(betaZeroOrNot) == static_cast<int8_t>(0)) {
        AddrWithoutBetaProc<schMode>(x2, x3, y, tiling);
    } else {
        AddrWithBetaWithAlphaProc<schMode>(x1, x2, x3, y, tiling);
    }
}
