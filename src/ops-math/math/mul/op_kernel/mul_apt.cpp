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
 * \file mul_apt.cpp
 * \brief mul kernel
 */

#include "kernel_operator.h"
#include "arch35/mul_dag.h"
#include "arch35/mul_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void mul(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
        // int8
        using OpDag = MulDag::MulInt8Op::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
        // uint8
        using OpDag = MulDag::MulUint8Op::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, bool>::value) {
        // bool
        using OpDag = MulDag::MulBoolOp::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, double>::value) {
        // double
        using OpDag = MulDag::MulDoubleOp<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, complex32>::value) {
        // complex32
        using OpDag = MulDag::MulComplex32Op<complex32, complex64>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, DTYPE_X2>::value &&
                         (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value)) {
        // bfloat16, float16
        using OpDag = MulDag::MulXfp16Op<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, DTYPE_X2>::value) {
        // float, int32, int64, int16, complex64
        using OpDag = MulDag::MulOp<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (!std::is_same<DTYPE_X1, DTYPE_X2>::value) {
        // mix float、float16、bfloat16
        using OpDag = MulDag::MulMixFpOp<DTYPE_X1, DTYPE_X2, DTYPE_Y>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}
