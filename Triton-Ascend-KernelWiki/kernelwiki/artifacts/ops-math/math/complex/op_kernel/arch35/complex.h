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
 * \file complex.h
 * \brief complex broadcast kernel entry for arch35 (ascend950)
 */

#ifndef COMPLEX_ARCH35_H_
#define COMPLEX_ARCH35_H_

#include "kernel_operator.h"
#include "complex_dag.h"
#include "atvoss/broadcast/broadcast_sch.h"
#include "complex_struct.h"

using namespace AscendC;
using namespace ComplexOp;

//   Complex_float32 : DTYPE_REAL=float, DTYPE_OUT=complex64
//   Complex_float16 : DTYPE_REAL=half,  DTYPE_OUT=complex32
#ifndef DTYPE_REAL
#define DTYPE_REAL float
#endif

template <uint64_t schMode>
__global__ __aicore__ void complex(GM_ADDR real, GM_ADDR imag, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_REAL, float>::value) {
        using OpDag = ComplexBrcDag<complex64, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(real, imag, out);
    } else {
        using OpDag = ComplexBrcDag<complex32, half>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(real, imag, out);
    }
}

#endif // COMPLEX_ARCH35_H_
