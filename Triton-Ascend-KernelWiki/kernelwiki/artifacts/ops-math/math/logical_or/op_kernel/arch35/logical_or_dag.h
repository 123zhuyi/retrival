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
 * \file logical_or_dag.h
 * \brief logical_or dag
 */

#ifndef LOGICAL_OR_DAG_H
#define LOGICAL_OR_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

using namespace Ops::Base;

namespace LogicalOrOp {
constexpr int LOGICAL_OR_CMP_NE_MODE = 5;
constexpr int LOGICAL_OR_SEL_MODE = 2;

// 浮点型自定义Op：除了DAG图中的搬入搬出，其余内容全部存放在此自定义Op中
// 流程：位Or(当作整形) -> Compare(NE 0) -> Select(1/0) -> 输出uint8 BOOL
// 只有Or操作时当作对应位宽的整形处理，其他操作在整形上下文
// T = half/bfloat16_t/float (浮点输入类型)
// IntT = uint16_t/uint16_t/uint32_t (对应位宽的整形，用于位OR)
template <typename T, typename IntT>
struct LogicalOrFloatCustom : public Vec::ElemwiseBinaryOP<uint8_t, T, T> {
    __aicore__ inline LogicalOrFloatCustom(LocalTensor<uint8_t>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2,
                                           const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);

        __local_mem__ T* src1Addr = (__local_mem__ T*)src1.GetPhyAddr();
        __local_mem__ T* src2Addr = (__local_mem__ T*)src2.GetPhyAddr();
        __local_mem__ uint8_t* dstAddr = (__local_mem__ uint8_t*)dst.GetPhyAddr();

        uint16_t loopTimes = (count + VL_T - 1) / VL_T;
        uint32_t sregMask = count;

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> src1Value;
            Reg::RegTensor<T> src2Value;
            Reg::RegTensor<T> orRes;
            Reg::RegTensor<uint8_t> dstReg;
            Reg::RegTensor<uint8_t> oneReg;
            Reg::RegTensor<uint8_t> zeroReg;
            Reg::MaskReg preg;
            Reg::MaskReg cmpMask;

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T>(sregMask);
                // 以浮点类型 T 加载输入
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src1Value, src1Addr + VL_T * j);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src2Value, src2Addr + VL_T * j);

                // 位OR操作：当作对应位宽的整形处理
                Reg::Or((Reg::RegTensor<IntT>&)orRes, (Reg::RegTensor<IntT>&)src1Value,
                        (Reg::RegTensor<IntT>&)src2Value, preg);

                // Compare: OR结果 != 0 (NE模式)
                Reg::Compares<T, CMPMODE::NE>(cmpMask, orRes, static_cast<T>(0), preg);

                // Select: 根据cmpMask选择1或0 (uint8_t)
                Reg::Duplicate(oneReg, static_cast<uint8_t>(1));
                Reg::Duplicate(zeroReg, static_cast<uint8_t>(0));
                Reg::Select<uint8_t>(dstReg, oneReg, zeroReg, cmpMask);

                // 存储uint8 BOOL结果
                if constexpr (sizeof(T) == sizeof(half)) {
                    Reg::DataCopy<uint8_t, Reg::StoreDist::DIST_PACK_B16>(dstAddr + VL_T * j, dstReg, preg);
                } else {
                    Reg::DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(dstAddr + VL_T * j, dstReg, preg);
                }
            }
        }
#endif
    }
};

template <typename T>
struct LogicalOrCompute {
    // 通过Compute构造计算图 - BOOL类型专用，直接Or后输出
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using LogicalOrRes = Bind<Vec::Or<T>, InputX1, InputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, LogicalOrRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 整形类型专用DAG：Or -> Compare(NE 0) -> Select(1/0) -> CopyOut(BOOL)
template <typename T>
struct LogicalOrIntegralCompute {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OrRes = Bind<Vec::Or<T>, InputX1, InputX2>;

    using ConstZero = MAKE_CONST(T, 0);
    using DupZero = Bind<Vec::Duplicate<T>, ConstZero>;
    using CmpRes = Bind<Vec::Compare<uint8_t, T, LOGICAL_OR_CMP_NE_MODE>, OrRes, DupZero>;

    using ConstOneU8 = MAKE_CONST(uint8_t, 1);
    using ConstZeroU8 = MAKE_CONST(uint8_t, 0);
    using DupOneU8 = Bind<Vec::Duplicate<uint8_t>, ConstOneU8>;
    using DupZeroU8 = Bind<Vec::Duplicate<uint8_t>, ConstZeroU8>;
    using SelectRes = Bind<Vec::Select<uint8_t, uint8_t, LOGICAL_OR_SEL_MODE>, CmpRes, DupOneU8, DupZeroU8>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, SelectRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 浮点型DAG（float16/bfloat16/float32 → bool）：
// CopyInBrc(T) -> LogicalOrFloatCustom(位Or+Compare+Select全在自定义Op内) -> CopyOut(uint8)
// 除了搬入搬出，其余内容全部存放在自定义Op中
template <typename T, typename IntT>
struct LogicalOrFloatCompute {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using CustomRes = Bind<LogicalOrFloatCustom<T, IntT>, InputX1, InputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, CustomRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LogicalOrOp
#endif // LOGICAL_OR_DAG_H
