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
 * \file right_shift_dag.h
 * \brief right_shift dag
 */

#ifndef OPS_MATH_RIGHT_SHIFT_DAG_H
#define OPS_MATH_RIGHT_SHIFT_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace RightShiftOp {
using namespace Ops::Base;

template <typename T>
struct RightShiftCustom8 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline RightShiftCustom8(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t digitZero = 0;
        int8_t rightShifts = 7;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> zeroReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg calcMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(zeroReg, digitZero, scalarMaskReg);
            Reg::Duplicate(sizeReg, rightShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                if constexpr (std::is_same_v<T, int8_t>) {
                    Reg::Compare<T, CMPMODE::GE>(calcMask, yReg, zeroReg, mask);
                    Reg::Select(yReg, yReg, zeroReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<T>&)yReg, mask);
                } else if constexpr (std::is_same_v<T, uint8_t>) {
                    Reg::Compare<T, CMPMODE::LE>(calcMask, yReg, sizeReg, mask);
                    Reg::Select(yReg, yReg, sizeReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<int8_t>&)yReg, mask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct RightShiftCustom16 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline RightShiftCustom16(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2,
                                         uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t digitZero = 0;
        int8_t rightShifts = 15;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> zeroReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg calcMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(zeroReg, digitZero, scalarMaskReg);
            Reg::Duplicate(sizeReg, rightShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                if constexpr (std::is_same_v<T, int16_t>) {
                    Reg::Compare<T, CMPMODE::GE>(calcMask, yReg, zeroReg, mask);
                    Reg::Select(yReg, yReg, zeroReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<T>&)yReg, mask);
                } else if constexpr (std::is_same_v<T, uint16_t>) {
                    Reg::Compare<T, CMPMODE::LE>(calcMask, yReg, sizeReg, mask);
                    Reg::Select(yReg, yReg, sizeReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<int16_t>&)yReg, mask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct RightShiftCustom32 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline RightShiftCustom32(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2,
                                         uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t digitZero = 0;
        int8_t rightShifts = 31;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> zeroReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg calcMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(zeroReg, digitZero, scalarMaskReg);
            Reg::Duplicate(sizeReg, rightShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                if constexpr (std::is_same_v<T, int32_t>) {
                    Reg::Compare<T, CMPMODE::GE>(calcMask, yReg, zeroReg, mask);
                    Reg::Select(yReg, yReg, zeroReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<T>&)yReg, mask);
                } else if constexpr (std::is_same_v<T, uint32_t>) {
                    Reg::Compare<T, CMPMODE::LE>(calcMask, yReg, sizeReg, mask);
                    Reg::Select(yReg, yReg, sizeReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<int32_t>&)yReg, mask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct RightShiftCustom64 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline RightShiftCustom64(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2,
                                         uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        int8_t digitZero = 0;
        int8_t rightShifts = 63;
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> yReg;
        Reg::RegTensor<T> zReg;
        Reg::RegTensor<T> zeroReg;
        Reg::RegTensor<T> sizeReg;

        Reg::MaskReg mask;
        Reg::MaskReg calcMask;

        __VEC_SCOPE__
        {
            Reg::MaskReg scalarMaskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(zeroReg, digitZero, scalarMaskReg);
            Reg::Duplicate(sizeReg, rightShifts, scalarMaskReg);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                Reg::DataCopy(xReg, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(yReg, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                if constexpr (std::is_same_v<T, int64_t>) {
                    Reg::Compare<T, CMPMODE::GE>(calcMask, yReg, zeroReg, mask);
                    Reg::Select(yReg, yReg, zeroReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<T>&)yReg, mask);
                } else if constexpr (std::is_same_v<T, uint64_t>) {
                    Reg::Compare<T, CMPMODE::LE>(calcMask, yReg, sizeReg, mask);
                    Reg::Select(yReg, yReg, sizeReg, calcMask);
                    Reg::ShiftRight(zReg, xReg, (Reg::RegTensor<int64_t>&)yReg, mask);
                }
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), zReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct RightShiftDag8 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<RightShiftCustom8<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct RightShiftDag16 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<RightShiftCustom16<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct RightShiftDag32 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<RightShiftCustom32<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct RightShiftDag64 {
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpResult = Bind<RightShiftCustom64<T>, InputX, InputY>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace RightShiftOp

#endif // OPS_MATH_RIGHT_SHIFT_DAG_H
