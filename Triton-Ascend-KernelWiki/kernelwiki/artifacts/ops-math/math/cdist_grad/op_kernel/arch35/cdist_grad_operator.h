/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_grad_operator.h
 * \brief Custom Vec operators for cdist_grad — Compare+Select via Reg
 *
 * By encapsulating CompareScalar + Select inside a single Vec operator node,
 * we avoid the multi-consumer buffer conflicts that occur when Compare and Select
 * are separate DAG nodes.
 *
 * Operators:
 *   CdistGradSignOp       — sign(x): 1.0 / -1.0 / 0.0  (ElemwiseUnaryOP)
 *   CdistGradMaskGEOp     — a >= b ? 1.0 : 0.0          (ElemwiseBinaryOP)
 *   CdistGradMaskNEZeroOp — x != 0 ? 1.0 : 0.0          (ElemwiseUnaryOP)
 *   CdistGradSelectZeroOp — mask ? 0.0 : src             (ElemwiseBinaryOP)
 */

#ifndef CDIST_GRAD_OPERATOR_H
#define CDIST_GRAD_OPERATOR_H

#include "atvoss/util/vec.h"

namespace CdistGrad {
using namespace Ops::Base;

/**
 * \brief sign(x): 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0
 *
 * Uses CompareScalar GT/LT to detect positive/negative, Select to choose values.
 * Replaces arithmetic approximation: diff / (|diff| + eps)
 */
template <typename PromoteT>
struct CdistGradSignOp : public Vec::ElemwiseUnaryOP<PromoteT, PromoteT> {
    __aicore__ inline CdistGradSignOp(LocalTensor<PromoteT>& dst, LocalTensor<PromoteT>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(PromoteT);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        __VEC_SCOPE__
        {
            __ubuf__ PromoteT* srcAddr = (__ubuf__ PromoteT*)src.GetPhyAddr();
            __ubuf__ PromoteT* dstAddr = (__ubuf__ PromoteT*)dst.GetPhyAddr();

            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregSrc;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregDst;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregOne;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregNegOne;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregZero;
            AscendC::Reg::MaskReg opMask;
            AscendC::Reg::MaskReg posMask;
            AscendC::Reg::MaskReg negMask;

            AscendC::Reg::Duplicate(vregOne, static_cast<PromoteT>(1.0));
            AscendC::Reg::Duplicate(vregNegOne, static_cast<PromoteT>(-1.0));
            AscendC::Reg::Duplicate(vregZero, static_cast<PromoteT>(0.0));

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                opMask = AscendC::Reg::UpdateMask<PromoteT, AscendC::Reg::RegTraitNumOne>(count);

                AscendC::Reg::DataCopy(vregSrc, srcAddr + loopIdx * vlSize);

                // posMask = (x > 0)
                AscendC::Reg::CompareScalar<PromoteT, CMPMODE::GT>(posMask, vregSrc, static_cast<PromoteT>(0), opMask);
                // negMask = (x < 0)
                AscendC::Reg::CompareScalar<PromoteT, CMPMODE::LT>(negMask, vregSrc, static_cast<PromoteT>(0), opMask);

                // Select(dst, true_val, false_val, mask): mask=1 → true_val, mask=0 → false_val
                // Step 1: posMask=1 → 1.0, posMask=0 → 0.0
                AscendC::Reg::Select(vregDst, vregOne, vregZero, posMask);
                // Step 2: negMask=1 → -1.0, negMask=0 → keep step1 result
                AscendC::Reg::Select(vregDst, vregNegOne, vregDst, negMask);

                AscendC::Reg::DataCopy(dstAddr + loopIdx * vlSize, vregDst, opMask);
            }
        }
#endif
    }
};

/**
 * \brief a == b ? 1.0 : 0.0  (binary mask: exact equality)
 *
 * Computes diff = a - b, then CompareScalar EQ(diff, 0) → mask.
 * Select(1.0, 0.0, mask) gives 1.0 where a == b, 0.0 otherwise.
 * Matches PyTorch's p=inf backward: mask = 1 - min(1, ceil(| |diff| - dist |))
 */
template <typename PromoteT>
struct CdistGradMaskEQOp : public Vec::ElemwiseBinaryOP<PromoteT, PromoteT, PromoteT> {
    __aicore__ inline CdistGradMaskEQOp(LocalTensor<PromoteT>& dst, LocalTensor<PromoteT>& src1,
                                        LocalTensor<PromoteT>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(PromoteT);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        __VEC_SCOPE__
        {
            __ubuf__ PromoteT* src1Addr = (__ubuf__ PromoteT*)src1.GetPhyAddr();
            __ubuf__ PromoteT* src2Addr = (__ubuf__ PromoteT*)src2.GetPhyAddr();
            __ubuf__ PromoteT* dstAddr = (__ubuf__ PromoteT*)dst.GetPhyAddr();

            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregA;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregB;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregDiff;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregOne;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregZero;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregResult;
            AscendC::Reg::MaskReg opMask;
            AscendC::Reg::MaskReg eqMask;

            AscendC::Reg::Duplicate(vregOne, static_cast<PromoteT>(1.0));
            AscendC::Reg::Duplicate(vregZero, static_cast<PromoteT>(0.0));

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                opMask = AscendC::Reg::UpdateMask<PromoteT, AscendC::Reg::RegTraitNumOne>(count);

                AscendC::Reg::DataCopy(vregA, src1Addr + loopIdx * vlSize);
                AscendC::Reg::DataCopy(vregB, src2Addr + loopIdx * vlSize);

                // diff = a - b
                AscendC::Reg::Sub(vregDiff, vregA, vregB, opMask);

                // eqMask = (diff == 0)  i.e. a == b
                AscendC::Reg::CompareScalar<PromoteT, CMPMODE::EQ>(eqMask, vregDiff, static_cast<PromoteT>(0), opMask);

                // vregResult = 1.0 where eqMask=1(a==b), 0.0 where eqMask=0(a!=b)
                AscendC::Reg::Select(vregResult, vregOne, vregZero, eqMask);
                AscendC::Reg::DataCopy(dstAddr + loopIdx * vlSize, vregResult, opMask);
            }
        }
#endif
    }
};

/**
 * \brief x != 0 ? 1.0 : 0.0  (non-zero mask)
 *
 * CompareScalar EQ(x, 0) → eqMask, then Select(1.0, 0.0, eqMask).
 * Where x==0: eqMask=1 → Select picks 0.0 (true branch = first arg).
 * Where x!=0: eqMask=0 → Select picks 1.0 (false branch = second arg).
 * Replaces arithmetic approximation: x / (x + eps)
 */
template <typename PromoteT>
struct CdistGradMaskNEZeroOp : public Vec::ElemwiseUnaryOP<PromoteT, PromoteT> {
    __aicore__ inline CdistGradMaskNEZeroOp(LocalTensor<PromoteT>& dst, LocalTensor<PromoteT>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t elemSize = sizeof(PromoteT);
        uint32_t vecLen = AscendC::VECTOR_REG_WIDTH / elemSize;
        uint16_t loopCnt = CeilDivision(count, vecLen);
        uint32_t vlLen = vecLen;
        __VEC_SCOPE__
        {
            __ubuf__ PromoteT* inputAddr = (__ubuf__ PromoteT*)src.GetPhyAddr();
            __ubuf__ PromoteT* outputAddr = (__ubuf__ PromoteT*)dst.GetPhyAddr();

            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregIn;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregValOne;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregValZero;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregOut;
            AscendC::Reg::MaskReg cmpMask;
            AscendC::Reg::MaskReg zeroMask;

            AscendC::Reg::Duplicate(vregValOne, static_cast<PromoteT>(1.0));
            AscendC::Reg::Duplicate(vregValZero, static_cast<PromoteT>(0.0));

            for (uint16_t idx = 0; idx < loopCnt; idx++) {
                cmpMask = AscendC::Reg::UpdateMask<PromoteT, AscendC::Reg::RegTraitNumOne>(count);

                AscendC::Reg::DataCopy(vregIn, inputAddr + idx * vlLen);

                // zeroMask = (x == 0)
                AscendC::Reg::CompareScalar<PromoteT, CMPMODE::EQ>(zeroMask, vregIn, static_cast<PromoteT>(0), cmpMask);

                // vregOut = 0.0 where zeroMask=1(x==0), 1.0 where zeroMask=0(x!=0)
                AscendC::Reg::Select(vregOut, vregValZero, vregValOne, zeroMask);
                AscendC::Reg::DataCopy(outputAddr + idx * vlLen, vregOut, cmpMask);
            }
        }
#endif
    }
};

/**
 * \brief dst = (cdist == 0) ? 0.0 : src
 *
 * Per-element conditional select: where cdist==0, output 0.0; otherwise pass through src.
 * This replaces multiply-by-mask (formula * mask), allowing formula to contain inf/NaN
 * at cdist==0 without producing inf*0=NaN. Equivalent to PyTorch's ternary (cond ? 0 : formula).
 * This eliminates the need for eps on Power bases and denominators.
 */
template <typename PromoteT>
struct CdistGradSelectZeroOp : public Vec::ElemwiseBinaryOP<PromoteT, PromoteT, PromoteT> {
    __aicore__ inline CdistGradSelectZeroOp(LocalTensor<PromoteT>& dst, LocalTensor<PromoteT>& src,
                                            LocalTensor<PromoteT>& cond, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t elemSize = sizeof(PromoteT);
        uint32_t vecLen = AscendC::VECTOR_REG_WIDTH / elemSize;
        uint16_t loopCnt = CeilDivision(count, vecLen);
        uint32_t vlLen = vecLen;
        __VEC_SCOPE__
        {
            __ubuf__ PromoteT* srcAddr = (__ubuf__ PromoteT*)src.GetPhyAddr();
            __ubuf__ PromoteT* condAddr = (__ubuf__ PromoteT*)cond.GetPhyAddr();
            __ubuf__ PromoteT* dstAddr = (__ubuf__ PromoteT*)dst.GetPhyAddr();

            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregSrc;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregCond;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregZero;
            AscendC::Reg::RegTensor<PromoteT, AscendC::Reg::RegTraitNumOne> vregOut;
            AscendC::Reg::MaskReg opMask;
            AscendC::Reg::MaskReg zeroMask;

            AscendC::Reg::Duplicate(vregZero, static_cast<PromoteT>(0.0));

            for (uint16_t idx = 0; idx < loopCnt; idx++) {
                opMask = AscendC::Reg::UpdateMask<PromoteT, AscendC::Reg::RegTraitNumOne>(count);

                AscendC::Reg::DataCopy(vregSrc, srcAddr + idx * vlLen);
                AscendC::Reg::DataCopy(vregCond, condAddr + idx * vlLen);

                // zeroMask = (cond == 0)
                AscendC::Reg::CompareScalar<PromoteT, CMPMODE::EQ>(zeroMask, vregCond, static_cast<PromoteT>(0),
                                                                   opMask);

                // dst = 0.0 where zeroMask=1(cdist==0), src where zeroMask=0(cdist!=0)
                AscendC::Reg::Select(vregOut, vregZero, vregSrc, zeroMask);
                AscendC::Reg::DataCopy(dstAddr + idx * vlLen, vregOut, opMask);
            }
        }
#endif
    }
};

/**
 * \brief dst = base ^ expVal  (high-accuracy Power, DOUBLE_FLOAT_TECH).
 *
 * Calls the WITH-sharedTmpBuffer overload of AscendC::Power. DOUBLE_FLOAT_TECH needs 3 float
 * buffers (high/low/log); the no-tmpBuffer overload fetches them via PopStackBuffer, which
 * returns garbage inside the DAG (no stack pool). So tmpBuf is allocated by an outer DAG node
 * (Abs(base)) and passed in here, only reinterpret-cast to uint8_t.
 * Inputs: base (tensor) / tmpBuf (scratch, value unused) / expVal (scalar exponent).
 */
template <typename PromoteT>
struct CdistGradPowDftOp : public Vec::ElemwiseTernaryOP<PromoteT, PromoteT, PromoteT, PromoteT> {
    __aicore__ inline CdistGradPowDftOp(LocalTensor<PromoteT>& dst, LocalTensor<PromoteT>& base,
                                        LocalTensor<PromoteT>& tmpBuf, PromoteT expVal, uint32_t count)
    {
#ifdef __CCE_AICORE__
        LocalTensor<uint8_t> sharedTmp = tmpBuf.template ReinterpretCast<uint8_t>();
        static constexpr AscendC::PowerConfig dftCfg = {AscendC::PowerAlgo::DOUBLE_FLOAT_TECH};
        AscendC::Power<PromoteT, false, dftCfg>(dst, base, expVal, sharedTmp, count);
#endif
    }
};

} // namespace CdistGrad

#endif // CDIST_GRAD_OPERATOR_H
