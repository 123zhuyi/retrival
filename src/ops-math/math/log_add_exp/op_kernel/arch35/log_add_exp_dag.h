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
 * \file log_add_exp_dag.h
 * \brief log_add_exp dag
 *
 * Simplified (base=-1, scale=1.0, shift=0.0):
 *   y = max(x1, x2) + ln(1 + exp(-|x1 - x2|))
 *
 * Full (any non-default):
 *   base=-1: y = max(x1, x2) + ln(1 + exp((-|x1 - x2|) * scale + shift))
 *   base>0:  y = max(x1, x2) + ln(1 + exp(((-|x1-x2|)*scale+shift)*ln(base))) / ln(base)
 */

#ifndef LOG_ADD_EXP_DAG_H
#define LOG_ADD_EXP_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
using namespace AscendC;

namespace LogAddExpOp {
#ifdef __CCE_AICORE__
constexpr static MicroAPI::CastTrait CAST_B16_TO_B32_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static MicroAPI::CastTrait CAST_B32_TO_B16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
#endif

// Keep the complete arithmetic chain in registers.  Besides removing UB round trips between
// individual DAG nodes, comparing the inputs directly handles both equal infinities with one
// compare/select pair.  A NaN never compares equal, so NaN propagation is unchanged.
template <typename StorageT, typename ComputeT = StorageT>
struct LogAddExpSimplifiedCustom : public Vec::ElemwiseBinaryOP<StorageT, StorageT, StorageT> {
    __aicore__ inline LogAddExpSimplifiedCustom(LocalTensor<StorageT>& dst, LocalTensor<StorageT>& src1,
                                                LocalTensor<StorageT>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        const uint32_t vectorLength = VECTOR_REG_WIDTH / sizeof(ComputeT);
        const uint16_t loopNum = CeilDivision(count, vectorLength);
        __ubuf__ StorageT* src1Addr = (__ubuf__ StorageT*)src1.GetPhyAddr();
        __ubuf__ StorageT* src2Addr = (__ubuf__ StorageT*)src2.GetPhyAddr();
        __ubuf__ StorageT* dstAddr = (__ubuf__ StorageT*)dst.GetPhyAddr();

        MicroAPI::RegTensor<StorageT, MicroAPI::RegTraitNumOne> input1StorageReg;
        MicroAPI::RegTensor<StorageT, MicroAPI::RegTraitNumOne> input2StorageReg;
        MicroAPI::RegTensor<StorageT, MicroAPI::RegTraitNumOne> outputStorageReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> input1Reg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> input2Reg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> maxReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> workReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> resultReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> addOneReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> correctionReg;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg compareMask;
        const ComputeT negOne = static_cast<ComputeT>(-1.0f);
        const ComputeT one = static_cast<ComputeT>(1.0f);
        const ComputeT posInf = static_cast<ComputeT>(__builtin_inff());

        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
                mask = MicroAPI::UpdateMask<ComputeT, MicroAPI::RegTraitNumOne>(count);
                MicroAPI::Duplicate(resultReg, static_cast<ComputeT>(0.0f), mask);
                if constexpr (std::is_same_v<StorageT, ComputeT>) {
                    MicroAPI::DataCopy(input1Reg, src1Addr + loopIdx * vectorLength);
                    MicroAPI::DataCopy(input2Reg, src2Addr + loopIdx * vectorLength);
                } else {
                    MicroAPI::DataCopy<StorageT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        input1StorageReg, src1Addr + loopIdx * vectorLength);
                    MicroAPI::DataCopy<StorageT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        input2StorageReg, src2Addr + loopIdx * vectorLength);
                    MicroAPI::Cast<ComputeT, StorageT, CAST_B16_TO_B32_TRAIT>(input1Reg, input1StorageReg, mask);
                    MicroAPI::Cast<ComputeT, StorageT, CAST_B16_TO_B32_TRAIT>(input2Reg, input2StorageReg, mask);
                }

                MicroAPI::Max(maxReg, input1Reg, input2Reg, mask);
                MicroAPI::Sub(workReg, input1Reg, input2Reg, mask);
                MicroAPI::Compare<ComputeT, CMPMODE::EQ>(compareMask, input1Reg, input2Reg, mask);
                MicroAPI::Select(workReg, resultReg, workReg, compareMask);
                MicroAPI::Abs(workReg, workReg, mask);
                MicroAPI::Muls(workReg, workReg, negOne, mask);
                MicroAPI::Exp(workReg, workReg, mask);

                // Stable log1p(exp(x)): preserve tiny exp values rounded away by 1 + exp(x),
                // and repair the inf / inf correction path when exp(x) overflows.
                MicroAPI::Adds(addOneReg, workReg, one, mask);
                MicroAPI::Adds(correctionReg, addOneReg, negOne, mask);
                MicroAPI::Div(correctionReg, workReg, correctionReg, mask);
                MicroAPI::Log(resultReg, addOneReg, mask);
                MicroAPI::Mul(resultReg, resultReg, correctionReg, mask);
                MicroAPI::CompareScalar<ComputeT, CMPMODE::NE>(compareMask, addOneReg, one, mask);
                MicroAPI::Select(resultReg, resultReg, workReg, compareMask);
                MicroAPI::CompareScalar<ComputeT, CMPMODE::NE>(compareMask, addOneReg, posInf, mask);
                MicroAPI::Duplicate(correctionReg, posInf, mask);
                MicroAPI::Select(resultReg, resultReg, correctionReg, compareMask);
                MicroAPI::Add(workReg, maxReg, resultReg, mask);

                if constexpr (std::is_same_v<StorageT, ComputeT>) {
                    MicroAPI::DataCopy(dstAddr + loopIdx * vectorLength, workReg, mask);
                } else {
                    MicroAPI::Cast<StorageT, ComputeT, CAST_B32_TO_B16_TRAIT>(outputStorageReg, workReg, mask);
                    MicroAPI::DataCopy<StorageT, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + loopIdx * vectorLength,
                                                                                     outputStorageReg, mask);
                }
            }
        }
#endif
    }
};

template <typename StorageT, typename ComputeT = StorageT>
struct LogAddExpFullCustom : public Vec::Elemwise6OP<StorageT, StorageT, StorageT, float, float, float, float> {
    __aicore__ inline LogAddExpFullCustom(LocalTensor<StorageT>& dst, LocalTensor<StorageT>& src1,
                                          LocalTensor<StorageT>& src2, float negScale, float shift, float lnBase,
                                          float invLnBase, uint32_t count)
    {
#ifdef __CCE_AICORE__
        const uint32_t vectorLength = VECTOR_REG_WIDTH / sizeof(ComputeT);
        const uint16_t loopNum = CeilDivision(count, vectorLength);
        __ubuf__ StorageT* src1Addr = (__ubuf__ StorageT*)src1.GetPhyAddr();
        __ubuf__ StorageT* src2Addr = (__ubuf__ StorageT*)src2.GetPhyAddr();
        __ubuf__ StorageT* dstAddr = (__ubuf__ StorageT*)dst.GetPhyAddr();

        MicroAPI::RegTensor<StorageT, MicroAPI::RegTraitNumOne> input1StorageReg;
        MicroAPI::RegTensor<StorageT, MicroAPI::RegTraitNumOne> input2StorageReg;
        MicroAPI::RegTensor<StorageT, MicroAPI::RegTraitNumOne> outputStorageReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> input1Reg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> input2Reg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> maxReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> workReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> resultReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> addOneReg;
        MicroAPI::RegTensor<ComputeT, MicroAPI::RegTraitNumOne> correctionReg;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg compareMask;
        const ComputeT negScaleValue = static_cast<ComputeT>(negScale);
        const ComputeT shiftValue = static_cast<ComputeT>(shift);
        const ComputeT lnBaseValue = static_cast<ComputeT>(lnBase);
        const ComputeT invLnBaseValue = static_cast<ComputeT>(invLnBase);
        const ComputeT negOne = static_cast<ComputeT>(-1.0f);
        const ComputeT one = static_cast<ComputeT>(1.0f);
        const ComputeT posInf = static_cast<ComputeT>(__builtin_inff());

        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
                mask = MicroAPI::UpdateMask<ComputeT, MicroAPI::RegTraitNumOne>(count);
                MicroAPI::Duplicate(resultReg, static_cast<ComputeT>(0.0f), mask);
                if constexpr (std::is_same_v<StorageT, ComputeT>) {
                    MicroAPI::DataCopy(input1Reg, src1Addr + loopIdx * vectorLength);
                    MicroAPI::DataCopy(input2Reg, src2Addr + loopIdx * vectorLength);
                } else {
                    MicroAPI::DataCopy<StorageT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        input1StorageReg, src1Addr + loopIdx * vectorLength);
                    MicroAPI::DataCopy<StorageT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        input2StorageReg, src2Addr + loopIdx * vectorLength);
                    MicroAPI::Cast<ComputeT, StorageT, CAST_B16_TO_B32_TRAIT>(input1Reg, input1StorageReg, mask);
                    MicroAPI::Cast<ComputeT, StorageT, CAST_B16_TO_B32_TRAIT>(input2Reg, input2StorageReg, mask);
                }

                MicroAPI::Max(maxReg, input1Reg, input2Reg, mask);
                MicroAPI::Sub(workReg, input1Reg, input2Reg, mask);
                MicroAPI::Compare<ComputeT, CMPMODE::EQ>(compareMask, input1Reg, input2Reg, mask);
                MicroAPI::Select(workReg, resultReg, workReg, compareMask);
                MicroAPI::Abs(workReg, workReg, mask);
                MicroAPI::Muls(workReg, workReg, negScaleValue, mask);
                MicroAPI::Adds(workReg, workReg, shiftValue, mask);
                MicroAPI::Muls(workReg, workReg, lnBaseValue, mask);
                MicroAPI::Exp(workReg, workReg, mask);

                // Stable log1p(exp(x)), matching the latest origin/master precision path.
                MicroAPI::Adds(addOneReg, workReg, one, mask);
                MicroAPI::Adds(correctionReg, addOneReg, negOne, mask);
                MicroAPI::Div(correctionReg, workReg, correctionReg, mask);
                MicroAPI::Log(resultReg, addOneReg, mask);
                MicroAPI::Mul(resultReg, resultReg, correctionReg, mask);
                MicroAPI::CompareScalar<ComputeT, CMPMODE::NE>(compareMask, addOneReg, one, mask);
                MicroAPI::Select(resultReg, resultReg, workReg, compareMask);
                MicroAPI::CompareScalar<ComputeT, CMPMODE::NE>(compareMask, addOneReg, posInf, mask);
                MicroAPI::Duplicate(correctionReg, posInf, mask);
                MicroAPI::Select(resultReg, resultReg, correctionReg, compareMask);
                MicroAPI::Muls(resultReg, resultReg, invLnBaseValue, mask);
                MicroAPI::Add(workReg, maxReg, resultReg, mask);

                if constexpr (std::is_same_v<StorageT, ComputeT>) {
                    MicroAPI::DataCopy(dstAddr + loopIdx * vectorLength, workReg, mask);
                } else {
                    MicroAPI::Cast<StorageT, ComputeT, CAST_B32_TO_B16_TRAIT>(outputStorageReg, workReg, mask);
                    MicroAPI::DataCopy<StorageT, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + loopIdx * vectorLength,
                                                                                     outputStorageReg, mask);
                }
            }
        }
#endif
    }
};

// ==================== Simplified (base=-1, scale=1.0, shift=0.0) ====================

template <typename T>
struct LogAddExpSimplifiedCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCompute = Bind<LogAddExpSimplifiedCustom<T>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCompute>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LogAddExpSimplifiedWithCastCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCompute = Bind<LogAddExpSimplifiedCustom<T, float>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCompute>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ==================== Full (with base, scale, shift) ====================
// Pos(=SetScalar 顺序): 0=negScale, 1=shift, 2=lnBase, 3=invLnBase

template <typename T>
struct LogAddExpFullCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCompute = Bind<LogAddExpFullCustom<T>, OpInputX1, OpInputX2, Placeholder::Var<float, 0>,
                           Placeholder::Var<float, 1>, Placeholder::Var<float, 2>, Placeholder::Var<float, 3>>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCompute>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LogAddExpFullWithCastCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCompute = Bind<LogAddExpFullCustom<T, float>, OpInputX1, OpInputX2, Placeholder::Var<float, 0>,
                           Placeholder::Var<float, 1>, Placeholder::Var<float, 2>, Placeholder::Var<float, 3>>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCompute>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace LogAddExpOp

#endif // LOG_ADD_EXP_DAG_H
