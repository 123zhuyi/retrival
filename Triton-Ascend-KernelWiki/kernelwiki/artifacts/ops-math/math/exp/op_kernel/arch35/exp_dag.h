/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file exp_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_EXP_DAG_H
#define CANN_CUSTOM_OPS_EXP_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

#ifdef __CCE_AICORE__
constexpr static AscendC::Reg::CastTrait castTrait0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};
#endif

namespace ExpDag {
using namespace Ops::Base;
const int PLACEHOLDER_INDEX_0 = 0;
const int PLACEHOLDER_INDEX_1 = 1;
const int PLACEHOLDER_INDEX_2 = 2;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

template <class T>
struct ExpCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline ExpCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregInput;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vregOutput;
        AscendC::Reg::MaskReg mask;

        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                    Reg::Exp(vregOutput, vregInput, mask);
                    // OpCopyOut
                    Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                    vregOutput, mask);
                }
            }
        } else {
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput16;
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput16;
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                     (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);

                    Reg::Exp(vregOutput, vregInput, mask);

                    Reg::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
                    // OpCopyOut
                    Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                    vregOutput16, mask);
                }
            }
        }
#endif
    }
};

template <typename U, typename T = float>
struct ExpScaleOneShiftZeroLnbaseOne {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpExpResult = Bind<ExpCustom<U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpExpResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct ExpScaleNotOneShiftNotZeroLnbaseNotOne {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;

    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn0Cast1 = Bind<Vec::Muls<T>, OpCopyIn0Cast, Placeholder::Var<T, PLACEHOLDER_INDEX_0>>;
    using OpCopyIn0Cast2 = Bind<Vec::Adds<T>, OpCopyIn0Cast1, Placeholder::Var<T, PLACEHOLDER_INDEX_1>>;
    using OpCopyIn0Cast3 = Bind<Vec::Muls<T>, OpCopyIn0Cast2, Placeholder::Var<T, PLACEHOLDER_INDEX_2>>;
    using OpExpResult = Bind<ExpCustom<T>, OpCopyIn0Cast3>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpExpResult>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace ExpDag
#endif // CANN_CUSTOM_OPS_EXP_DAG_H
