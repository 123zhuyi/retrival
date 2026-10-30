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
 * \file nan_to_num_dag.h
 * \brief nan_to_num dag
 */

#ifndef OPS_MATH_NAN_TO_NUM_DAG_H
#define OPS_MATH_NAN_TO_NUM_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
constexpr static AscendC::Reg::CastTrait castTrait0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};
#endif

namespace NanToNumOp {
using namespace Ops::Base;
const int PLACEHOLDER_INDEX_0 = 0;
const int PLACEHOLDER_INDEX_1 = 1;
const int PLACEHOLDER_INDEX_2 = 2;

const uint32_t MAX_VALUE_FP32 = 0x7f800000;
const uint32_t MIN_VALUE_FP32 = 0xff800000;

template <class T>
struct NanToNumCustom : public Vec::ElemwiseQuaternaryOP<T, T, float, float, float> {
    __aicore__ inline NanToNumCustom(LocalTensor<T>& dst, LocalTensor<T>& src, float nan, float posinf, float neginf,
                                     uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        T maxValue = 0;
        T minValue = 0;
        maxValue = *reinterpret_cast<const float*>(&MAX_VALUE_FP32);
        minValue = *reinterpret_cast<const float*>(&MIN_VALUE_FP32);
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInput;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregOutput;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput16;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput16;
        Reg::RegTensor<float> nanTensor;
        Reg::RegTensor<float> posinfTensor;
        Reg::RegTensor<float> neginfTensor;
        Reg::MaskReg mask, cmpMaskNan, cmpMaskPosinf, cmpMaskNeginf;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                Reg::Duplicate(nanTensor, nan);
                Reg::Duplicate(posinfTensor, posinf);
                Reg::Duplicate(neginfTensor, neginf);

                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Compare<T, CMPMODE::NE>(cmpMaskNan, vregInput, vregInput, mask);
                    Reg::CompareScalar<T, CMPMODE::EQ>(cmpMaskPosinf, vregInput, maxValue, mask);
                    Reg::CompareScalar<T, CMPMODE::EQ>(cmpMaskNeginf, vregInput, minValue, mask);
                    Reg::Select<T>(vregOutput, nanTensor, vregInput, cmpMaskNan);
                    Reg::Select<T>(vregOutput, posinfTensor, vregOutput, cmpMaskPosinf);
                    Reg::Select<T>(vregOutput, neginfTensor, vregOutput, cmpMaskNeginf);
                    // OpCopyOut
                    Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                    vregOutput, mask);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                Reg::Duplicate(nanTensor, nan);
                Reg::Duplicate(posinfTensor, posinf);
                Reg::Duplicate(neginfTensor, neginf);

                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                     (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
                    Reg::Compare<float, CMPMODE::NE>(cmpMaskNan, vregInput, vregInput, mask);
                    Reg::CompareScalar<float, CMPMODE::EQ>(cmpMaskPosinf, vregInput, maxValue, mask);
                    Reg::CompareScalar<float, CMPMODE::EQ>(cmpMaskNeginf, vregInput, minValue, mask);
                    Reg::Select<float>(vregOutput, nanTensor, vregInput, cmpMaskNan);
                    Reg::Select<float>(vregOutput, posinfTensor, vregOutput, cmpMaskPosinf);
                    Reg::Select<float>(vregOutput, neginfTensor, vregOutput, cmpMaskNeginf);
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

template <typename U>
struct NanToNumDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpResult = Bind<NanToNumCustom<U>, OpCopyIn0, Placeholder::Var<float, PLACEHOLDER_INDEX_0>,
                          Placeholder::Var<float, PLACEHOLDER_INDEX_1>, Placeholder::Var<float, PLACEHOLDER_INDEX_2>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResult>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace NanToNumOp

#endif // OPS_MATH_NAN_TO_NUM_DAG_H
