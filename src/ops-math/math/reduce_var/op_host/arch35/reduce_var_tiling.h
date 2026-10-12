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
 * \file reduce_var_tiling.h
 * \brief
 */
#ifndef REDUCE_VAR_TILING_H
#define REDUCE_VAR_TILING_H
#include <vector>
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "atvoss/reduce/reduce_util.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "math/reduce_var/op_kernel/arch35/reduce_var_struct.h"

namespace optiling {

struct ReduceVarCompileInfo {
    Ops::Base::ReduceOpCompileInfo opInfo;
};

class ReduceVarTiling {
public:
    explicit ReduceVarTiling(gert::TilingContext* context, const ReduceVarCompileInfo* varCompileInfo,
                             ReduceVarTilingData* varTilingData)
        : context_(context)
    {
        reduceVarTilingData_ = varTilingData;
        reduceVarComileInfo_ = varCompileInfo;
    };

    ge::graphStatus RunTiling(Ops::Base::ReduceTilingKey& key); // ReduceOpInputParam& opInput, ReduceTilingKey& key);

private:
    void ComputeInnerUbRCnt(const uint64_t* shape);
    void CalcUserBasicBlock(bool patternA);
    void CalcUserWorkSpace();
    ge::graphStatus PrepareCompileInfo();
    ge::graphStatus ReduceVarGetInputParams(Ops::Base::ReduceOpInputParam& inputParam);
    void ReduceVarCalcInput(const Ops::Base::ReduceOpInputParam& inputParam);
    void SetReduceCntEachGroupR();
    void SetReduceVarTilingData();
    ge::graphStatus PreProcessOptionalParam();
    void EliminateOne(const std::vector<int64_t>& oriShape, std::vector<int64_t>& axes, uint64_t* shape,
                      int32_t& shapeSize);
    void MergeAxis(std::vector<int64_t>& axes, uint64_t* shape, int32_t& shapeSize);
    void TransformShape(const std::vector<int64_t>& oriShape, std::vector<int64_t>& axes, uint64_t* shape,
                        int32_t& shapeSize);
    ge::graphStatus DoTilingMatchPattern(uint64_t* shape, int32_t shapeSize);
    void GetTilingKey(Ops::Base::ReduceTilingKey& key);
    void PrintTilingData();
    void DoReduceTiling(Ops::Base::ReduceTilingKey& key);
    template <class Pattern>
    ge::graphStatus ComputeTiling(uint64_t* shape);
    template <class Pattern>
    ge::graphStatus CalcBasicBlock();
    template <class Pattern>
    ge::graphStatus ComputeEmptyTiling(uint64_t* shape);
    template <class Pattern>
    bool IsEmptyTensor(const uint64_t* shape);
    template <class Pattern>
    void ComputeCacheLineBlockAndUnit(const uint64_t* shape);
    template <class Pattern>
    void ComputeUnitA(const uint64_t* shape);
    template <class Pattern>
    void ComputeUnitR(const uint64_t* shape);
    template <class Pattern>
    void ComputeProgressUnitA(const uint64_t* shape);
    template <class Pattern>
    void SetTilingData(const uint64_t* shape);
    template <class Pattern>
    void SetTilingKey();
    template <class Pattern>
    bool IsAxisA(int32_t idx);
    template <class Pattern>
    void ComputeCacheLineBlock(const uint64_t* shape);
    template <class Pattern>
    void InitUnit(const uint64_t* shape);
    template <class Pattern>
    int32_t IsUseNddma(const uint64_t* shape);
    template <class Pattern>
    int32_t IsInvert(const uint64_t* shape);
    template <class Pattern>
    void ComputeStride(const uint64_t* shape);
    template <class Pattern>
    void PadDimOne(uint64_t* shape);
    uint64_t Ratio();
    ge::graphStatus ParamCheck(Ops::Base::ReduceOpInputParam& opInput);
    ge::graphStatus AxesCheck(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes);
    template <class Pattern>
    uint64_t CaculateReduceSize(const uint64_t* shape);
    void MakeWrapDim(const std::vector<int64_t>& shape, std::vector<int64_t>& axes);
    void AssembleUnit(Ops::Base::ReduceTilingUnit& unit, int32_t idx, uint64_t inner, uint64_t outer, uint64_t step);

    ge::graphStatus DoTiling(Ops::Base::ReduceOpInputParam& opInput, Ops::Base::ReduceTilingKey& key);

private:
    ReduceVarTilingData* reduceVarTilingData_ = nullptr;
    const ReduceVarCompileInfo* reduceVarComileInfo_ = nullptr;
    Ops::Base::ReduceOpCompileInfo compileInfo_;
    gert::TilingContext* context_ = nullptr;
    uint64_t basicBlock_ = 0;  // 算子搬入的buffer大小
    uint64_t resultBlock_ = 0; // reduce计算后的buffer大小
    uint64_t maxInputBytes_ = 0;
    int32_t dimNum_ = 0;
    size_t workSpaceSize_ = 0;
    Ops::Base::CacheLineBlock cBlock_;
    Ops::Base::ReduceTilingUnit unitA_;
    Ops::Base::ReduceTilingUnit unitR_;
    Ops::Base::ReduceTilingKey tilingKey_;
    Ops::Base::ReduceOpInputParam opInput_;

    int64_t correctionInvalid_ = 0;
    int64_t correction_ = 0;
    int64_t isMeanOut_ = 1;
    int64_t totalReduceSize_ = 1;
    uint64_t innerUbRCnt_ = 1; // ub切分R轴右侧的r的大小，不包含切分轴
    uint8_t isInvert_ = 0;     // cached during SetTilingData, written to tiling data
    double varFactor_ = 1.0;
    double meanFactor_ = 1.0;
};

} // namespace optiling
#endif // REDUCE_VAR_TILING_H
