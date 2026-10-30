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
#include "atvoss/reduce/reduce_tiling_data.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr int64_t MAX_CORE_COUNT = 128;

struct ReduceVarCompileInfo {
    Ops::Base::ReduceOpCompileInfo opInfo;
};

BEGIN_TILING_DATA_DEF(ReduceVarTilingDataStru)
TILING_DATA_FIELD_DEF(uint64_t, factorACntPerCore);
TILING_DATA_FIELD_DEF(uint64_t, factorATotalCnt);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorA);
TILING_DATA_FIELD_DEF(uint64_t, factorRCntPerCore);
TILING_DATA_FIELD_DEF(uint64_t, factorRTotalCnt);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorR);
TILING_DATA_FIELD_DEF(uint64_t, groupR);
TILING_DATA_FIELD_DEF(uint64_t, outSize);
TILING_DATA_FIELD_DEF(uint64_t, basicBlock);
TILING_DATA_FIELD_DEF(uint64_t, resultBlock);
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, useNddma);
TILING_DATA_FIELD_DEF(float, meanVar);
TILING_DATA_FIELD_DEF_ARR(uint64_t, Ops::Base::ReduceOpTmpl::MAX_DIM, shape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, Ops::Base::ReduceOpTmpl::MAX_DIM, stride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, Ops::Base::ReduceOpTmpl::MAX_DIM, dstStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceVarTilingDataStruOp, ReduceVarTilingDataStru);

BEGIN_TILING_DATA_DEF(ReduceVarTilingData)
TILING_DATA_FIELD_DEF_STRUCT(ReduceVarTilingDataStru, reduceOpTiling);
TILING_DATA_FIELD_DEF(int64_t, correction);
TILING_DATA_FIELD_DEF(int64_t, correctionInvalid);
TILING_DATA_FIELD_DEF(int64_t, isMeanOut);
TILING_DATA_FIELD_DEF(int64_t, workSpaceSize);
TILING_DATA_FIELD_DEF(int32_t, useNddma);
TILING_DATA_FIELD_DEF(float, varFactor); // 1/(R-correction)
TILING_DATA_FIELD_DEF(float, meanFactor); // 1/R
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, reduceCntEachGroupR); // 每个groupR处理的R的个数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceVar, ReduceVarTilingData);
REGISTER_TILING_DATA_CLASS(ReduceStdV2, ReduceVarTilingData);

class ReduceVarTiling {
public:
    explicit ReduceVarTiling(gert::TilingContext* context, const ReduceVarCompileInfo *varCompileInfo,
        ReduceVarTilingData *varTilingData, Ops::Base::ReduceOpTilingData* reduceTiling) :
            context_(context), tilingData_(reduceTiling)
    {
        reduceVarTilingData_ = varTilingData;
        reduceVarComileInfo_ = varCompileInfo;
    };

    ge::graphStatus RunTiling(Ops::Base::ReduceTilingKey &key); // ReduceOpInputParam& opInput, ReduceTilingKey& key);

private:
    void ComputeInnerUbRCnt(const uint64_t* shape);
    void CalcUserBasicBlock(bool patternA);
    void CalcUserWorkSpace();
    ge::graphStatus PrepareCompileInfo();
    ge::graphStatus ReduceVarGetInputParams(Ops::Base::ReduceOpInputParam &inputParam);
    void ReduceVarCalcInput(const Ops::Base::ReduceOpInputParam &inputParam);
    void ConvertReduceOpTilingData(ReduceVarTilingDataStru* dst, const Ops::Base::ReduceOpTilingData* src);
    void SetReduceCntEachGroupR();
    void SetReduceVarTilingData();
    void SetUseNddma();
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
    ReduceVarTilingData *reduceVarTilingData_ = nullptr;
    const ReduceVarCompileInfo *reduceVarComileInfo_ = nullptr;
    Ops::Base::ReduceOpCompileInfo compileInfo_;
    gert::TilingContext* context_ = nullptr;
    Ops::Base::ReduceOpTilingData* tilingData_ = nullptr;
    uint64_t basicBlock_ = 0;   // 算子搬入的buffer大小
    uint64_t resultBlock_ = 0;  // reduce计算后的buffer大小
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
    uint64_t innerUbRCnt_ = 1;  // ub切分R轴右侧的r的大小，不包含切分轴
    double varFactor_ = 1.0;
    double meanFactor_ = 1.0;
};

}  // namespace optiling
#endif  // REDUCE_VAR_TILING_H
