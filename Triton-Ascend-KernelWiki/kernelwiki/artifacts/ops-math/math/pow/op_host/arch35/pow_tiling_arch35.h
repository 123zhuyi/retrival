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
 * \file pow_tiling_arch35.h
 * \brief
 */

#ifndef POW_CANNDEV_OPS_BUILT_IN_OP_TILING_RUNTIME_POW_POW_TILING_H_
#define POW_CANNDEV_OPS_BUILT_IN_OP_TILING_RUNTIME_POW_POW_TILING_H_
#include <map>
#include <tuple>
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"

namespace optiling
{
const int INPUT_MAX_DIMS = 8;
using namespace Ops::Base;
BEGIN_TILING_DATA_DEF(PowBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pow, PowBaseTilingData)

BEGIN_TILING_DATA_DEF(PowTensorScalarTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubOuter);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, totalSize);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
END_TILING_DATA_DEF;

// 1001 - float16 ascendc_api tensor scalar
// 2001 - bfloat16 ascendc_api tensor scalar
// 3001 - float32 ascendc_api tensor scalar
// 4001 - uint8_t ascendc_api tensor scalar
// 5001 - int8_t ascendc_api tensor scalar
// 6001 - int16_t ascendc_api tensor scalar
// 7001 - int32_t ascendc_api tensor scalar
REGISTER_TILING_DATA_CLASS(Pow_1001, PowTensorScalarTilingData)
REGISTER_TILING_DATA_CLASS(Pow_2001, PowTensorScalarTilingData)
REGISTER_TILING_DATA_CLASS(Pow_3001, PowTensorScalarTilingData)
REGISTER_TILING_DATA_CLASS(Pow_4001, PowTensorScalarTilingData)
REGISTER_TILING_DATA_CLASS(Pow_5001, PowTensorScalarTilingData)
REGISTER_TILING_DATA_CLASS(Pow_6001, PowTensorScalarTilingData)
REGISTER_TILING_DATA_CLASS(Pow_7001, PowTensorScalarTilingData)

BEGIN_TILING_DATA_DEF(PowTensorTensorTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubOuter);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, shapeLen);
TILING_DATA_FIELD_DEF(int64_t, ubSplitAxis);
TILING_DATA_FIELD_DEF(int64_t, dimProductBeforeUbInner);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, INPUT_MAX_DIMS, input0Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, INPUT_MAX_DIMS, input1Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, INPUT_MAX_DIMS, outputDims);
TILING_DATA_FIELD_DEF_ARR(int64_t, INPUT_MAX_DIMS, input0Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, INPUT_MAX_DIMS, input1Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, INPUT_MAX_DIMS, outputStrides);
END_TILING_DATA_DEF;

// input_x is float16, input_y is float16, output_z is float16, max dims in ub is 5 and nddma does not need loops
// input_x is float16, input_y is float16, output_z is float16, max dims in ub is 8 and nddma needs loops
// input_x is bfloat16, input_y is bfloat16, output_z is bfloat16, max dims in ub is 5 and nddma does not need loops
// input_x is bfloat16, input_y is bfloat16, output_z is bfloat16, max dims in ub is 8 and nddma needs loops
// input_x is float32, input_y is float32, output_z is float32, max dims in ub is 5 and nddma does not need loops
// input_x is float32, input_y is float32, output_z is float32, max dims in ub is 8 and nddma needs loops
// input_x is uint8, input_y is uint8, output_z is uint8, max dims in ub is 5 and nddma does not need loops
// input_x is uint8, input_y is uint8, output_z is uint8, max dims in ub is 8 and nddma needs loops
// input_x is int8, input_y is int8, output_z is int8, max dims in ub is 5 and nddma does not need loops
// input_x is int8, input_y is int8, output_z is int8, max dims in ub is 8 and nddma needs loops
// input_x is int16, input_y is int16, output_z is int16, max dims in ub is 5 and nddma does not need loops
// input_x is int16, input_y is int16, output_z is int16, max dims in ub is 8 and nddma needs loops
// input_x is int32, input_y is int32, output_z is int32, max dims in ub is 5 and nddma does not need loops
// input_x is int32, input_y is int32, output_z is int32, max dims in ub is 8 and nddma needs loops
REGISTER_TILING_DATA_CLASS(Pow_100000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_100000001001100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_200000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_200000001001100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_300000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_300000001001100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_400000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_400000001001100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_500000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_500000001001100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_600000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_600000001001100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_700000001000100, PowTensorTensorTilingData)
REGISTER_TILING_DATA_CLASS(Pow_700000001001100, PowTensorTensorTilingData)

struct PowCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    bool isRegBase = false;
    int64_t vectorLength = 0;
    int64_t blockSize = 0;
};

struct InputParams {
    // platform input info
    int64_t coreNum;
    int64_t ubSize;
    bool isRegBase = false;
    int64_t vectorLength = 0;
    int64_t blockSize = 0;

    // input desc info
    gert::Shape baseShape;
    gert::Shape expShape;
    gert::Shape powShape;
    ge::DataType baseDtype;
    ge::DataType expDtype;
    ge::DataType powDtype;
    int64_t baseDtypeSize;
    int64_t computeDtypeSize;
};

class PowTilingBase : public Ops::Base::TilingBaseClass
{
public:
    explicit PowTilingBase(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {
    }

    ~PowTilingBase() override
    {
    }
    InputParams params_;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
};

class PowTensorScalarTiling : public PowTilingBase
{
public:
    explicit PowTensorScalarTiling(gert::TilingContext* context) : PowTilingBase(context)
    {
    }
    ~PowTensorScalarTiling() override
    {
    }
    PowTensorScalarTilingData tilingData_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    int64_t GetUbFactor(int64_t dim0, int64_t oriUbFactor) const;
};

class PowTensorTensorTiling : public PowTilingBase
{
public:
    explicit PowTensorTensorTiling(gert::TilingContext* context) : PowTilingBase(context)
    {
    }
    ~PowTensorTensorTiling() override
    {
    }
    PowTensorTensorTilingData tilingData_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    void SetOpKey();
    uint64_t GetOpKey(ge::DataType inputXDtype, ge::DataType inputYDtype, ge::DataType outputZDtype);
    uint64_t GenerateTilingKey(uint64_t innerKey);
    std::map<uint64_t, BroadcastComputeParams> GetComputeMap(uint64_t opKey);

    uint64_t opKey_{0};
    uint64_t blockNum{0};
    std::map<std::tuple<ge::DataType, ge::DataType, ge::DataType>, uint64_t> opKeys;
    int64_t input0Dims[INPUT_MAX_DIMS] = {0};
    int64_t input1Dims[INPUT_MAX_DIMS] = {0};
    int64_t outputDims[INPUT_MAX_DIMS] = {0};
    int64_t input0Strides[INPUT_MAX_DIMS] = {0};
    int64_t input1Strides[INPUT_MAX_DIMS] = {0};
    int64_t outputStrides[INPUT_MAX_DIMS] = {0};
};

}  // namespace optiling
#endif  // POW_CANNDEV_OPS_BUILT_IN_OP_TILING_RUNTIME_POW_POW_TILING_H_
