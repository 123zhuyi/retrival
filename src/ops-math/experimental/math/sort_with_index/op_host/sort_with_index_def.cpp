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
 * \file sort_with_index_def.cpp
 * \brief SortWithIndex op_host op definition (ascend910b native AscendC).
 *
 * dtype scope: 910B declares ONLY the 4 (value, index) pairs value{fp16,fp32,bf16,int32} x
 * index{int32}. The int64-index kernel code is RETAINED (double int32-view Gather) but NOT declared
 * here: the 910B binary-select runtime forces sorted_index = int32 for this sort family on non-RegBase
 * (DAV_2201), so an int64 binary can never be matched through aclnn. int64-index is left for Ascend950
 * RegBase / a framework int64-binary-select opening. The DataType lists below are POSITIONAL: index k
 * across the four lists (x / index / y / sorted_index) forms one combination. x and y always share the
 * value dtype; index and sorted_index always share the index dtype.
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
// 4 (value, index) combinations (910B, index int32 only). Position-aligned:
//   pos: 0      1      2      3
//   x  : fp16   fp32   bf16   int32
//   idx: int32  int32  int32  int32
// The TilingKey 3-D dispatch (VALUE_DT/INDEX_DT via DTYPE_* macros + SIZE_MODE via ASCENDC_TPL
// schMode) selects the matching kernel template instantiation for each pair.
static const std::vector<ge::DataType> DataTypeValue = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32};
static const std::vector<ge::DataType> DataTypeIndex = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
static const std::vector<ge::Format> Formats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class SortWithIndex : public OpDef {
public:
    explicit SortWithIndex(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(DataTypeValue)
            .Format(Formats)
            .UnknownShapeFormat(Formats);
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType(DataTypeIndex)
            .Format(Formats)
            .UnknownShapeFormat(Formats);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(DataTypeValue)
            .Format(Formats)
            .UnknownShapeFormat(Formats);
        this->Output("sorted_index")
            .ParamType(REQUIRED)
            .DataType(DataTypeIndex)
            .Format(Formats)
            .UnknownShapeFormat(Formats);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("descending").AttrType(OPTIONAL).Bool(false);
        this->Attr("stable").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sort_with_index");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(SortWithIndex);
}  // namespace ops
