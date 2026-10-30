/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_stitch_def.cpp
 * \brief op config of dynamic_stitch
 */
#include <vector>
#include "register/op_def_registry.h"
#include "graph/operator.h"

namespace {
using namespace ge;

constexpr int32_t AICORE_CHECK_LIST_CNT = 64;

const std::vector<DataType> xDType = {
    DT_INT8,   DT_UINT8, DT_INT16,   DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
    DT_UINT64, DT_BOOL,  DT_FLOAT16, DT_BF16,   DT_FLOAT, DT_DOUBLE, DT_COMPLEX64};
const std::vector<DataType> indicesDType = {
    DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32,
    DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
const std::vector<Format> xFormat = {
    FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND,
    FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND};

graphStatus CheckIfAICoreSupported(const Operator& op, AscendString& result)
{
    int32_t attrN = 0;
    graphStatus retStatus = op.GetAttr("N", attrN);
    if (retStatus != GRAPH_SUCCESS) {
        result = AscendString(R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "GetAttr N error"})");
        return GRAPH_FAILED;
    }
    if (attrN < 1 || attrN > AICORE_CHECK_LIST_CNT) {
        result = AscendString(R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "attr out of range."})");
        return GRAPH_FAILED;
    }
    result = AscendString(R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "CheckSupported success."})");
    return GRAPH_SUCCESS;
}

} // namespace

namespace ops {
class DynamicStitch : public OpDef
{
public:
    explicit DynamicStitch(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(DYNAMIC)
            .DataType(indicesDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(xDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Attr("N").AttrType(OPTIONAL).Int(1);
        this->AICore().SetCheckSupport(CheckIfAICoreSupported);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "dynamic_stitch_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(DynamicStitch);
// 手动注册opDef.AICore()里设置的CheckSupport函数
// 需要当前目录下的CMakeLists.txt将本_def.cpp加入${OPHOST_NAME}_tiling_obj编译目标内
static int g_DynamicStitch_register_check_support = [](const char* name) {
    DynamicStitch opDef(name);
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, name, opDef.AICore().GetCheckSupport());
    return 0;
}("DynamicStitch");

} // namespace ops
