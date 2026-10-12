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
 * \file log_space_def.cpp
 * \brief LogSpace 算子定义：属性 start/end/steps/base → 输出 result
 *        Ascend950（原有）: FLOAT / FLOAT16 / BF16
 *        Atlas A2 训练系列产品 (ascend910b) / Atlas A3 系列产品 (ascend910_93)（本次新增）:
 *            上述浮点 + INT8 / INT16 / INT32 / UINT8
 */
#include "register/op_def_registry.h"

namespace ops {
class LogSpace : public OpDef {
public:
    explicit LogSpace(const char* name) : OpDef(name)
    {
        // LogSpace 是纯生成类算子：无输入 tensor，仅输出 result
        // OpDef 级声明 dtype 全集（浮点 + 整型）作为基准约定：A2/A3 (ascend910b/910_93) 与之一致，
        // 直接继承；仅 ascend950 因整型未验证，由下方 AddConfig 的 OpAICoreConfig 收窄为浮点。
        // 整型路径在 kernel 内以 fp32 计算 base^x 后末步 Cast 落整型。
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 属性：start/end 已在 Host 侧由 aclScalar 转为 float 传入；base 同样转为 float
        this->Attr("start").AttrType(REQUIRED).Float();
        this->Attr("end").AttrType(REQUIRED).Float();
        this->Attr("steps").AttrType(REQUIRED).Int();
        this->Attr("base").AttrType(REQUIRED).Float();

        // Ascend950：维持原有能力，仅浮点输出。整型是本次在 Atlas A2/A3 上新增的，未在 950 上验证，
        // 故此处按芯片收窄 dtype，与 op_api/log_space.cpp 的 DAV_3510 门控保持一致；否则整型组合会出现
        // 「def 注册了、aclnn 却拒绝」的不一致（图模式据 def 选中下发，aclnn 路径却报错）。
        OpAICoreConfig aiCoreConfig950;
        aiCoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "log_space");
        aiCoreConfig950.Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->AICore().AddConfig("ascend950", aiCoreConfig950);

        // Atlas A2 训练系列产品 (ascend910b) / Atlas A3 系列产品 (ascend910_93)：本次新增，浮点 + 整型。
        // 支持范围与 OpDef 级声明一致，故此 config 只设 flag、省去 Output 重声明，继承 OpDef 级的 7 种
        // dtype（同 angle_v2 的 ascend910b/910/910_93 flags-only config 约定；Output 仅在 dtype 与
        // OpDef 级不同的 SoC 上覆写，见上方 ascend950）。
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "log_space");
        this->AICore().AddConfig("ascend910b", aiCoreConfig);
        this->AICore().AddConfig("ascend910_93", aiCoreConfig);
    }
};
OP_ADD(LogSpace);
} // namespace ops
