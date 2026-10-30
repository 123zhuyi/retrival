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
 * \file log_add_exp_tiling_arch35.h
 * \brief log_add_exp_tiling_arch35 head file
 */

#ifndef OPS_MATH_LOG_ADD_EXP_OP_HOST_LOG_ADD_EXP_TILING_ARCH35_H
#define OPS_MATH_LOG_ADD_EXP_OP_HOST_LOG_ADD_EXP_TILING_ARCH35_H

#include "op_host/tiling_base_class.h"

namespace optiling {
using namespace Ops::Base;

struct LogAddExpCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class LogAddExpTiling : public Ops::Base::TilingBaseClass {
public:
    explicit LogAddExpTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckDtype(ge::DataType& input0Dtype);
    ge::graphStatus DoFullFormulaTiling(ge::DataType input0Dtype);
    ge::graphStatus DoSimplifiedTiling(ge::DataType input0Dtype);

    uint64_t tilingKey = 0;
    float base_ = -1.0f;
    float scale_ = 1.0f;
    float shift_ = 0.0f;
    bool useFullFormula_ = false;
};

} // namespace optiling

#endif // OPS_MATH_LOG_ADD_EXP_OP_HOST_LOG_ADD_EXP_TILING_ARCH35_H
