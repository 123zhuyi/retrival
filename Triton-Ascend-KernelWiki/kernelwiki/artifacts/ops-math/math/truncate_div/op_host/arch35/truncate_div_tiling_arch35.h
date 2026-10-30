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
 * \file truncate_div_tiling_arch35.h
 * \brief truncate_div_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_TRUNCATE_DIV_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_TRUNCATE_DIV_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

class TruncateDivTiling : public Ops::Base::TilingBaseClass {
public:
    explicit TruncateDivTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    uint64_t tilingKey_ = 0;
    float reciprocal_ = 0.0f;
    int64_t ubSize_ = 0;
    uint32_t schMode_ = 0;
    bool canUseMul_ = false;

    template <typename T>
    ge::graphStatus GetConstData(uint32_t inputIdx, T& data);

    float GetReciprocal(float data);

    template <typename OpDag>
    ge::graphStatus ExecTiling(bool isScalarBranch = false);

    ge::graphStatus GetScalarReciprocal(ge::DataType x2DType);

    ge::graphStatus SelectAndExecTiling(ge::DataType x1DType, ge::DataType x2DType);

    ge::graphStatus HandleFloat16WithFloat();
    ge::graphStatus HandleFloat16OrBf16();
    ge::graphStatus HandleFloat(ge::DataType x2DType);
    ge::graphStatus HandleIntTypes(ge::DataType x1DType, ge::DataType x2DType);
    ge::graphStatus HandleInt64();
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_TRUNCATE_DIV_TILING_H
