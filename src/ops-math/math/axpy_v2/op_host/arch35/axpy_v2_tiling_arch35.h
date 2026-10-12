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
 * \file axpy_v2_tiling_arch35.h
 * \brief axpy_v2_tiling_arch35 head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_AXPY_V2_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_AXPY_V2_TILING_H

#include "op_host/tiling_base_class.h"

namespace optiling
{
struct AxpyV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class AxpyV2Tiling : public Ops::Base::TilingBaseClass
{
public:
    explicit AxpyV2Tiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;

private:
    uint64_t tilingKey = 0;
    bool CheckDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype,
                    const ge::DataType& inputAlphaDtype, const ge::DataType& outputDtype) const;
};

}  // namespace optiling

#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_AXPY_V2_TILING_H
