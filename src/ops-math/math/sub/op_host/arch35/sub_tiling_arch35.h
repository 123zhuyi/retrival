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
 * \file sub_tiling_arch35.h
 * \brief sub_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_SUB_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_SUB_TILING_H

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"

namespace optiling {
struct SubCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class SubTiling : public Ops::Base::TilingBaseClass {
public:
    explicit SubTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {}

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
    uint64_t tilingKey = 0;
    bool CheckDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype,
                    const ge::DataType& outputDtype) const;
};

}  // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_SUB_TILING_H
