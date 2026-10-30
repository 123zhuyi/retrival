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
 * \file mod_tiling.h
 * \brief mod_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADDCDIV_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADDCDIV_TILING_H

#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

using namespace Ops::Base;
class ModTiling : public Ops::Base::TilingBaseClass {
public:
    explicit ModTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;

    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool CheckDtype(
        const ge::DataType& inputX1Dtype, const ge::DataType& inputX2Dtype, const ge::DataType& outputDtype) const;

private:
    uint64_t tilingKey = 0;
    int64_t ubSize_ = 0;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADDCDIV_TILING_H
