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
 * \file clip_by_value_v2_tiling.h
 * \brief clip_by_value_v2 tiling head file
 */

#ifndef OPS_MATH_CONVERSION_CLIP_BY_VALUE_V2_TILING_H
#define OPS_MATH_CONVERSION_CLIP_BY_VALUE_V2_TILING_H

#include "op_host/tiling_base_class.h"

namespace optiling {

struct ClipByValueV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class ClipByValueV2Tiling : public Ops::Base::TilingBaseClass {
public:
    explicit ClipByValueV2Tiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
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
    uint64_t tilingKey = 0;
    bool CheckDtype(
        const ge::DataType& xDtype, const ge::DataType& minDtype, const ge::DataType& maxDtype,
        const ge::DataType& yDtype) const;
};

} // namespace optiling

#endif // OPS_MATH_CONVERSION_CLIP_BY_VALUE_V2_TILING_H
