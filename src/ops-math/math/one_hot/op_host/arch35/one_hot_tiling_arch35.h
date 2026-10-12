/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file one_hot_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ONE_HOT_TILING_ARCH35_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ONE_HOT_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include <numeric>
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
using namespace std;
const static int32_t ONEHOT_INPUT_DEPENDENCY_IDX = 1;
ge::graphStatus OneHotTilingForAscendC(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(OneHotTilingData)
TILING_DATA_FIELD_DEF(int16_t, hasTail);
TILING_DATA_FIELD_DEF(int16_t, usedCoreNum); // 实际使用的核数
TILING_DATA_FIELD_DEF(int16_t, offValueHasTail);
TILING_DATA_FIELD_DEF(int16_t, offValueUsedCoreNum); // 尾核尾循环要处理的数据大小
TILING_DATA_FIELD_DEF(int64_t, perCoreCalSize);      // 单次循环要处理的数据大小
TILING_DATA_FIELD_DEF(int64_t, offValuePerCoreCalSize);
TILING_DATA_FIELD_DEF(int64_t, offValueCalTailSize);
TILING_DATA_FIELD_DEF(int64_t, tailCalSize);
TILING_DATA_FIELD_DEF(int64_t, prefixDim);
TILING_DATA_FIELD_DEF(int64_t, suffixDim);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OneHot, OneHotTilingData)

class OneHotTilingBase : public Ops::Base::TilingBaseClass {
public:
    explicit OneHotTilingBase(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {};
    ~OneHotTilingBase() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetTilingParam(void);
    ge::graphStatus CalUbTilingData(void);
    ge::graphStatus MergeDims(void);
    bool AnalyzeInputs(void);
    bool AnalyzeDtypeAndFormat(void);
    bool AnalyzeAttrs(void);
    bool CheckSingleShape(const gert::Shape& shape, const string name) const;

private:
    uint64_t workspaceSize_;
    int32_t aivNum_;
    int64_t ubSize_;
    int32_t axis_;
    int32_t depth_;
    uint64_t inputTotalDimNum_;
    bool isUsedUbInit_;
    const char* opName_;

    static constexpr int32_t RESERVED_SIZE = 1024 * 5;
    static constexpr int32_t INPUT_X_IDX = 0;
    static constexpr int32_t INPUT_DEPTH_IDX = 1;
    static constexpr int32_t INPUT_ON_VALUE_IDX = 2;
    static constexpr int32_t INPUT_OFF_VALUE_IDX = 3;
    static constexpr int32_t OUTPUT_IDX = 0;
    static constexpr int32_t ATTR_AXIS_IDX = 0;
    static constexpr int32_t OFFVALUE_INIT_SIZE = 190 * 1024;
    static constexpr int32_t DEPTH_THRESHHOLD = 3;
    static constexpr uint32_t TOTALSIZE_THRESHHOLD = 1024;
    static constexpr int32_t MIN_UB_CAL_SIZE = 64;
    static constexpr uint32_t MIN_FACTOR = 1024;

    ge::DataType indicesType;
    ge::DataType depthType;
    ge::DataType onValueType;
    ge::DataType offValueType;
    ge::DataType outputType;
    OneHotTilingData tilingData_;
    gert::Shape inputOriginShape;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ONE_HOT_TILING_ARCH35_H_