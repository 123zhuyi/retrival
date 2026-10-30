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
 * \file range_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_RANGE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_RANGE_H_

#include <cstdint>
#include <vector>

#include "log/log.h"
#include "op_host/util/fp16.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"
#include "util/math_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RangeTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);     // tilingKey的大小
TILING_DATA_FIELD_DEF(int64_t, workspaceSize); // 总workspace的大小
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Range, RangeTilingData)

BEGIN_TILING_DATA_DEF(RangeTilingDataInt)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);     // 使用核数
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);    // 物理总核数
TILING_DATA_FIELD_DEF(int64_t, hardwareUbSize);  // 总UB可用的大小
TILING_DATA_FIELD_DEF(int64_t, totalElementNum); // 总处理元素个数
TILING_DATA_FIELD_DEF(int64_t, numOfPerCore);    // 非尾核处理元素个数
TILING_DATA_FIELD_DEF(int64_t, numOfTailCore);   // 尾核处理元素个数
TILING_DATA_FIELD_DEF(int64_t, ubOneLoopNum);    // 单次loop处理元素个数上限
TILING_DATA_FIELD_DEF(int64_t, loopOfPerCore);   // 非尾核中loop次数
TILING_DATA_FIELD_DEF(int64_t, perOfPerCore);    // 非尾核中非尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tailOfPerCore);   // 非尾核中尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, loopOfTailCore);  // 尾核中loop次数
TILING_DATA_FIELD_DEF(int64_t, perOfTailCore);   // 尾核中非尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tailOfTailCore);  // 尾核中尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tilingKey);       // tilingKey的大小
TILING_DATA_FIELD_DEF(int64_t, workspaceSize);   // 总workspace的大小
TILING_DATA_FIELD_DEF(int64_t, start);
TILING_DATA_FIELD_DEF(int64_t, delta);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Range_1001, RangeTilingDataInt)
REGISTER_TILING_DATA_CLASS(Range_1002, RangeTilingDataInt)

BEGIN_TILING_DATA_DEF(RangeTilingDataFloat)
TILING_DATA_FIELD_DEF(int64_t, loopOfTailCore);  // 尾核中loop次数
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);    // 物理总核数
TILING_DATA_FIELD_DEF(int64_t, workspaceSize);   // 总workspace的大小
TILING_DATA_FIELD_DEF(int64_t, hardwareUbSize);  // 总UB可用的大小
TILING_DATA_FIELD_DEF(int64_t, tilingKey);       // tilingKey的大小
TILING_DATA_FIELD_DEF(int64_t, numOfPerCore);    // 非尾核处理元素个数
TILING_DATA_FIELD_DEF(int64_t, numOfTailCore);   // 尾核处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tailOfTailCore);  // 尾核中尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, ubOneLoopNum);    // 单次loop处理元素个数上限
TILING_DATA_FIELD_DEF(int64_t, perOfTailCore);   // 尾核中非尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tailOfPerCore);   // 非尾核中尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, loopOfPerCore);   // 非尾核中loop次数
TILING_DATA_FIELD_DEF(int64_t, totalElementNum); // 总处理元素个数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);     // 使用核数
TILING_DATA_FIELD_DEF(int64_t, perOfPerCore);    // 非尾核中非尾loop处理元素个数
TILING_DATA_FIELD_DEF(float, start);
TILING_DATA_FIELD_DEF(float, delta);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Range_1003, RangeTilingDataFloat)
REGISTER_TILING_DATA_CLASS(Range_1004, RangeTilingDataFloat)
REGISTER_TILING_DATA_CLASS(Range_1005, RangeTilingDataFloat)

struct RangeTilingParam {
    int64_t usedCoreNum;
    int64_t totalCoreNum;
    int64_t hardwareUbSize;
    int64_t totalElementNum;
    int64_t numOfPerCore;
    int64_t numOfTailCore;
    int64_t ubOneLoopNum;
    int64_t loopOfPerCore;
    int64_t perOfPerCore;
    int64_t tailOfPerCore;
    int64_t loopOfTailCore;
    int64_t perOfTailCore;
    int64_t tailOfTailCore;
    int64_t tilingKey;
};

constexpr int32_t INT32_TILING_KEY = 1001;
constexpr int32_t INT64_TILING_KEY = 1002;
constexpr int32_t FP32_TILING_KEY = 1003;
constexpr int32_t FP16_TILING_KEY = 1004;
constexpr int32_t BF16_TILING_KEY = 1005;
constexpr size_t RESERVED_WORKSPACE = static_cast<size_t>(16 * 1024 * 1024);

class RangeRegBaseTilingClass : public Ops::Base::TilingBaseClass {
public:
    explicit RangeRegBaseTilingClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
        outDataType_ = context_->GetOutputDesc(0)->GetDataType();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override;

    RangeTilingParam tilingParam_;
    ge::DataType outDataType_;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_RANGE_H_