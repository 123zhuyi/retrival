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
 * \file random_uniform_int_v2_tiling_arch35.h
 * \brief
 */
#ifndef RANDOM_UNIFORM_INT_V2_TILING_ARCH35_H
#define RANDOM_UNIFORM_INT_V2_TILING_ARCH35_H

#include <string>
#include "op_host/tiling_base_class.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/random_uniform_int_v2_struct.h"

namespace optiling {

static constexpr uint16_t IN_SHAPE_IDX = 0;
static constexpr uint16_t IN_MIN_IDX = 1;
static constexpr uint16_t IN_MAX_IDX = 2;
static constexpr uint16_t IN_OFFSET_IDX = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t ATTR_SEED_IDX = 0;
static constexpr uint16_t ATTR_SEED2_IDX = 1;
static constexpr int64_t CORE_ALIGN_SIZE = 512;
static constexpr uint32_t MIN_TILING_SIZE = 256;
static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 0;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DCACHE_SIZE = 32 * 1024;

class RandomUniformIntV2Tiling : public Ops::Base::TilingBaseClass
{
public:
    explicit RandomUniformIntV2Tiling(gert::TilingContext* context)
        : Ops::Base::TilingBaseClass(context), opName_(context->GetNodeName())
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void DumpTilingInfo() override;

private:
    const std::string opName_;
    int64_t ubSize_{0};
    int64_t totalCoreNum_{0};
    int64_t shapeSize_{1};
    int64_t outputSize_{0};
    int64_t seed_{0};
    int64_t seed2_{0};
    ge::DataType outDtype_{ge::DT_INT64};
    ge::DataType minDtype_{ge::DT_INT64};

    int64_t outputDtypeSize_{0};
    int64_t blockNum_{0};
    int64_t normalCoreProNum_{0};
    int64_t tailCoreProNum_{0};
    int64_t singleUbSize_{0};
    uint64_t range_{0};
    int64_t lo_{0};

    void SetTilingData();
    template <typename T>
    ge::graphStatus GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape);
    ge::graphStatus GetIntValueByDtype(const gert::Tensor *constTensor, gert::Shape &constShape, ge::DataType dType);
    ge::graphStatus GetMinAndMaxValue();
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetOutputInfo();
    ge::graphStatus GetAttrInfo();
    void DoBlockTiling();
    void UbTiling();
};

struct RandomUniformIntV2CompileInfo {
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

} // namespace optiling

#endif // RANDOM_UNIFORM_INT_V2_TILING_ARCH35_H