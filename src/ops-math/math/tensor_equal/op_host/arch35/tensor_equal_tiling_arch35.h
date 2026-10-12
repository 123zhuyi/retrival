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
 * \file tensor_equal_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TENSOR_EQUAL_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TENSOR_EQUAL_H
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
TILING_DATA_FIELD_DEF(int64_t, inputShapeSize);
TILING_DATA_FIELD_DEF(int64_t, inputDtypeSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoopTimes);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopTimes);
TILING_DATA_FIELD_DEF(int64_t, perCoreTailFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailFactor);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)

class TensorEqualTiling : public Ops::Base::TilingBaseClass {
public:
    explicit TensorEqualTiling(gert::TilingContext *context) : Ops::Base::TilingBaseClass(context), opName_(context->GetNodeName()) {}

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
    // 8、打印结果
    void DumpTilingInfo() override;

private:
    ge::graphStatus CheckDType();
    void SetTilingData();

private:
    const std::string opName_;

    ge::DataType inputXDType_;
    ge::DataType inputYDType_;
    ge::DataType outputDType_;

    int64_t inputShapeSize_ {0};
    int64_t inputDtypeSize_ {0};
    int64_t usedCoreNum_ {0};
    int64_t ubSize_ {0};
    int64_t vRegSize_ {0};
    int64_t ubFactor_ {0};
    int64_t platformUbFactor_ {0};
    int64_t tilingKey_ {0};
    int64_t perCoreLoopTimes_ {0};
    int64_t tailCoreLoopTimes_ {0};
    int64_t perCoreTailFactor_ {0};
    int64_t tailCoreTailFactor_ {0};
    int64_t totalCoreNum_ {0};

    TensorEqualTilingData tilingData_;
};

struct TensorEqualCompileInfo {
    int64_t totalCoreNum {0};
    uint64_t ubSizePlatForm {0};
};
} // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TENSOR_EQUAL_H