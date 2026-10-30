/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file eye_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_EYE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_EYE_H_
#include "register/op_def_registry.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"

namespace optiling {
// ///////////////////////////////////
// tilingdata define
// ///////////////////////////////////
BEGIN_TILING_DATA_DEF(EyeForAscendCTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, normBlockData);
TILING_DATA_FIELD_DEF(int64_t, tailBlockData);
TILING_DATA_FIELD_DEF(int64_t, loopLength);
TILING_DATA_FIELD_DEF(int64_t, numRows);
TILING_DATA_FIELD_DEF(int64_t, numColumns);
TILING_DATA_FIELD_DEF(int64_t, batch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Eye, EyeForAscendCTilingData)

struct EyeCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class EyeTiling : public Ops::Base::TilingBaseClass {
public:
    explicit EyeTiling(gert::TilingContext *context) : Ops::Base::TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus CheckOutputShape();

private:
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t loopLength_ = 0;
    int64_t normBlockData_ = 0;
    int64_t tailBlockData_ = 0;
    int64_t numRows_ = 0;
    int64_t numColumns_ = 0;
    int64_t batch_ = 1;
    int64_t allAxis_ = 0;
    int64_t typeSize_ = 0;

    ge::DataType dType_ = ge::DT_UNDEFINED;
    EyeForAscendCTilingData tilingData_;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_EYE_H_