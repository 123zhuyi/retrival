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
 * \file dynamic_partition_tiling.h
 * \brief head file of DynamicPartition tiling
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_PARTITION_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_PARTITION_TILING_H_

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "conversion/dynamic_partition/op_kernel/arch35/dynamic_partition_tiling_data_struct.h"

namespace optiling {
namespace DynPart {
struct DynamicPartitionCompileInfo {
    uint32_t coreNum{0};
    uint32_t ubSize{0};
    uint32_t clSize{0};
    uint32_t blockSize{0};
    bool isAscendC{false};
};

class DynamicPartitionTiling {
public:
    explicit DynamicPartitionTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetInputShapeAndType();
    ge::graphStatus GetAttrNumPartitions();
    bool CheckInputs();
    void ReshapeInputShape();
    void CalcTilingExtFirstOutDims();
    void CalcTilingUB();
    void CalcTilingHWLpUnit();
    void CalcTilingMCHWSize();
    void CalcTilingKey();
    void CalcTilingData();
    void CalcTilingDataNumPartOne();
    std::string PrintTilingData();
    ge::graphStatus WriteTilingData();

private:
    gert::TilingContext* context_{nullptr};
    const DynamicPartitionCompileInfo* compileInfo_{nullptr};
    ::DynPart::DynPartTilingData tilingData_;
    ::DynPart::DynPartNumPartOneTilingData numPartOneTilingData_;
    gert::Shape xShape_;
    gert::Shape partShape_;
    uint32_t dtypeSize_{1};
    bool isHBlockAxis_{false};
    uint32_t coreWS_{0};
};
} // namespace DynPart
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_PARTITION_TILING_H_
