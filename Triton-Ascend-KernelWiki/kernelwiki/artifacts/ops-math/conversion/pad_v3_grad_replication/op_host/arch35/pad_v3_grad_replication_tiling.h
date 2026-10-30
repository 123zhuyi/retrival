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
 * \file pad_v3_grad_replication_tiling.h
 * \brief tiling header for pad_v3_grad_replication operator
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_REPLICATION_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_REPLICATION_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "conversion/pad_v3_grad_replication/op_kernel/arch35/pad_v3_grad_replication_struct.h"
#include "conversion/pad_v3_grad_replication/op_kernel/arch35/pad_v3_grad_replication_tilingkey.h"
#include <string>

namespace optiling {

struct PadV3GradReplicationCompileInfo {
    int64_t core_num;
    uint64_t ub_size;
    int64_t total_ub_size;
    int64_t sysWorkspaceSize;
    int64_t dtype_rate;
    int64_t x_bytes_size;
    std::string soc_version;
};

class PadV3GradReplicationTiling {
public:
    explicit PadV3GradReplicationTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    uint64_t GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize);

    ge::graphStatus Init();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus GetPaddings();

    template <typename T>
    void GetPaddingsToShape(const gert::Tensor* paddingsTensor);

    void CalcStrideAligned();
    void CalcSplitStrategy();
    bool TrySplitAxis(uint32_t axis, uint64_t ubAvailable);
    bool IsPaddingDim(uint32_t axis) const;
    uint64_t CalcWorstFactor(uint32_t axis) const;
    void CalcUsedCore();
    void FillTilingData(PadV3GradReplicationTilingData* tilingData);

    template <typename T>
    std::string ToString(const T* value, size_t size);

private:
    gert::TilingContext* context_ = nullptr;
    uint32_t coreNum_{0};
    uint64_t ubSize_{0};
    uint64_t blockSize_{0};

    uint8_t dimNum_{1};
    uint8_t splitAxis_{0};
    uint32_t splitCount_{0};
    uint32_t splitSize_{0};
    uint32_t usedCoreNum_{0};
    uint32_t tilesPerCore_{0};

    uint64_t inputShape_[PAD_GRAD_REPLICATION_MAX_DIMS_NUM] = {0};
    uint64_t outputShape_[PAD_GRAD_REPLICATION_MAX_DIMS_NUM] = {0};
    uint64_t strideAligned_[PAD_GRAD_REPLICATION_MAX_DIMS_NUM] = {0};
    int64_t leftPad_[PAD_GRAD_REPLICATION_MAX_DIMS_NUM] = {0}; // 所有维度左padding（实际运行时前N-5维为0）
    int64_t rightPad_[PAD_GRAD_REPLICATION_MAX_DIMS_NUM] = {0}; // 所有维度右padding（实际运行时前N-5维为0）;

    uint32_t padDimNum_{0};
    uint32_t padDimIndices_[PAD_GRAD_REPLICATION_MAX_PAD_DIMS_NUM] = {0};

    uint32_t dataSize_{0}; // 数据类型大小（字节）

    ge::DataType paramsDtype_;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_REPLICATION_TILING_H_
