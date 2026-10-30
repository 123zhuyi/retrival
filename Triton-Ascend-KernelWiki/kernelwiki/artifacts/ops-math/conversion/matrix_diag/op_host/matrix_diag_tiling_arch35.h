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
 * \file matrix_diag_tiling_arch35.h
 * \brief head file of MatrixDiag tiling
 */
#ifndef OPS_MATH_CONVERSION_MATRIX_DIAG_OP_HOST_MATRIX_DIAG_TILING_ARCH35_H_
#define OPS_MATH_CONVERSION_MATRIX_DIAG_OP_HOST_MATRIX_DIAG_TILING_ARCH35_H_

#include "../op_kernel/arch35/matrix_diag_tiling_data_struct.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
namespace MatrixDiagAsc {
constexpr int64_t BASE_2 = 2;
constexpr int32_t B8_BYTES = 1;
constexpr int32_t B16_BYTES = 2;
constexpr int32_t B32_BYTES = 4;
constexpr int32_t B64_BYTES = 8;
struct MatrixDiagCompileInfo {
    uint32_t coreNum{0};
    uint32_t ubSize{0};
    uint32_t clSize{0};
    uint32_t blockSize{0};
    bool isAscendC{false};
};

class MatrixDiagTiling {
public:
    explicit MatrixDiagTiling(gert::TilingContext* context) : context_(context){};
    ge::graphStatus DoTiling();

private:
    void CalcTilingData();
    void FuseInputShape();
    std::string PrintTilingData();
    ge::graphStatus GetInputShapeAndType();
    ge::graphStatus WriteTilingData();
    ge::graphStatus SetTilingData();
    template <typename T>
    ge::graphStatus SetTilingStruct(T& tilingSturct);
    void CalcPureCopyTilingData();
    void CalcScatterLowTilingData();
    void CalcScatterHighTilingData();
    void CalcScatterLargeTilingData();
    void CalcSimtTilingData();
    void SetBlockSplitInfo(int64_t batchBlockCnt, int64_t nBlockCnt, int64_t batchSize, int64_t nSize);

private:
    gert::TilingContext* context_ = nullptr;
    const MatrixDiagCompileInfo* compileInfo_ = nullptr;
    ::MatrixDiagAsc::MatrixDiagScatterTilingData tilingDataScatter_;
    ::MatrixDiagAsc::MatrixDiagSimtTilingData tilingDataSimt_;
    ::MatrixDiagAsc::MatrixDiagPureCopyTilingData tilingDataPureCopy_;
    ::MatrixDiagAsc::MatrixDiagScatterLargeTilingData tilingDataLarge_;
    gert::Shape inputShape_;
    int64_t fusedShape_[BASE_2] = {0};
    int64_t xDtypeSize_ = 0;
    uint64_t tilingKey_ = 0;
    int32_t realCoreNum_ = 0;
    int64_t mBlockCount_ = 0;  // M轴方向核切分块数
    int64_t mBlockFactor_ = 0; // M轴核切分的 main_factor -- tail_factor=main_factor-1
    int64_t mBlockFactorTail_ = 0;
    int64_t mBlockFactorCount_ = 0; // M轴核切分的主块个数 -- 尾块个数=mBlockCount_ - mBlockFactorCount_
    int64_t nBlockCount_ = 0;
    int64_t nBlockFactor_ = 0;
    int64_t nBlockFactorTail_ = 0;
    int64_t nBlockFactorCount_ = 0;
    int64_t ubSize_ = 0;
};
} // namespace MatrixDiagAsc
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MATRIX_DIAG_TILING_H_
