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
 * \file quant_base_block_calculator.cpp
 * \brief BaseBlockCalculator subclass that forces baseN 32-alignment for MX quant.
 */
#include "quant_base_block_calculator.h"

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "util/math_util.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"

namespace optiling {

using Ops::NN::MathUtil;

namespace {
constexpr uint64_t MX_BASEN_ALIGN = 32UL;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
} // namespace

QuantBaseBlockCalculator::QuantBaseBlockCalculator(const QuantBatchMatmulInfo& inputParams,
                                                   const QuantBatchMatmulV3CompileInfo& compileInfo,
                                                   uint64_t batchCoreCnt)
    : BaseBlockCalculator(inputParams, compileInfo, batchCoreCnt),
      quantInputParams_(inputParams),
      quantCompileInfo_(compileInfo)
{}

uint64_t QuantBaseBlockCalculator::GetBaseNAlignSize(uint64_t innerAlignSize) const
{
    return this->inputParams_.transB ? MX_BASEN_ALIGN : GetShapeWithDataType(innerAlignSize, this->inputParams_.bDtype);
}

} // namespace optiling
