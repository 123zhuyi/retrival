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
 * \file stateless_sample_multinomial.h
 * \brief Op API header for StatelessSampleMultinomial
 */
#ifndef STATELESS_SAMPLE_MULTINOMIAL_OP_API_H
#define STATELESS_SAMPLE_MULTINOMIAL_OP_API_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Generate multinomial samples with replacement using binary search on x.
 *        Fuses U(0,1] generation (Philox RNG) + binary search for direct index output.
 *        Handles zero-probability categories via backward walk (PyTorch CUDA compatible).
 *        numDist and numCategories are derived from xTensor shape.
 *
 * @param xTensor Input tensor (DT_FLOAT, shape [numDist, numCategories] or [numCategories])
 * @param seedTensor    Seed tensor (INT64/UINT64, shape [1])
 * @param offsetTensor  Offset tensor (INT64/UINT64, shape [1])
 * @param numsamples    Number of samples per distribution
 * @param executor      Op executor
 * @return Output tensor with shape {numDist, numsamples}, dtype DT_INT64
 */
const aclTensor* StatelessSampleMultinomial(
    const aclTensor* xTensor,
    const aclTensor* seedTensor,
    const aclTensor* offsetTensor,
    int64_t numsamples,
    aclOpExecutor* executor);

/**
 * @brief Generate exponential random samples on 950 AICore for multinomial without replacement.
 *        Uses SimThreadExponential kernel with Philox RNG (offset aligned to 4).
 *
 * @param self          Input tensor, determines output shape and dtype
 * @param seed          RNG seed (scalar)
 * @param offset        RNG offset (scalar, will be aligned to multiple of 4)
 * @param lambd         Lambda parameter for exponential distribution
 * @param executor      Op executor
 * @return Output tensor with same shape and dtype as self
 */
const aclTensor* Run950AicoreExponentialWithoutReplacement(
    const aclTensor* self, int64_t seed, int64_t offset, float lambd, aclOpExecutor* executor);

} // namespace l0op

#endif // STATELESS_SAMPLE_MULTINOMIAL_OP_API_H
