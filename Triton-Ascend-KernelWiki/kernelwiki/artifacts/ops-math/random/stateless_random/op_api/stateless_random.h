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
 * \file stateless_random.h
 * \brief Stateless random number generation API header
 */

#ifndef STATELESS_RANDOM_API_H_
#define STATELESS_RANDOM_API_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* StatelessRandom(
    const aclTensor* self, int64_t seed, int64_t offset,
    int64_t from, int64_t to, aclOpExecutor* executor);


const aclTensor* StatelessRandom(
    const aclTensor* self,
    const aclTensor* seedTensor, const aclTensor* offsetTensor,
    int64_t from, int64_t to, aclOpExecutor* executor);

const aclTensor* StatelessRandomWithoutFromTo(
    const aclTensor* self, int64_t seed, int64_t offset, aclOpExecutor* executor);


const aclTensor* StatelessRandomWithoutFromTo(
    const aclTensor* self, const aclTensor* seedTensor, const aclTensor* offsetTensor, aclOpExecutor* executor);

} // namespace l0op

#endif // STATELESS_RANDOM_API_H_