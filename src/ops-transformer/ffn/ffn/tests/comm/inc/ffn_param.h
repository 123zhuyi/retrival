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
 * \file ffn_param.h
 * \brief FFN 参数信息.
 */

#ifndef UTEST_FFN_PARAM_H
#define UTEST_FFN_PARAM_H

#include <register/op_impl_registry.h>
#include "tests/utils/tensor.h"

namespace ops::adv::tests::ffn {

using ops::adv::tests::utils::Tensor;

class Param {
public:
    std::map<std::string, Tensor> mTensors;
    std::vector<int64_t> mExpertTokensData = {};
    std::string mActivation;
    int32_t mInnerPrecise = 0;
    int32_t mOutputDtype = -1;
    bool mTokensIndexFlag = false;

public:
    Param() = default;
    Param(std::vector<Tensor> inputs, std::vector<int64_t> expertTokensData, std::string activation,
          int32_t innerPrecise, int32_t outputDtype, bool tokensIndexFlag = false);
};

} // namespace ops::adv::tests::ffn
#endif // UTEST_FFN_PARAM_H
