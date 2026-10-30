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
 * \file non_finite_check.cpp
 * \brief
 */
 
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_function_def.h"
#include "../../../op_host/non_finite_check_tiling.h"
#include "data_utils.h"

using namespace NonFiniteCheckTest;

extern "C" __global__ __aicore__ void non_finite_check(
    GM_ADDR tensor_list, GM_ADDR found_flag, GM_ADDR workspace, GM_ADDR tiling);

class non_finite_check_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "non_finite_check_test SetUp\n" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "non_finite_check_test TearDown\n" << std::endl;
    }

    std::string GetShapesString(const std::vector<std::vector<uint64_t>>& shapeInfo)
    {
        std::string ret = "{";
        for (auto shape : shapeInfo) {
            ret += "{";
            for (auto dim : shape) {
                ret += std::to_string(dim) + ",";
            }
            ret += "},";
        }
        return ret + "}";
    }
};
