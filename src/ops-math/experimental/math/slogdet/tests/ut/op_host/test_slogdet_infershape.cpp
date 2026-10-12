/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file test_slogdet_infershape.cpp
 * \brief Slogdet op_host InferShape / InferDataType UT。
 *
 * 覆盖点（spec outputs.shape_rule: signOut/logOut.shape = self.shape[:-2]）：
 *   - [n,n] ⇒ 标量输出 [] （batchRank=0）；
 *   - [b,n,n] ⇒ [b]；
 *   - 多维 batch [d0,d1,n,n] ⇒ [d0,d1]；
 *   - 动态维 -1 透传；
 *   - 非法 rank<2 ⇒ GRAPH_FAILED；
 *   - （迭代三）large-n 形状推导 [b,512,512] ⇒ [b]（shape 推导与 n 规模/BLOCKED 路径无关）；
 *   - （迭代三）高维 batch（spec rank 上界 8：6 batch 维 + 2 矩阵维）⇒ batch 形状透传。
 * 数据类型：两个输出 dtype = self.dtype（fp32）。
 *
 * 说明：infershape_case_executor 仅校验 OutputShape；dtype 由 InferDataType
 *       直接透传 self.dtype，shape 用例已覆盖 fp32 路径（输出描述 dtype 与 self 一致）。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SlogdetInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SlogdetInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SlogdetInfershape TearDown" << std::endl;
    }
};

// 从 vector<int64_t> 构造 StorageShape（origin == storage），规避只接受 initializer_list 的构造限制。
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 构造 单输入(self) + 双输出(signOut/logOut) 的 InfershapeContextPara。
static gert::InfershapeContextPara MakePara(const std::vector<int64_t>& selfShape,
                                            const std::vector<int64_t>& outShape)
{
    gert::StorageShape self = MakeStorageShape(selfShape);
    gert::StorageShape sign = MakeStorageShape(outShape);
    gert::StorageShape log = MakeStorageShape(outShape);
    return gert::InfershapeContextPara(
        "Slogdet",
        {{self, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{sign, ge::DT_FLOAT, ge::FORMAT_ND}, {log, ge::DT_FLOAT, ge::FORMAT_ND}});
}

// [n,n] ⇒ 标量输出 [] （signOut/logOut 都是空 shape）
TEST_F(SlogdetInfershape, slogdet_infershape_scalar_output)
{
    auto para = MakePara({5, 5}, {});
    std::vector<std::vector<int64_t>> expect = {{}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// [b,n,n] ⇒ [b]
TEST_F(SlogdetInfershape, slogdet_infershape_batch_1d)
{
    auto para = MakePara({3, 6, 6}, {3});
    std::vector<std::vector<int64_t>> expect = {{3}, {3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// [d0,d1,n,n] ⇒ [d0,d1]
TEST_F(SlogdetInfershape, slogdet_infershape_multi_dim_batch)
{
    auto para = MakePara({2, 4, 7, 7}, {2, 4});
    std::vector<std::vector<int64_t>> expect = {{2, 4}, {2, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// n=1 边界：[3,1,1] ⇒ [3]
TEST_F(SlogdetInfershape, slogdet_infershape_n1)
{
    auto para = MakePara({3, 1, 1}, {3});
    std::vector<std::vector<int64_t>> expect = {{3}, {3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// 动态维 -1 透传：[-1, 5, 5] ⇒ [-1]
TEST_F(SlogdetInfershape, slogdet_infershape_dynamic_batch)
{
    auto para = MakePara({-1, 5, 5}, {-1});
    std::vector<std::vector<int64_t>> expect = {{-1}, {-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// 非法 rank<2：[4] ⇒ GRAPH_FAILED
TEST_F(SlogdetInfershape, slogdet_infershape_invalid_rank_lt2)
{
    auto para = MakePara({4}, {});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 迭代三：large-n 形状推导（[2,512,512] ⇒ [2]）。InferShape 仅取 self.shape[:-2]，
// 与 n 规模/BLOCKED 路径完全无关，大 n 形状推导正确。
TEST_F(SlogdetInfershape, slogdet_infershape_large_n)
{
    auto para = MakePara({2, 512, 512}, {2});
    std::vector<std::vector<int64_t>> expect = {{2}, {2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// 迭代三：高维 batch（spec rank 上界 8 = 6 batch 维 + 2 矩阵维）。
// self=[2,3,2,3,2,3,4,4] ⇒ batch 形状 [2,3,2,3,2,3] 透传（高维 batch 形状推导）。
TEST_F(SlogdetInfershape, slogdet_infershape_high_dim_batch_rank8)
{
    auto para = MakePara({2, 3, 2, 3, 2, 3, 4, 4}, {2, 3, 2, 3, 2, 3});
    std::vector<std::vector<int64_t>> expect = {{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}
