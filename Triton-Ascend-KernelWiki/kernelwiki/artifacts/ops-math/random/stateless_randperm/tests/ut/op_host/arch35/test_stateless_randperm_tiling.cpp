/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_randperm_tiling_arch35.h"

#include "../../../../op_kernel/stateless_randperm_struct.h"
#include "../../../../../../math/sort/op_kernel/arch35/sort_tiling_data.h"

using namespace std;
using namespace ge;

template <typename StructType, std::size_t Index>
struct StructMemberType;

#define DEFINE_STRUCT_MEMBER_TYPE(StructType, Index, Member)      \
    template <>                                                   \
    struct StructMemberType<StructType, Index> {                  \
        using type = decltype(std::declval<StructType>().Member); \
    };

template <typename T1, typename T2, std::size_t I>
struct CompareStructMembers {
    static constexpr bool
        value = std::is_same_v<typename StructMemberType<T1, I>::type, typename StructMemberType<T2, I>::type> &&
                CompareStructMembers<T1, T2, I - 1>::value;
};

template <typename T1, typename T2>
struct CompareStructMembers<T1, T2, 0> {
    static constexpr bool
        value = std::is_same_v<typename StructMemberType<T1, 0>::type, typename StructMemberType<T2, 0>::type>;
};

DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 0, numTileDataSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 1, unsortedDimParallel)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 2, lastDimTileNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 3, sortLoopTimes)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 4, lastDimNeedCore)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 5, keyParams0)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 6, keyParams1)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 7, keyParams2)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 8, keyParams3)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 9, keyParams4)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 10, keyParams5)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 11, tmpUbSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 12, lastAxisNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 13, unsortedDimNum)

DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 0, numTileDataSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 1, unsortedDimParallel)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 2, lastDimTileNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 3, sortLoopTimes)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 4, lastDimNeedCore)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 5, keyParams0)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 6, keyParams1)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 7, keyParams2)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 8, keyParams3)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 9, keyParams4)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 10, keyParams5)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 11, tmpUbSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 12, lastAxisNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 13, unsortedDimNum)

bool CheckTilingDataDefinitions()
{
    constexpr size_t size1 = sizeof(SortRegBaseTilingData);
    constexpr size_t size2 = sizeof(SortRegBaseTilingDataForRandperm);

    // 结构体字节长度是否相等
    if (size2 > size1) {
        std::cout << "Error: SortRegBaseTilingDataForRandperm size (" << size2
                  << ") larger than SortRegBaseTilingData size (" << size1 << ")" << std::endl;
        return false;
    }

    // 结构体每个成员数据类型是否相等
    constexpr size_t count1 = 14;
    constexpr bool
        typesMatch = CompareStructMembers<SortRegBaseTilingData, SortRegBaseTilingDataForRandperm, count1 - 1>::value;

    if (!typesMatch) {
        std::cout << "Error: Struct member type is not same." << std::endl;
        return false;
    }

    std::cout << "Success: Struct definitions are compatible." << std::endl;
    return true;
}

class StatelessRandpermTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessRandperm SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessRandperm TearDown" << std::endl; }
};

TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_001)
{
    optiling::StatelessRandpermCompileInfo compileInfo = {64, 196608};
    gert::StorageShape n_shape = {{1}, {1}};
    gert::StorageShape seed_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{7}, {7}};
    auto dtype = Ops::Math::AnyValue::CreateFrom<int64_t>(0);

    vector<int64_t> shape_value = {7};
    vector<int64_t> seed_value = {7};
    vector<int64_t> offset_value = {7};

    gert::TilingContextPara tilingContextPara("StatelessRandperm",
                                              {{n_shape, ge::DT_INT64, ge::FORMAT_ND, true, shape_value.data()},
                                               {seed_shape, ge::DT_INT64, ge::FORMAT_ND, true, seed_value.data()},
                                               {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
                                              {{out_shape, ge::DT_INT64, ge::FORMAT_ND}},
                                              {gert::TilingContextPara::OpAttr("dtype", dtype)}, &compileInfo);
    uint64_t expectTilingKey = 16843008;
    string expectTilingData = "4294967304 4294967297 4294967297 70 30064771073 0 0 0 30064771072 30064771072 7 "
                              "4294967303 4294967297 137438953473 34359738400 0 0 7 1 ";
    std::vector<size_t> expectWorkspaces = {16777286};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 该用例拦截Sort单方面修改tilingdata，防止StatelessRandperm功能异常
TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_struct_check)
{
    bool result = CheckTilingDataDefinitions();
    EXPECT_TRUE(result);
}
