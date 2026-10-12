/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

 
 class GroupedMatmulFinalizeRouting : public testing::Test {
 protected:
     static void SetUpTestCase() {
         std::cout << "GroupedMatmulFinalizeRouting Proto Test SetUp" << std::endl;
     }
 
     static void TearDownTestCase() {
         std::cout << "GroupedMatmulFinalizeRouting Proto Test TearDown" << std::endl;
     }
 };
 
 TEST_F(GroupedMatmulFinalizeRouting, grouped_matmul_finalize_routing_1) {
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bs = 64;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, n}, {e, n}};
    gert::StorageShape pertoken_scaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};
    gert::StorageShape yShape = {{bs, n}, {bs, n}};

 
    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulFinalizeRouting",
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {wShape, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {scaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {pertoken_scaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
            {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{bs, n}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }


 TEST_F(GroupedMatmulFinalizeRouting, grouped_matmul_finalize_routing_91095_normal_1)
 {
     int m = 1024;
     int k = 2048;
     int n = 7168;
     int e = 16;
     int bs = 64;
     gert::StorageShape xShape = {{m, k}, {m, k}};
     gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
     gert::StorageShape scaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
     gert::StorageShape pertoken_scaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
     gert::StorageShape groupListShape = {{e}, {e}};
     gert::StorageShape shared_inputShape = {{bs, n}, {bs, n}};
     gert::StorageShape logitShape = {{m}, {m}};
     gert::StorageShape rowindexShape = {{m}, {m}};
     gert::StorageShape yShape = {{bs, n}, {bs, n}};


     gert::InfershapeContextPara infershapeContextPara(
         "GroupedMatmulFinalizeRouting",
         {{xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
          {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_FRACTAL_NZ},
          {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
          {pertoken_scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
          {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
          {shared_inputShape, ge::DT_BF16, ge::FORMAT_ND},
          {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
          {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}},
         {
             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         },
         {
             {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
             {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
             {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         });

     std::vector<std::vector<int64_t>> expectOuputShape = {{bs, n}}; // 预期输出shape
     ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }
