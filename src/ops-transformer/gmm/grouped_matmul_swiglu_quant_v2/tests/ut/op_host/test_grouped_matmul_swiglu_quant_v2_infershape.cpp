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


 class GroupedMatmulSwigluQuantV2 : public testing::Test {
 protected:
     static void SetUpTestCase() {
         std::cout << "GroupedMatmulSwigluQuantV2 Proto Test SetUp" << std::endl;
     }

     static void TearDownTestCase() {
         std::cout << "GroupedMatmulSwigluQuantV2 Proto Test TearDown" << std::endl;
     }
 };

 TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_w8a8_normal_1) {
    int m = 1024;
    int k = 2048;
    int n = 4096;
    int e = 16;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

  TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_normal_1) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

 TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_illegal_1) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n}, {e, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

 TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_illegal_2) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

 TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_illegal_3) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_NDC1HWC0},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_illegal_4) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 1}, {e, k / 64, n, 1}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_NDC1HWC0},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

 TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_illegal_5) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 1}, {e, k / 64, n, 1}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_NDC1HWC0},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_illegal_6) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 1025;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_NDC1HWC0},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m, n /64 / 2 , 2}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }

 TEST_F(GroupedMatmulSwigluQuantV2, test_infershape_91095_perroken_normal_1) {
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::InfershapeContextPara infershapeContextPara("GroupedMatmulSwigluQuantV2",
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {{wShape}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{wScaleShape}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dequant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dequant_dtype", Ops::Transformer::AnyValue::CreateFrom<float>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOuputShape = {{m, n / 2}, {m}}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOuputShape);
 }