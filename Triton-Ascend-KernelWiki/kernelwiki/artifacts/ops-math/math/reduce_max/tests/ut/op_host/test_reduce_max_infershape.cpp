/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class reduce_max : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "reduce_max SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "reduce_max TearDown" << std::endl;
  }
};

TEST_F(reduce_max, reduce_max_infer_shape_rt01) {
  std::vector<int32_t> axesValue = {1, 2};
 	gert::InfershapeContextPara infershapeContextPara(
    "ReduceMax",
 	  {
 	    {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
 	    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
 	  },
 	  {
 	    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
 	  },
 	  {
 	    gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))
 	  });
 	std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 1}};
 	ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(reduce_max, reduce_max_infer_shape_rt02) {
  std::vector<int32_t> axesValue = {1, 2};
 	gert::InfershapeContextPara infershapeContextPara(
    "ReduceMax",
 	  {
 	    {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
 	    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
 	  },
 	  {
 	    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
 	  },
 	  {
 	    gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))
 	  });
 	std::vector<std::vector<int64_t>> expectOutputShape = {{2,}};
 	ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}