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

class TEST_VIEWCOPY_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "viewcopy test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "viewcopy test TearDown" << std::endl;
  }
};

TEST_F(TEST_VIEWCOPY_UT, viewcopy_infershape) {
 	gert::InfershapeContextPara infershapeContextPara(
    "ViewCopy",
 	  {
      {{{5, 3, 4, 1}, {5, 3, 4, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
 	    {{{3, 1, 1}, {3, 1, 1}}, ge::DT_UINT8, ge::FORMAT_ND}
 	  },
 	  {
 	    {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
 	  });
 	std::vector<std::vector<int64_t>> expectOutputShape = {{5, 3, 4, 1}};
 	ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TEST_VIEWCOPY_UT, viewcopy_size_equal) {
    gert::InfershapeContextPara infershapeContextPara(
    "ViewCopy",
 	  {
      {{{3, 1, 1}, {3, 1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
 	    {{{3, 1, 1}, {3, 1, 1}}, ge::DT_UINT8, ge::FORMAT_ND}
 	  },
 	  {
 	    {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
 	  });
 	std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1}};
 	ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

