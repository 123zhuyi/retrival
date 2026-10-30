#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from op_test_frame.ut import OpUT
ut_case = OpUT("util_binary", "impl.util.util_binary")

# pylint: disable=unused-argument,import-outside-toplevel
def is_inf_test_util_binary(test_arg):
    """
    test for util_binary api
    """
    from impl.dynamic import register_binary_match
    register_binary_match("IsInf")

ut_case.add_cust_test_func("all", test_func=is_inf_test_util_binary)

if __name__ == '__main__':
    ut_case.run("Ascend310P3", "Ascend910B2")
    
