#/*
 # This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 # and is contributed to the CANN Open Software.
 #
 # Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 # All Rights Reserved.
 #
 # Authors (accounts):
 # - Tu Yuanhang <@TuYHAAAAAA>
 # - Su Tonghua <@sutonghua>
 #
 # This program is free software: you can redistribute it and/or modify it.
 # Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 # You may not use this file except in compliance with the License.
 # See the LICENSE file at the root of the repository for the full text of the License.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 #*/

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 需要将Tiling依赖的文件添加到CMakeLists.txt中
    # set(elewise_common_tiling_files
    #         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
    #         )
    # 算子自己的tiling文件路径
    set(mul_v2_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/mul_v2_tiling.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："ascend950PR_99;AscendB1"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    # AddOpTestCase(mul_v2 "ascend910B1" "" "${mul_v2_tiling_files}")
endif()
