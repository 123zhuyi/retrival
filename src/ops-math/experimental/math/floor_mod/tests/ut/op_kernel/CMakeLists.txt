# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的tiling文件路径
    set(floor_mod_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/floor_mod_tiling.cpp
    )

    # 使用AddOpTestCase
    # param1：算子名称
    # param2：soc版本
    # param3：自定义编译选项，这里不需要特殊的宏定义来选择代码路径，
    #        但如果代码中有 #ifdef 依赖 DTYPE_X 等宏，需要在这里传入。
    #        floor_mod kernel 是模板，在 cpp 测试文件中直接实例化，这里留空或根据需要添加。
    # param4：该算子依赖的所有tiling源码文件
    
    AddOpTestCase(floor_mod "ascend910B1" "" "${floor_mod_tiling_files}")
endif()