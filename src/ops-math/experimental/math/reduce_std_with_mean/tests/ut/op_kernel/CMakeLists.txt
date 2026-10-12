# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------

# 自定义 op_kernel UT 构建，不使用 AddOpTestCase 宏：
# AddOpTestCase 会无条件触发 gen_tiling_head_file.sh（python import tbe），
# 依赖 CI 环境的 numpy/scipy 版本匹配；而本算子 kernel/test 仅需源目录
# op_kernel/reduce_std_with_mean_tiling_data.h 中的 TilingData 结构定义，
# 不依赖 gen_head 生成的 InitTilingData/GET_TILING_DATA_WITH_STRUCT，故直接
# 手写等价构建逻辑并跳过 gen_head 流程。

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # --soc 过滤：仅支持 ascend910b（与 AddOpTestCase 内逻辑一致）
    set(supported_socs "ascend910b")
    if(NOT ${ASCEND_COMPUTE_UNIT} IN_LIST supported_socs)
        return()
    endif()
    # --ops 过滤：ASCEND_OP_NAME 为空或包含本算子时才构建
    list(FIND ASCEND_OP_NAME reduce_std_with_mean INDEX)
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
    endif()

    set(socVersion "Ascend910B1")

    # 填充 fastOpTestSocVersions，使顶层 add_opkernel_ut_modules 为该 soc
    # 生成 cases 动态库和 math_op_kernel_ut_Ascend910B1 可执行文件
    # （与 AddOpTestCase 内逻辑一致）
    list(FIND fastOpTestSocVersions "${socVersion}" index)
    if(index EQUAL -1)
        set(fastOpTestSocVersions ${fastOpTestSocVersions} ${socVersion}
            CACHE STRING "fastOp Test SocVersions" FORCE)
    endif()

    # cases_obj: kernel + test（TilingData 结构来自 op_kernel/ 源目录头文件）
    add_library(reduce_std_with_mean_${socVersion}_cases_obj OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/reduce_std_with_mean.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_std_with_mean.cpp)
    target_compile_options(
        reduce_std_with_mean_${socVersion}_cases_obj
        PRIVATE -g -DDTYPE_SELF=half -DUT_SOC_VERSION="${socVersion}"
                -include ${CMAKE_CURRENT_SOURCE_DIR}/reduce_std_with_mean_ut_tiling_data.h)
    target_include_directories(
        reduce_std_with_mean_${socVersion}_cases_obj
        PRIVATE
            ${GTEST_INCLUDE}
            ${ASCEND_DIR}/include/base/context_builder
            ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
            ${PROJECT_SOURCE_DIR}/tests/ut/common
            ${OP_TILING_INCLUDE}
            ${dlog_MATH_INCLUDE_DIR}
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/)
    target_link_libraries(
        reduce_std_with_mean_${socVersion}_cases_obj
        PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                tikicpulib::${socVersion} GTest::gtest)

    # 聚合到 ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj（与 AddOpTestCase 相同的聚合目标）
    if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
        add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT
                    $<TARGET_OBJECTS:reduce_std_with_mean_${socVersion}_cases_obj>)
    else()
        target_sources(
            ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
            PRIVATE $<TARGET_OBJECTS:reduce_std_with_mean_${socVersion}_cases_obj>)
    endif()
    target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
        PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                $<TARGET_OBJECTS:reduce_std_with_mean_${socVersion}_cases_obj>)
endif()
