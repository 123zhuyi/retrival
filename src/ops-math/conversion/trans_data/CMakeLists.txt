# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend950")
# 设置每种芯片类型对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(OPTYPE trans_data ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE HOSTCPU TRUE)

# format_transfer 辅助源文件不匹配 *_aicpu.cpp glob，需显式追加到已创建的 OBJECT 目标
# DISABLE_AICPU 时 add_all_modules_sources 不会创建 AICPU 目标，无需追加
if(NOT DISABLE_AICPU)
  file(GLOB TRANS_DATA_FT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/format_transfer/*.cpp)
  if(TRANS_DATA_FT_SRCS)
    if(TARGET ${OPHOST_NAME}_aicpu_obj)
      target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${TRANS_DATA_FT_SRCS})
    endif()
    if(TARGET trans_data_obj)
      target_sources(trans_data_obj PRIVATE ${TRANS_DATA_FT_SRCS})
    endif()
    if(TARGET trans_data_cases_obj)
      target_sources(trans_data_cases_obj PRIVATE ${TRANS_DATA_FT_SRCS})
    endif()
    # 常量折叠（HOSTCPU）目标：format_transfer 也需追加到 host 侧 x86 OBJECT
    if(TARGET trans_data_host_const_obj)
      target_sources(trans_data_host_const_obj PRIVATE ${TRANS_DATA_FT_SRCS})
    endif()
  endif()
endif()
