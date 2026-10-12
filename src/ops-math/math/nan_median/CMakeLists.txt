# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
# ----------------------------------------------------------------------------

set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(OPTYPE nan_median ACLNNTYPE aclnn_exclude
    DEPENDENCIES kth_value sort
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)

# Keep graph_base loaded for the Host UT shared library under --as-needed linkers.
if(OP_HOST_UT AND TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj INTERFACE
        "-Wl,--no-as-needed,$<TARGET_FILE:graph_base>,--as-needed")
endif()
