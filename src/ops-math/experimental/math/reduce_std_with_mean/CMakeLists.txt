# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------

add_all_modules_sources(OPTYPE reduce_std_with_mean ACLNNTYPE aclnn DEPENDENCIES reduce_std_v2)

# Workaround: asc_opc imports "dynamic.ReduceStdWithMean" (PascalCase) but
# TBE generates "reduce_std_with_mean.py". Create a symlink.
# The target file doesn't need to exist yet for a symlink.
set(_DST_DIR ${CMAKE_BINARY_DIR}/tbe/dynamic)
file(MAKE_DIRECTORY ${_DST_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  reduce_std_with_mean.py
  ${_DST_DIR}/ReduceStdWithMean.py
  WORKING_DIRECTORY ${_DST_DIR}
  ERROR_QUIET
)
