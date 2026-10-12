# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    OPTYPE bernoulli_mask
    ACLNNTYPE aclnn_exclude
)

# A custom ACLNN package compiles most of its L0 dependency closure into
# libcust_opapi.so. Keep those implementation symbols local to the custom
# package so they cannot preempt identically named L0 symbols used by the
# system libopapi.so. ACLNN_API declarations retain default visibility, so the
# four scalar Bernoulli entry points remain exported. Do not change visibility
# for the built-in opapi target: other custom packages may depend on its L0 ABI.
if(ENABLE_CUSTOM AND TARGET ${OPHOST_NAME}_opapi_obj)
    # The scalar API keeps the upstream non-DAV2201 fallback for compatibility.
    # Compile that L0 wrapper into the custom package so libcust_opapi.so can be
    # loaded independently instead of relying on a feature library having
    # already populated the process-wide symbol scope.
    target_sources(
        ${OPHOST_NAME}_opapi_obj
        PRIVATE
            ${CMAKE_SOURCE_DIR}/random/stateless_bernoulli/op_api/stateless_bernoulli.cpp
    )
    target_compile_options(
        ${OPHOST_NAME}_opapi_obj
        PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden
    )
endif()
