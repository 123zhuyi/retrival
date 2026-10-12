/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fusion_pass_common.h
 * \brief
 */

#include "log/log.h"
#include "version/ge-compiler_version.h"

namespace {
#pragma once
#define GE_COMPILER_VERSION_910 90100000

extern "C" {
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

bool IsTargetVersion()
{
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version < GE_COMPILER_VERSION_910) {
        OP_LOGD("IsTargetVersion", "GE runtime version is %d < %d.", version, GE_COMPILER_VERSION_910);
        return false;
    }
    return true;
}
} // end namespace
