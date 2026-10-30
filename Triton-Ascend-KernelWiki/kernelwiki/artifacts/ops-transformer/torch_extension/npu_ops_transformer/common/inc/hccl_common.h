/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_common.h
 * \brief
 */

#include "aclnn_common.h"
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
    #include "torch_npu/csrc/flopcount/FlopCount.h"
#endif
#define NPU_NAME_SPACE at_npu::native

using _HcclKfcAllocOpArgs = HcclResult (*)(void **);                                        // 通信配置对象创建
using _HcclKfcOpArgsSetAlgConfig = HcclResult (*)(void *, char *);                          // 设置通信类型
using _HcclKfcOpArgsSetCommEngine = HcclResult (*)(void *, uint8_t);                        // 设置通信方式
using _HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void *, void **);             // 创建HcclContext
using _HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm, uint64_t, void **, uint64_t *); // 获取远端地址
using _HcclKfcFreeOpArgs = HcclResult (*)(void *);                                          // 释放通信配置对象
using _HcclCommGetHandleWithName = HcclResult (*)(const char *, HcclComm*);                 // 通过groupName获取groupHandle
using _HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t *);                              // 获取通信域大小
using _HcclGetRankId = HcclResult (*)(HcclComm, uint32_t *);                                // 获取卡号
using _HcclGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);                   // 获取本卡地址

static _HcclKfcAllocOpArgs HcclKfcAllocOpArgsFunc = nullptr;
static _HcclKfcOpArgsSetAlgConfig HcclKfcOpArgsSetAlgConfigFunc = nullptr;
static _HcclKfcOpArgsSetCommEngine HcclKfcOpArgsSetCommEngineFunc = nullptr;
static _HcclCreateOpResCtx HcclCreateOpResCtxFunc = nullptr;
static _HcclGetRemoteIpcHcclBuf HcclGetRemoteIpcHcclBufFunc = nullptr;
static _HcclKfcFreeOpArgs HcclKfcFreeOpArgsFunc = nullptr;
static _HcclCommGetHandleWithName HcclCommGetHandleWithNameFunc = nullptr;
static _HcclGetRankSize HcclGetRankSizeFunc = nullptr;
static _HcclGetRankId HcclGetRankIdFunc = nullptr;
static _HcclGetHcclBuffer HcclGetHcclBufferFunc = nullptr;

inline const char *GetHcclLibName(void)
{
    return "libhccl.so";
}

inline const char *GetHcclFwkLibName(void)
{
    return "libhccl_fwk.so";
}

template <typename T>
inline T GetFuncAddr(void * opApiHandler, const char *libName, const char *apiName)
{
    auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetHcclLibName(), apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s", apiName, GetHcclLibName(), dlerror());
        return nullptr;
    }
    T func = reinterpret_cast<T>(funcAddr);
    return func;
}

template <typename T>
inline T GetHcclFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclLibName(), apiName);
}

template <typename T>
inline T GetHcclFwkFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclFwkLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclFwkLibName(), apiName);
}

inline void InitHcclFunctions()
{
    HcclKfcAllocOpArgsFunc = GetHcclFuncAddr<_HcclKfcAllocOpArgs>("HcclKfcAllocOpArgs"); // 通信配置对象创建
    TORCH_CHECK(HcclKfcAllocOpArgsFunc != nullptr, "getHcclKfcAllocOpArgs failed.");
    HcclKfcFreeOpArgsFunc = GetHcclFuncAddr<_HcclKfcFreeOpArgs>("HcclKfcFreeOpArgs"); // 释放通信配置对象
    TORCH_CHECK(HcclKfcFreeOpArgsFunc != nullptr, "getHcclKfcFreeOpArgs failed.");
    HcclKfcOpArgsSetCommEngineFunc = GetHcclFuncAddr<_HcclKfcOpArgsSetCommEngine>("HcclKfcOpArgsSetCommEngine"); // 设置通信方式
    TORCH_CHECK(HcclKfcOpArgsSetCommEngineFunc != nullptr, "getHcclKfcOpArgsSetCommEngine failed.");
    HcclGetRankIdFunc = GetHcclFuncAddr<_HcclGetRankId>("HcclGetRankId"); // 获取本卡卡号
    TORCH_CHECK(HcclGetRankIdFunc != nullptr, "getFuncHcclGetRankId failed.");
    HcclGetHcclBufferFunc = GetHcclFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer"); // 获取本卡地址
    TORCH_CHECK(HcclGetHcclBufferFunc != nullptr, "getFuncHcclGetHcclBuffer failed.");
    HcclGetRemoteIpcHcclBufFunc = GetHcclFwkFuncAddr<_HcclGetRemoteIpcHcclBuf>("HcclGetRemoteIpcHcclBuf"); // 获取远端地址
    TORCH_CHECK(HcclGetRemoteIpcHcclBufFunc != nullptr, "getFuncHcclGetRemoteIpcHcclBuf failed.");
    HcclKfcOpArgsSetAlgConfigFunc = GetHcclFuncAddr<_HcclKfcOpArgsSetAlgConfig>("HcclKfcOpArgsSetAlgConfig");  // 设置通信类型
    TORCH_CHECK(HcclKfcOpArgsSetAlgConfigFunc != nullptr, "getFuncHcclKfcOpArgsSetAlgConfig failed.");
    HcclCommGetHandleWithNameFunc = GetHcclFwkFuncAddr<_HcclCommGetHandleWithName>("HcclCommGetHandleWithName"); // 通过groupName获取groupHandle
    TORCH_CHECK(HcclCommGetHandleWithNameFunc != nullptr, "getFuncHcclCommGetHandleWithName failed.");
    HcclCreateOpResCtxFunc = GetHcclFuncAddr<_HcclCreateOpResCtx>("HcclCreateOpResCtx");  // 创建HcclContext
    TORCH_CHECK(HcclCreateOpResCtxFunc != nullptr, "getFuncHcclCreateOpResCtx failed.");
    HcclGetRankSizeFunc = GetHcclFuncAddr<_HcclGetRankSize>("HcclGetRankSize"); // 获取通信域大小
    TORCH_CHECK(HcclGetRankSizeFunc != nullptr, "getFuncHcclGetRankSize failed.");
}