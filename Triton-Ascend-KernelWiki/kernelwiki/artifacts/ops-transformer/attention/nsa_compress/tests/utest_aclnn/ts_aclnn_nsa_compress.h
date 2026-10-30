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
 * \file ts_nsa_compress.h
 * \brief NsaCompress UTest 相关基类定义.
 */

#pragma once

#include "tests/utest/ts.h"
#include "nsa_compress_case.h"
#include "aclnn_nsa_compress_case.h"

using AclnnNsaCompressParam = ops::adv::tests::NsaCompress::AclnnNsaCompressParam;
using LayoutType = ops::adv::tests::NsaCompress::NsaCompressParam::LayoutType;
using AclnnNsaCompressCase = ops::adv::tests::NsaCompress::AclnnNsaCompressCase;


class Ts_Aclnn_NsaCompress : public Ts<AclnnNsaCompressCase> {};
class Ts_Aclnn_NsaCompress_Ascend910B1 : public Ts_Ascend910B1<AclnnNsaCompressCase> {};
class Ts_Aclnn_NsaCompress_Ascend910B2 : public Ts_Ascend910B2<AclnnNsaCompressCase> {};
class Ts_Aclnn_NsaCompress_Ascend910B3 : public Ts_Ascend910B3<AclnnNsaCompressCase> {};


class Ts_Aclnn_NsaCompress_WithParam : public Ts_WithParam<AclnnNsaCompressCase> {};
class Ts_Aclnn_NsaCompress_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<AclnnNsaCompressCase> {};
class Ts_Aclnn_NsaCompress_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<AclnnNsaCompressCase> {};
class Ts_Aclnn_NsaCompress_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<AclnnNsaCompressCase> {};