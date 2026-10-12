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
 * \file ts_lightning_indexer_grad.h
 * \brief LightningIndexerGrad UTest 相关基类定义.
 */
#pragma once

#include "tests/utest/ts.h"
#include "lightning_indexer_grad_case.h"
#include "lightning_indexer_grad_param.h"

using LightningIndexerGradCase = ops::adv::tests::LIG::LightningIndexerGradCase;
using LightningIndexerGradParam = ops::adv::tests::LIG::LightningIndexerGradParam;
class Ts_LightningIndexerGrad : public Ts<LightningIndexerGradCase> {};
