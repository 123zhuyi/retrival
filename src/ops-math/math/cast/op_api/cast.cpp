/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/cast.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Cast);

static const std::initializer_list<std::pair<op::DataType, op::DataType>> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}, {op::DataType::DT_FLOAT16, op::DataType::DT_INT32},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16}, {op::DataType::DT_FLOAT, op::DataType::DT_INT32},
    {op::DataType::DT_INT32, op::DataType::DT_FLOAT16}, {op::DataType::DT_INT32, op::DataType::DT_FLOAT},
    {op::DataType::DT_INT32, op::DataType::DT_INT8},    {op::DataType::DT_INT32, op::DataType::DT_UINT8},
    {op::DataType::DT_INT32, op::DataType::DT_BOOL},    {op::DataType::DT_INT8, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT8, op::DataType::DT_FLOAT},    {op::DataType::DT_INT8, op::DataType::DT_INT32},
    {op::DataType::DT_UINT8, op::DataType::DT_FLOAT16}, {op::DataType::DT_UINT8, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_INT32},   {op::DataType::DT_BOOL, op::DataType::DT_FLOAT16},
    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT},    {op::DataType::DT_BOOL, op::DataType::DT_INT32},
    {op::DataType::DT_BOOL, op::DataType::DT_UINT8},    {op::DataType::DT_FLOAT16, op::DataType::DT_UINT8},
    {op::DataType::DT_FLOAT16, op::DataType::DT_INT8},  {op::DataType::DT_FLOAT16, op::DataType::DT_BOOL},
    {op::DataType::DT_INT64, op::DataType::DT_INT32},   {op::DataType::DT_INT32, op::DataType::DT_INT64},
    {op::DataType::DT_UINT1, op::DataType::DT_FLOAT16}, {op::DataType::DT_INT64, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT, op::DataType::DT_BOOL},    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_INT64},   {op::DataType::DT_INT64, op::DataType::DT_UINT8},
    {op::DataType::DT_BOOL, op::DataType::DT_INT64},    {op::DataType::DT_INT64, op::DataType::DT_BOOL},
    {op::DataType::DT_BOOL, op::DataType::DT_INT8},     {op::DataType::DT_INT8, op::DataType::DT_BOOL}};

static const std::initializer_list<std::pair<op::DataType, op::DataType>> ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST = {
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}, {op::DataType::DT_FLOAT16, op::DataType::DT_INT32},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16}, {op::DataType::DT_FLOAT, op::DataType::DT_INT32},
    {op::DataType::DT_INT32, op::DataType::DT_FLOAT16}, {op::DataType::DT_INT32, op::DataType::DT_FLOAT},
    {op::DataType::DT_INT32, op::DataType::DT_INT8},    {op::DataType::DT_INT32, op::DataType::DT_UINT8},
    {op::DataType::DT_INT32, op::DataType::DT_BOOL},    {op::DataType::DT_INT8, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT8, op::DataType::DT_FLOAT},    {op::DataType::DT_INT8, op::DataType::DT_INT32},
    {op::DataType::DT_UINT8, op::DataType::DT_FLOAT16}, {op::DataType::DT_UINT8, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_INT32},   {op::DataType::DT_BOOL, op::DataType::DT_FLOAT16},
    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT},    {op::DataType::DT_BOOL, op::DataType::DT_INT32},
    {op::DataType::DT_BOOL, op::DataType::DT_UINT8},    {op::DataType::DT_FLOAT16, op::DataType::DT_UINT8},
    {op::DataType::DT_INT64, op::DataType::DT_INT32},   {op::DataType::DT_INT32, op::DataType::DT_INT64},
    {op::DataType::DT_UINT1, op::DataType::DT_FLOAT16}, {op::DataType::DT_INT64, op::DataType::DT_FLOAT16}};

static const std::initializer_list<std::pair<op::DataType, op::DataType>> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}, {op::DataType::DT_FLOAT16, op::DataType::DT_INT32},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16}, {op::DataType::DT_FLOAT, op::DataType::DT_INT32},
    {op::DataType::DT_INT32, op::DataType::DT_FLOAT16}, {op::DataType::DT_INT32, op::DataType::DT_FLOAT},
    {op::DataType::DT_INT32, op::DataType::DT_INT8},    {op::DataType::DT_INT32, op::DataType::DT_UINT8},
    {op::DataType::DT_INT32, op::DataType::DT_BOOL},    {op::DataType::DT_INT8, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT8, op::DataType::DT_FLOAT},    {op::DataType::DT_INT8, op::DataType::DT_INT32},
    {op::DataType::DT_UINT8, op::DataType::DT_FLOAT16}, {op::DataType::DT_UINT8, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_INT32},   {op::DataType::DT_BOOL, op::DataType::DT_FLOAT16},
    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT},    {op::DataType::DT_BOOL, op::DataType::DT_INT32},
    {op::DataType::DT_BOOL, op::DataType::DT_UINT8},    {op::DataType::DT_FLOAT16, op::DataType::DT_UINT8},
    {op::DataType::DT_INT64, op::DataType::DT_FLOAT},   {op::DataType::DT_FLOAT, op::DataType::DT_INT64},
    {op::DataType::DT_INT64, op::DataType::DT_INT32},   {op::DataType::DT_INT32, op::DataType::DT_INT64},
    {op::DataType::DT_FLOAT, op::DataType::DT_BF16},    {op::DataType::DT_BF16, op::DataType::DT_FLOAT},
    {op::DataType::DT_BF16, op::DataType::DT_INT32},    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_BF16},  {op::DataType::DT_FLOAT16, op::DataType::DT_INT8},
    {op::DataType::DT_FLOAT16, op::DataType::DT_BOOL},  {op::DataType::DT_UINT1, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT64, op::DataType::DT_FLOAT16}, {op::DataType::DT_FLOAT, op::DataType::DT_BOOL},
    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT},    {op::DataType::DT_UINT8, op::DataType::DT_INT64},
    {op::DataType::DT_INT64, op::DataType::DT_UINT8},   {op::DataType::DT_BOOL, op::DataType::DT_INT64},
    {op::DataType::DT_INT64, op::DataType::DT_BOOL},    {op::DataType::DT_INT16, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_INT16}, {op::DataType::DT_BF16, op::DataType::DT_BOOL},
    {op::DataType::DT_BF16, op::DataType::DT_INT8},     {op::DataType::DT_BF16, op::DataType::DT_UINT8},
    {op::DataType::DT_BOOL, op::DataType::DT_BF16},     {op::DataType::DT_INT32, op::DataType::DT_BF16},
    {op::DataType::DT_INT64, op::DataType::DT_BF16},    {op::DataType::DT_INT8, op::DataType::DT_BF16},
    {op::DataType::DT_UINT8, op::DataType::DT_BF16},    {op::DataType::DT_FLOAT, op::DataType::DT_UINT8},
    {op::DataType::DT_BOOL, op::DataType::DT_INT8},     {op::DataType::DT_INT8, op::DataType::DT_BOOL},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT16},   {op::DataType::DT_INT16, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT8},    {op::DataType::DT_INT64, op::DataType::DT_UINT32},
    {op::DataType::DT_INT64, op::DataType::DT_INT16},   {op::DataType::DT_INT64, op::DataType::DT_UINT16},
    {op::DataType::DT_INT64, op::DataType::DT_INT8},    {op::DataType::DT_INT32, op::DataType::DT_UINT32},
    {op::DataType::DT_INT32, op::DataType::DT_INT16},   {op::DataType::DT_INT32, op::DataType::DT_UINT16},
    {op::DataType::DT_UINT32, op::DataType::DT_INT64},  {op::DataType::DT_UINT32, op::DataType::DT_INT32},
    {op::DataType::DT_UINT32, op::DataType::DT_INT16},  {op::DataType::DT_UINT32, op::DataType::DT_UINT16},
    {op::DataType::DT_UINT32, op::DataType::DT_INT8},   {op::DataType::DT_UINT32, op::DataType::DT_UINT8},
    {op::DataType::DT_INT16, op::DataType::DT_INT64},   {op::DataType::DT_INT16, op::DataType::DT_INT32},
    {op::DataType::DT_INT16, op::DataType::DT_UINT32},  {op::DataType::DT_INT16, op::DataType::DT_UINT16},
    {op::DataType::DT_INT16, op::DataType::DT_INT8},    {op::DataType::DT_INT16, op::DataType::DT_UINT8},
    {op::DataType::DT_UINT16, op::DataType::DT_INT64},  {op::DataType::DT_UINT16, op::DataType::DT_INT32},
    {op::DataType::DT_UINT16, op::DataType::DT_UINT32}, {op::DataType::DT_UINT16, op::DataType::DT_INT16},
    {op::DataType::DT_UINT16, op::DataType::DT_INT8},   {op::DataType::DT_UINT16, op::DataType::DT_UINT8},
    {op::DataType::DT_INT8, op::DataType::DT_INT64},    {op::DataType::DT_INT8, op::DataType::DT_UINT32},
    {op::DataType::DT_INT8, op::DataType::DT_INT16},    {op::DataType::DT_INT8, op::DataType::DT_UINT16},
    {op::DataType::DT_INT8, op::DataType::DT_UINT8},    {op::DataType::DT_UINT8, op::DataType::DT_UINT32},
    {op::DataType::DT_UINT8, op::DataType::DT_INT16},   {op::DataType::DT_UINT8, op::DataType::DT_UINT16},
    {op::DataType::DT_UINT8, op::DataType::DT_INT8},    {op::DataType::DT_FLOAT16, op::DataType::DT_INT64},
    {op::DataType::DT_BF16, op::DataType::DT_INT64}};

static const std::initializer_list<std::pair<op::DataType, op::DataType>> REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT16, op::DataType::DT_INT32},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT32},
    {op::DataType::DT_INT32, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT32, op::DataType::DT_FLOAT},
    {op::DataType::DT_INT32, op::DataType::DT_INT8},
    {op::DataType::DT_INT32, op::DataType::DT_UINT8},
    {op::DataType::DT_INT32, op::DataType::DT_BOOL},
    {op::DataType::DT_INT8, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT8, op::DataType::DT_FLOAT},
    {op::DataType::DT_INT8, op::DataType::DT_INT32},
    {op::DataType::DT_UINT8, op::DataType::DT_FLOAT16},
    {op::DataType::DT_UINT8, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_INT32},
    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT16},
    {op::DataType::DT_BOOL, op::DataType::DT_FLOAT},
    {op::DataType::DT_BOOL, op::DataType::DT_INT32},
    {op::DataType::DT_BOOL, op::DataType::DT_UINT8},
    {op::DataType::DT_FLOAT16, op::DataType::DT_UINT8},
    {op::DataType::DT_INT64, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT64},
    {op::DataType::DT_INT64, op::DataType::DT_INT32},
    {op::DataType::DT_INT32, op::DataType::DT_INT64},
    {op::DataType::DT_FLOAT, op::DataType::DT_BF16},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT},
    {op::DataType::DT_BF16, op::DataType::DT_INT32},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_BF16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_INT8},
    {op::DataType::DT_FLOAT16, op::DataType::DT_BOOL},
    {op::DataType::DT_UINT1, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT64, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT, op::DataType::DT_BOOL},
    {op::DataType::DT_INT16, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_INT64},
    {op::DataType::DT_INT64, op::DataType::DT_UINT8},
    {op::DataType::DT_BOOL, op::DataType::DT_INT64},
    {op::DataType::DT_INT64, op::DataType::DT_BOOL},
    {op::DataType::DT_INT16, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_INT16},
    {op::DataType::DT_BF16, op::DataType::DT_BOOL},
    {op::DataType::DT_BF16, op::DataType::DT_INT8},
    {op::DataType::DT_BF16, op::DataType::DT_UINT8},
    {op::DataType::DT_BOOL, op::DataType::DT_BF16},
    {op::DataType::DT_INT32, op::DataType::DT_BF16},
    {op::DataType::DT_INT64, op::DataType::DT_BF16},
    {op::DataType::DT_INT8, op::DataType::DT_BF16},
    {op::DataType::DT_UINT8, op::DataType::DT_BF16},
    {op::DataType::DT_FLOAT, op::DataType::DT_UINT8},
    {op::DataType::DT_BOOL, op::DataType::DT_INT8},
    {op::DataType::DT_INT8, op::DataType::DT_BOOL},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT8_E4M3FN},
    {op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT8_E5M2},
    {op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT, op::DataType::DT_HIFLOAT8},
    {op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT16, op::DataType::DT_HIFLOAT8},
    {op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT16},
    {op::DataType::DT_BF16, op::DataType::DT_HIFLOAT8},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT8_E4M3FN},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT8_E5M2},
    {op::DataType::DT_BF16, op::DataType::DT_COMPLEX64},
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT8_E4M3FN},
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT8_E5M2},
    {op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX32},
    {op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64},
    {op::DataType::DT_HIFLOAT8, op::DataType::DT_BF16},
    {op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_BF16},
    {op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_BF16},
    {op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT16},
    {op::DataType::DT_INT8, op::DataType::DT_INT64},
    {op::DataType::DT_INT8, op::DataType::DT_UINT32},
    {op::DataType::DT_INT8, op::DataType::DT_INT16},
    {op::DataType::DT_INT8, op::DataType::DT_UINT16},
    {op::DataType::DT_INT8, op::DataType::DT_UINT8},
    {op::DataType::DT_INT16, op::DataType::DT_INT64},
    {op::DataType::DT_INT16, op::DataType::DT_INT32},
    {op::DataType::DT_INT16, op::DataType::DT_UINT32},
    {op::DataType::DT_INT16, op::DataType::DT_UINT16},
    {op::DataType::DT_INT16, op::DataType::DT_INT8},
    {op::DataType::DT_INT16, op::DataType::DT_UINT8},
    {op::DataType::DT_INT32, op::DataType::DT_UINT32},
    {op::DataType::DT_INT32, op::DataType::DT_INT16},
    {op::DataType::DT_INT32, op::DataType::DT_UINT16},
    {op::DataType::DT_INT64, op::DataType::DT_UINT32},
    {op::DataType::DT_INT64, op::DataType::DT_INT16},
    {op::DataType::DT_INT64, op::DataType::DT_UINT16},
    {op::DataType::DT_INT64, op::DataType::DT_INT8},
    {op::DataType::DT_UINT8, op::DataType::DT_UINT32},
    {op::DataType::DT_UINT8, op::DataType::DT_INT16},
    {op::DataType::DT_UINT8, op::DataType::DT_UINT16},
    {op::DataType::DT_UINT8, op::DataType::DT_INT8},
    {op::DataType::DT_UINT16, op::DataType::DT_INT64},
    {op::DataType::DT_UINT16, op::DataType::DT_INT32},
    {op::DataType::DT_UINT16, op::DataType::DT_UINT32},
    {op::DataType::DT_UINT16, op::DataType::DT_INT16},
    {op::DataType::DT_UINT16, op::DataType::DT_INT8},
    {op::DataType::DT_UINT16, op::DataType::DT_UINT8},
    {op::DataType::DT_UINT32, op::DataType::DT_INT64},
    {op::DataType::DT_UINT32, op::DataType::DT_INT32},
    {op::DataType::DT_UINT32, op::DataType::DT_INT16},
    {op::DataType::DT_UINT32, op::DataType::DT_UINT16},
    {op::DataType::DT_UINT32, op::DataType::DT_INT8},
    {op::DataType::DT_UINT32, op::DataType::DT_UINT8},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT8},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT16},
    {op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_HIFLOAT8},
    {op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT8_E5M2},
    {op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT8_E4M3FN},
    {op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_BF16},
    {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_HIFLOAT8},
    {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT8_E5M2},
    {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT8_E4M3FN},
    {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_BF16},
    {op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT4_E2M1},
    {op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E2M1},
    {op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT4_E2M1},
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT4_E2M1},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT4_E2M1},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT4_E2M1},
    {op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT4_E1M2},
    {op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E1M2},
    {op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT4_E1M2},
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT4_E1M2},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT4_E1M2},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT4_E1M2},
    {op::DataType::DT_COMPLEX64, op::DataType::DT_BF16},
    {op::DataType::DT_COMPLEX64, op::DataType::DT_FLOAT},
    {op::DataType::DT_COMPLEX32, op::DataType::DT_FLOAT16},
    {op::DataType::DT_BF16, op::DataType::DT_INT64},
    {op::DataType::DT_BF16, op::DataType::DT_INT16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_INT64},
    {op::DataType::DT_INT16, op::DataType::DT_BF16},
    {op::DataType::DT_BOOL, op::DataType::DT_INT16},
    {op::DataType::DT_INT16, op::DataType::DT_BOOL},
    {op::DataType::DT_UINT8, op::DataType::DT_BOOL},
    {op::DataType::DT_UINT16, op::DataType::DT_BOOL},
    {op::DataType::DT_UINT32, op::DataType::DT_BOOL},
    {op::DataType::DT_UINT16, op::DataType::DT_FLOAT16},
    {op::DataType::DT_UINT16, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT16, op::DataType::DT_BF16},
    {op::DataType::DT_UINT32, op::DataType::DT_FLOAT16},
    {op::DataType::DT_UINT32, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT32, op::DataType::DT_BF16},
    {op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX32},
    {op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64},
    {op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT},
    {op::DataType::DT_DOUBLE, op::DataType::DT_BF16},
    {op::DataType::DT_INT64, op::DataType::DT_DOUBLE},
    {op::DataType::DT_INT32, op::DataType::DT_INT4},
    {op::DataType::DT_DOUBLE, op::DataType::DT_INT32},
    {op::DataType::DT_DOUBLE, op::DataType::DT_INT64}};

static const std::initializer_list<std::pair<op::DataType, op::DataType>> ASCEND310P_SUPPORT_910_NOT_SUPPORT_LIST = {
    {op::DataType::DT_FLOAT, op::DataType::DT_INT16},
    {op::DataType::DT_FLOAT, op::DataType::DT_INT8},
    {op::DataType::DT_INT16, op::DataType::DT_FLOAT},
};

static const std::initializer_list<op::DataType> AICPU_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,      op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_INT16,     op::DataType::DT_UINT16,     op::DataType::DT_UINT8,   op::DataType::DT_INT32,
    op::DataType::DT_INT64,     op::DataType::DT_UINT32,     op::DataType::DT_UINT64,  op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self, op::DataType dstDtype)
{
    // Cast只需要根据soc信息判断对应芯片的dtype支持
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool isAscend310pSupport =
        std::find(
            ASCEND310P_SUPPORT_910_NOT_SUPPORT_LIST.begin(), ASCEND310P_SUPPORT_910_NOT_SUPPORT_LIST.end(),
            std::pair<op::DataType, op::DataType>(self->GetDataType(), dstDtype)) !=
        ASCEND310P_SUPPORT_910_NOT_SUPPORT_LIST.end();
    if (curArch == NpuArch::DAV_2002 && isAscend310pSupport) {
        return true;
    }

    bool isAscend610liteSupport =
        std::find(
            ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST.begin(), ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST.end(),
            std::pair<op::DataType, op::DataType>(self->GetDataType(), dstDtype)) !=
        ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST.end();
    if (curArch == NpuArch::DAV_3102) {
        return isAscend610liteSupport;
    }

    if (curArch == NpuArch::DAV_2201) {
        return std::find(
                   ASCEND910B_AICORE_DTYPE_SUPPORT_LIST.begin(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST.end(),
                   std::pair<op::DataType, op::DataType>(self->GetDataType(), dstDtype)) !=
               ASCEND910B_AICORE_DTYPE_SUPPORT_LIST.end();
    }

    if (IsRegBase(curArch)) {
        return std::find(
                   REGBASE_AICORE_DTYPE_SUPPORT_LIST.begin(), REGBASE_AICORE_DTYPE_SUPPORT_LIST.end(),
                   std::pair<op::DataType, op::DataType>(self->GetDataType(), dstDtype)) !=
               REGBASE_AICORE_DTYPE_SUPPORT_LIST.end();
    }

    return std::find(
               ASCEND910_AICORE_DTYPE_SUPPORT_LIST.begin(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST.end(),
               std::pair<op::DataType, op::DataType>(self->GetDataType(), dstDtype)) !=
           ASCEND910_AICORE_DTYPE_SUPPORT_LIST.end();
}

static bool IsAiCpuSupport(const aclTensor* self, op::DataType dstDtype)
{
    if (std::find(AICPU_DTYPE_SUPPORT_LIST.begin(), AICPU_DTYPE_SUPPORT_LIST.end(), self->GetDataType()) ==
        AICPU_DTYPE_SUPPORT_LIST.end()) {
        return false;
    }
    return std::find(AICPU_DTYPE_SUPPORT_LIST.begin(), AICPU_DTYPE_SUPPORT_LIST.end(), dstDtype) !=
           AICPU_DTYPE_SUPPORT_LIST.end();
}

// AICORE算子kernel
static const aclTensor* CastAiCore(
    const aclTensor* self, op::DataType dstDtype, const aclTensor* castOut, aclOpExecutor* executor)
{
    L0_DFX(CastAiCore, self, dstDtype, castOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Cast算子加入任务队列
    // Cast是算子的OpType，self是算子的输入，castOut是算子的输出，dstDtype是算子的属性
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Cast, OP_INPUT(self), OP_OUTPUT(castOut), OP_ATTR(dstDtype));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return castOut;
}

// AICPU算子kernel
static const aclTensor* CastAiCpu(
    const aclTensor* self, op::DataType dstDtype, aclTensor* castOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Cast算子加入任务队列
    // Cast是算子的OpType，self是算子的输入，castOut是算子的输出，dstDtype是算子的属性
    L0_DFX(CastAiCpu, self, dstDtype, castOut);

    auto ret = ACL_SUCCESS;
    if (IsComplexType(self->GetDataType())) {
        static internal::AicpuTaskSpace space("Cast", ge::DEPEND_IN_SHAPE, true);
        ret = ADD_TO_LAUNCHER_LIST_AICPU(
            Cast, OP_ATTR_NAMES({"SrcT", "DstT", "Truncate"}), OP_INPUT(self), OP_OUTPUT(castOut),
            OP_ATTR(self->GetDataType(), dstDtype, false));
        OP_CHECK(
            ret == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CastAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed in ComplexType."),
            return nullptr);
        return castOut;
    }
    static internal::AicpuTaskSpace space("Cast");
    ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Cast, OP_ATTR_NAMES({"dst_type"}), OP_INPUT(self), OP_OUTPUT(castOut), OP_ATTR(dstDtype));

    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CastAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return castOut;
}

const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor)
{
    if (self->GetDataType() == dstDtype) {
        return self;
    }
    auto outTensor = executor->AllocTensor(self->GetViewShape(), dstDtype, self->GetStorageFormat());
    CHECK_RET(outTensor != nullptr, nullptr);
    if (IsAiCoreSupport(self, dstDtype)) {
        return CastAiCore(self, dstDtype, outTensor, executor);
    } else {
        CHECK_RET(IsAiCpuSupport(self, dstDtype), nullptr);
        return CastAiCpu(self, dstDtype, outTensor, executor);
    }
    return outTensor;
}

// 注意：该L0接口只针对卷积使用，其他接口不要用
const aclTensor* CastOnlyForConvBackward(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor)
{
    if (self->GetDataType() == dstDtype) {
        return self;
    }
    OP_LOGD("Entering L0 CastOnlyForConvBackward");

    auto outTensor = executor->AllocTensor(
        self->GetStorageShape(), self->GetViewShape(), dstDtype, self->GetStorageFormat(), self->GetViewFormat());
    CHECK_RET(outTensor != nullptr, nullptr);
    if (IsAiCoreSupport(self, dstDtype)) {
        return CastAiCore(self, dstDtype, outTensor, executor);
    } else {
        return CastAiCpu(self, dstDtype, outTensor, executor);
    }
    return outTensor;
}

} // namespace l0op
