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
 * \file fused_infer_attention_score_const.h
 * \brief
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_CONST_H_
#define FUSED_INFER_ATTENTION_SCORE_CONST_H_
const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type

};
constexpr uint32_t NUM_16 = 16;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

constexpr uint32_t DIM_BNSD_OR_BSND = 4;
constexpr uint32_t DIM_BSH = 3;
constexpr uint32_t DIM_TND = 3;

constexpr uint32_t NUM5 = 5;

constexpr uint32_t BSH_H_IDX = 2;
constexpr uint32_t BNSD_D_IDX = 3;

constexpr uint32_t MAX_BLOCK_SIZE = 512;
#endif
