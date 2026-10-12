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
 * \file bincount_def.cpp
 * \brief bincount def
 */

#include <vector>
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> arrayDType = {
    ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32,
    ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
    ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8,
    ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32,
    ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
    ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8,
    ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32,
    ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
    ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8,
    ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32,
    ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
    ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8,
    ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32,
    ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,  ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8};
static const std::vector<ge::DataType> sizeDType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
static const std::vector<ge::DataType> weightsDType = {
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,
    ge::DT_DOUBLE,  ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT16,
    ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,
    ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_BOOL,    ge::DT_BOOL,
    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_DOUBLE,
    ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT64,   ge::DT_INT64,
    ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,
    ge::DT_UINT8,   ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT16,   ge::DT_INT16,
    ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_UINT8,
    ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,
    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_DOUBLE,  ge::DT_DOUBLE,
    ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT16,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT64,   ge::DT_INT64,   ge::DT_INT64,
    ge::DT_INT64,   ge::DT_INT64,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,
    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL,    ge::DT_BOOL};
static const std::vector<ge::DataType> binsDType = {
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,
    ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT32,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE,
    ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE,
    ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE,
    ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE,
    ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE,
    ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE,
    ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE};
static const std::vector<ge::Format> allFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static bool IsDtypeConfigValid()
{
    return arrayDType.size() == sizeDType.size() && arrayDType.size() == weightsDType.size() &&
           arrayDType.size() == binsDType.size() && arrayDType.size() == allFormat.size();
}

class BincountDef : public OpDef {
public:
    explicit BincountDef(const char* name) : OpDef(name)
    {
        if (!IsDtypeConfigValid()) {
            return;
        }
        this->Input("array").ParamType(REQUIRED).DataType(arrayDType).Format(allFormat).UnknownShapeFormat(allFormat);
        this->Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(sizeDType)
            .Format(allFormat)
            .UnknownShapeFormat(allFormat);
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataType(weightsDType)
            .Format(allFormat)
            .UnknownShapeFormat(allFormat);
        this->Output("bins").ParamType(REQUIRED).DataType(binsDType).Format(allFormat).UnknownShapeFormat(allFormat);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "bincount");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

using Bincount = BincountDef;
OP_ADD(Bincount);
} // namespace ops
