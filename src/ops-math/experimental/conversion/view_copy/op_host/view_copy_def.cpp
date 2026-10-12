/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file view_copy_def.cpp
 * \brief ViewCopy operator definition.
 */

#include <initializer_list>

#include "register/op_def_registry.h"

namespace ops {
namespace {

static const std::initializer_list<ge::DataType> kDataDtypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8, ge::DT_INT16,
    ge::DT_UINT16,  ge::DT_INT32,  ge::DT_UINT32, ge::DT_BOOL,  ge::DT_INT64, ge::DT_FLOAT16,
    ge::DT_FLOAT,   ge::DT_BF16,   ge::DT_INT8,   ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
    ge::DT_INT32,   ge::DT_UINT32, ge::DT_BOOL,   ge::DT_INT64};

static const std::initializer_list<ge::DataType> kMetaDtypes = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::initializer_list<ge::Format> kNdFormats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

} // namespace

class ViewCopy : public OpDef {
public:
    explicit ViewCopy(const char* name) : OpDef(name)
    {
        this->Input("dst")
            .ParamType(REQUIRED)
            .DataType(kDataDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("dst_size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(kMetaDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("dst_stride")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(kMetaDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("dst_storage_offset")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(kMetaDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("src")
            .ParamType(REQUIRED)
            .DataType(kDataDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("src_size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(kMetaDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("src_stride")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(kMetaDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Input("src_storage_offset")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(kMetaDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType(kDataDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ViewCopy);
} // namespace ops
