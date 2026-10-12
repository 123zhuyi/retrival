/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"

using namespace ge;
namespace domi {
static Status ParseParamsSequenceConstruct(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int n = node->input_size();
    op_dest.DynamicInputRegister("inputs", n);
    op_dest.SetAttr("original_type", "ai.onnx::11::SequenceConstruct");
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

// register SequenceConstruct op info to GE
REGISTER_CUSTOM_OP("SequenceConstruct")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SequenceConstruct"), ge::AscendString("ai.onnx::12::SequenceConstruct"),
         ge::AscendString("ai.onnx::13::SequenceConstruct"), ge::AscendString("ai.onnx::14::SequenceConstruct"),
         ge::AscendString("ai.onnx::15::SequenceConstruct"), ge::AscendString("ai.onnx::16::SequenceConstruct"),
         ge::AscendString("ai.onnx::17::SequenceConstruct"), ge::AscendString("ai.onnx::18::SequenceConstruct")})
    .ParseParamsFn(ParseParamsSequenceConstruct)
    .ImplyType(ImplyType::AI_CPU);

static Status ParseParamsSequenceInsert(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::SequenceInsert");
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

// register SequenceInsert op info to GE
REGISTER_CUSTOM_OP("SequenceInsert")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SequenceInsert"), ge::AscendString("ai.onnx::12::SequenceInsert"),
         ge::AscendString("ai.onnx::13::SequenceInsert"), ge::AscendString("ai.onnx::14::SequenceInsert"),
         ge::AscendString("ai.onnx::15::SequenceInsert"), ge::AscendString("ai.onnx::16::SequenceInsert"),
         ge::AscendString("ai.onnx::17::SequenceInsert"), ge::AscendString("ai.onnx::18::SequenceInsert")})
    .ParseParamsFn(ParseParamsSequenceInsert)
    .ImplyType(ImplyType::AI_CPU);

static Status ParseParamsSequenceErase(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::SequenceErase");
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

// register SequenceErase op info to GE
REGISTER_CUSTOM_OP("SequenceErase")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SequenceErase"), ge::AscendString("ai.onnx::12::SequenceErase"),
         ge::AscendString("ai.onnx::13::SequenceErase"), ge::AscendString("ai.onnx::14::SequenceErase"),
         ge::AscendString("ai.onnx::15::SequenceErase"), ge::AscendString("ai.onnx::16::SequenceErase"),
         ge::AscendString("ai.onnx::17::SequenceErase"), ge::AscendString("ai.onnx::18::SequenceErase")})
    .ParseParamsFn(ParseParamsSequenceErase)
    .ImplyType(ImplyType::AI_CPU);

static Status ParseParamsSequenceEmpty(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("SequenceEmpty", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::SequenceEmpty");

    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

REGISTER_CUSTOM_OP("SequenceEmpty")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SequenceEmpty"), ge::AscendString("ai.onnx::12::SequenceEmpty"),
         ge::AscendString("ai.onnx::13::SequenceEmpty"), ge::AscendString("ai.onnx::14::SequenceEmpty"),
         ge::AscendString("ai.onnx::15::SequenceEmpty"), ge::AscendString("ai.onnx::16::SequenceEmpty"),
         ge::AscendString("ai.onnx::17::SequenceEmpty"), ge::AscendString("ai.onnx::18::SequenceEmpty")})
    .ParseParamsFn(ParseParamsSequenceEmpty)
    .ImplyType(ImplyType::AI_CPU);

static Status ParseParamsSequenceAt(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::SequenceAt");
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

// register SequenceAt op info to GE
REGISTER_CUSTOM_OP("SequenceAt")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SequenceAt"), ge::AscendString("ai.onnx::12::SequenceAt"),
         ge::AscendString("ai.onnx::13::SequenceAt"), ge::AscendString("ai.onnx::14::SequenceAt"),
         ge::AscendString("ai.onnx::15::SequenceAt"), ge::AscendString("ai.onnx::16::SequenceAt"),
         ge::AscendString("ai.onnx::17::SequenceAt"), ge::AscendString("ai.onnx::18::SequenceAt")})
    .ParseParamsFn(ParseParamsSequenceAt)
    .ImplyType(ImplyType::AI_CPU);

static Status ParseParamsSequenceLength(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        ge::AscendString op_name;
        (void)op_dest.GetName(op_name);
        OP_LOGE(op_name.GetString(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::SequenceLength");
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

// register SequenceLength op info to GE
REGISTER_CUSTOM_OP("SequenceLength")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SequenceLength"), ge::AscendString("ai.onnx::12::SequenceLength"),
         ge::AscendString("ai.onnx::13::SequenceLength"), ge::AscendString("ai.onnx::14::SequenceLength"),
         ge::AscendString("ai.onnx::15::SequenceLength"), ge::AscendString("ai.onnx::16::SequenceLength"),
         ge::AscendString("ai.onnx::17::SequenceLength"), ge::AscendString("ai.onnx::18::SequenceLength")})
    .ParseParamsFn(ParseParamsSequenceLength)
    .ImplyType(ImplyType::AI_CPU);
} // namespace domi
