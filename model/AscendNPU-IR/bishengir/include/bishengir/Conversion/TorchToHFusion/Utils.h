//===- Utils.h - Utilities for Torch to HFusion Conversion ------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_CONVERSION_TORCHTOHFUSION_UTILS_H
#define BISHENGIR_CONVERSION_TORCHTOHFUSION_UTILS_H

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace mlir {
void getElementwiseResultShape(OpBuilder &b, Location loc,
                               const ValueRange tensorOperands,
                               SmallVector<Value> &resultShape);

Value convertScalarToTensor(OpBuilder &b, Location loc, Value scalar,
                            SmallVector<Value> &resultShape,
                            Type resultElementType);

FailureOr<Value> broadcastTensorToShape(PatternRewriter &rewriter, Location loc,
                                        Value input,
                                        RankedTensorType broadcastType,
                                        SmallVector<Value> dynDims = {});

FailureOr<Value> unsqueezeDims(PatternRewriter &rewriter, Location loc,
                               Value operand, SmallVector<int64_t> &dimensions);

FailureOr<Value> createHFusionCastOp(PatternRewriter &rewriter, Location loc,
                                     Type dtype, Value input);

FailureOr<Value> permuteTensor(Operation *op, PatternRewriter &rewriter,
                               Location loc, SmallVector<int64_t> dimensions,
                               Value input);

FailureOr<Value> squeezeDims(PatternRewriter &rewriter, Location loc,
                             Value operand, SmallVector<int64_t> &dimensions);

// When comparing two numbers, the data types need to be consistent. This
// function supports int/float inputs and returns which type to promote to. If
// one input is type int and one input is type float, the promoted type is
// float. If the two types are the same, return the one with a larger bit width.
FailureOr<Type> getPromotionType(Operation *op, Type lhsDtype, Type rhsDtype);

LogicalResult broadcastToGivenShape(Operation *op, PatternRewriter &rewriter,
                                    Value input,
                                    SmallVector<Value> broadcastToShape,
                                    RankedTensorType broadcastType,
                                    bool ensureNoImplicitBroadcast,
                                    Value &result,
                                    SmallVector<bool> useBroadcastToShape);
} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOHFUSION_UTILS_H
