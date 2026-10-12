//===- MemRefExtOps.cpp ---------------------------------------------------===//
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

#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

#if BSPUB_DAVINCI_BISHENGIR
#include "mlir/Dialect/Utils/ExpandShapeUtils.h"
#endif

using namespace mlir;
using namespace bishengir::memref_ext;

Operation *MemRefExtDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

//===----------------------------------------------------------------------===//
// AllocWorkspaceOp
//===----------------------------------------------------------------------===//

LogicalResult AllocWorkspaceOp::verify() {
  MemRefType type = getType();
  if (static_cast<int64_t>(this->getDynamicSize().size()) !=
      type.getNumDynamicDims())
    return this->emitOpError("dimension operand count does not equal memref "
                             "dynamic dimension count");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"
