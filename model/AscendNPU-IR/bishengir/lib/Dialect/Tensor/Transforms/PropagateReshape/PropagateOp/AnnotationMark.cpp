//===- AnnotationMark.cpp - Annotation mark propagation -------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

namespace mlir::tensor {

LogicalResult PropagatableAnnotationMark::matchAndRewriteCollapse(
    PatternRewriter &rewriter, Operation *op,
    tensor::CollapseShapeOp collapseOp) {
  // MarkOp has zero results; it annotates $src in place. Point the mark at
  // the pre-collapse value (and expand any other matching-rank operands)
  // rather than leaving expand(collapse(x)) chains behind.
  auto markOp = cast<annotation::MarkOp>(op);
  const int64_t resultRank =
      cast<ShapedType>(markOp.getSrc().getType()).getRank();
  SmallVector<Value> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (operand == collapseOp.getResult()) {
      operands.push_back(collapseOp.getSrc());
      continue;
    }
    operands.push_back(
        tryExpandOperand(collapseOp, rewriter, operand, resultRank));
  }
  rewriter.modifyOpInPlace(op, [&] { markOp->setOperands(operands); });
  if (collapseOp.getResult().use_empty())
    rewriter.eraseOp(collapseOp);
  return success();
}

} // namespace mlir::tensor
