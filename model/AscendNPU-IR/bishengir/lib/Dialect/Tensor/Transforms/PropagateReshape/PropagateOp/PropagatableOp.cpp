//===- PropagatableOp.cpp - Propagatable operation base implementation ----===//
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
//============================================================================//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateExpandUp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#define DEBUG_TYPE "propagatable-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir::tensor {
using namespace mlir::tensor::reshape_utils;
using namespace mlir::utils::debugger;
LogicalResult
PropagatableOp::matchAndRewriteExpand(PatternRewriter &rewriter, Operation *op,
                                      tensor::ExpandShapeOp expandOp) {
  return rewriter.notifyMatchFailure(op, "Expand up is not implemented");
}

LogicalResult
PropagatableOp::matchAndRewriteCollapse(PatternRewriter &rewriter,
                                        Operation *op,
                                        tensor::CollapseShapeOp collapseOp) {
  return rewriter.notifyMatchFailure(op, "Collpase down is not implemented");
}
} // namespace mlir::tensor
