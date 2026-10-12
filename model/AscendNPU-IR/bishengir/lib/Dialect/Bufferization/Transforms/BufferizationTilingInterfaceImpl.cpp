//===- BufferizationTilingInterface.cpp - Tiling Interface models *-C++--*-===//
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

#include "bishengir/Dialect/Bufferization/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

struct ToTensorOpTiling
    : public TilingInterface::ExternalModel<ToTensorOpTiling, ToTensorOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto toTensorOp = cast<ToTensorOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        toTensorOp.getType().getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto toTensorOp = cast<ToTensorOp>(op);
    TensorType dstType = toTensorOp.getType();
    int64_t rank = dstType.getRank();
    SmallVector<Range> loopBounds(rank);
    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value src = toTensorOp.getMemref();
    for (auto dim : llvm::seq<int64_t>(0, rank)) {
      loopBounds[dim].offset = zero;
      loopBounds[dim].size = utils::getDimOFR(b, loc, src, dim);
      loopBounds[dim].stride = one;
    }
    return loopBounds;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto toTensorOp = cast<ToTensorOp>(op);
    TensorType dstType = toTensorOp.getType();
    int64_t rank = dstType.getRank();
    auto srcVal = toTensorOp.getMemref();
    // Create subview to tile the source memref
    SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));
    auto subviewOp = b.create<memref::SubViewOp>(op->getLoc(), srcVal, offsets,
                                                 sizes, strides);
    // Create a tiled `to_tensor` op
    [[maybe_unused]] SmallVector<Value> dynamicSizes;
    SmallVector<int64_t> staticSize;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSize);
    auto tiledOp = b.create<ToTensorOp>(
        op->getLoc(),
        RankedTensorType::get(staticSize, dstType.getElementType()),
        /*memref=*/subviewOp, toTensorOp.getRestrictAttr(),
        toTensorOp.getWritableAttr());
    return TilingResult{{tiledOp}, {tiledOp->getResult(0)}};
  }

  LogicalResult getResultTilePosition(
      Operation * /*op*/, OpBuilder & /*b*/, unsigned /*resultNumber*/,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation * /*op*/, OpBuilder & /*b*/, unsigned /*resultNumber*/,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(
      Operation *op, OpBuilder &b, unsigned /*resultNumber*/,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
};

} // namespace

void bishengir::bufferization::registerTilingInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx,
          mlir::bufferization::BufferizationDialect * /*dialect*/) {
        mlir::bufferization::ToTensorOp::attachInterface<ToTensorOpTiling>(
            *ctx);
      });
}
