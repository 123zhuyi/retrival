add_bishengir_dialect_library(BiShengIRSymbolTransforms
  EraseSymbol.cpp
  PropagateSymbol.cpp
  SymbolAndEncodingConverter.cpp
  UnfoldSymbolicInt.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Symbol/Transforms

  DEPENDS
  BiShengIRSymbolTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRSymbolDialect
  BiShengIRSymbolUtils
  MLIRDialect
  MLIRFuncDialect
  BiShengIRHFusionDialect
  BiShengIRHFusionUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
)
