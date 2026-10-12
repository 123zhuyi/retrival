set(BISHENGIR_PRIVATE_LIBS
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  BiShengIRHIVMTransformOps
  BiShengIRHACCDialect
  BiShengIRHACCUtils
)

add_bishengir_dialect_library(BishengIRLLVMTransforms
  LLVMParameterPacking.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/LLVMIR/Transforms

  DEPENDS
  BishengIRLLVMTransformsIncGen
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRLLVMDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)
