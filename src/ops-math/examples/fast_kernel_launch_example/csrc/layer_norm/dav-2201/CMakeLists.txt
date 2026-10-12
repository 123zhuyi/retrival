ascend_ops_add_current_op(OP_TARGET)
target_link_libraries(${OP_TARGET} PUBLIC platform)
target_link_libraries(${OP_TARGET} PUBLIC tiling_api)
