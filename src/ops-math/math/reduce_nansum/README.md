# ReduceNansum

本目录仅包含ReduceNansum算子对应的aclnn接口；如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_reduce_nansum](./examples/test_aclnn_reduce_nansum.cpp) | 通过[aclnnReduceNansum](./docs/aclnnReduceNansum.md)接口方式调用ReduceNansum算子。 |
