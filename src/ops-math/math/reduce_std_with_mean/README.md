# ReduceStdWithMean

本目录仅包含ReduceStdWithMean算子对应的aclnn接口；如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_batch_norm_stats](./examples/test_aclnn_batch_norm_stats.cpp) | 通过[aclnnBatchNormStats](./docs/aclnnBatchNormStats.md)接口方式调用ReduceStdWithMean算子。 |
| aclnn调用 | [test_aclnn_std_mean_correction](./examples/test_aclnn_std_mean_correction.cpp) | 通过[aclnnStdMeanCorrection](./docs/aclnnStdMeanCorrection.md)接口方式调用ReduceStdWithMean算子。 |
