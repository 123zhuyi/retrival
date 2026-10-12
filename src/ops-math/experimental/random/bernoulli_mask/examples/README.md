# aclnnBernoulli ACLNN 调用样例

`test_aclnn_bernoulli.cpp` 使用标准两段式 ACLNN 接口运行一个 FP32
`[256, 256]` 用例，并检查：

- 输出只包含 `0` 和 `1`；
- 相同 `seed=20260725`、`offset=4` 的两次调用逐元素一致；
- `prob=0.35` 的样本均值位于二项分布均值的 6 倍标准差内；
- 两次调用报告相同的 workspace 大小。

先构建并安装 `bernoulli_mask` 自定义算子包，再在 `ops-math` 仓库根目录运行：

```bash
bash experimental/random/bernoulli_mask/examples/run.sh
```

也可以只编译样例：

```bash
bash experimental/random/bernoulli_mask/examples/run.sh --noexec
```

默认使用 `${ASCEND_HOME_PATH}/opp/vendors/custom_math`；若安装在其他位置，
通过 `BERNOULLI_CUSTOM_VENDOR_ROOT` 指定 vendor 根目录。构建输出默认放在
`build_out/bernoulli_mask_example/`，也可通过 `BERNOULLI_EXAMPLE_BUILD_DIR`
覆盖。

自包含 `libcust_opapi.so` 会复用系统 `libopapi.so` 中的
`DSAGenBitMask`，样例 CMake 已显式链接两者。此文件是最小演示，
完整 dtype、shape、非连续 view、in-place、边界与统计验收应使用仓库外层
的矩阵 runner；它不是运行本样例的前置依赖。
