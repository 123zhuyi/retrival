# aclnnBernoulli ST

本目录是随算子提交的自包含 ACLNN 系统测试，不依赖竞赛工作区外层的
`tools/`。它覆盖：

- 10 种输出 dtype 的一般概率路径；
- 10 种 dtype 的非连续 view，分别验证 outplace 与 in-place，并检查
  view 外 storage guard 未被改写；
- 典型转置 view 的 outplace 与 in-place；
- 10 种 dtype 的 257 元素 alias/`SyncAll` 边界和百万元素多核路径；
- 1/2/4/8 字节输出的 alias/fallback workspace 阈值，以及
  127/128/129、255/256/257 mask 边界；
- FP16/FP32/FP64/BF16 四种 `prob` 标量 dtype；
- rank 0–8、空 Tensor、`prob=0/1` 和接近 0/1 的概率；
- 相同 seed/offset 的逐字节重现性，以及 seed/offset 改变后的流变化；
- 合法 offset `0/4/8`、非法 `offset % 4 != 0`，以及非法概率的参数拒绝。

大 shape 的 dense alias case 断言 workspace 不超过 4096 字节、不会随
元素数线性增长；1/2/4/8 字节阈值用例则成对断言 alias workspace 严格
小于 fallback。这个判据能区分旧 packed-mask workspace，同时允许 A2/A3
保留不同大小的 runtime 同步区。日志中的 `[METRIC]` 行会保留每个专项
case 的原始 workspace 字节数；本次 A2 实测 alias 为 1024、fallback
为 1536。

先从当前 checkout 构建并安装 `bernoulli_mask` 包，然后执行：

```bash
bash experimental/random/bernoulli_mask/tests/st/run.sh
```

只编译测试程序：

```bash
bash experimental/random/bernoulli_mask/tests/st/run.sh --noexec
```

默认使用逻辑 device 0。若未设置可见设备映射，可通过
`BERNOULLI_ST_DEVICE_ID` 选择设备：

```bash
unset ASCEND_RT_VISIBLE_DEVICES
BERNOULLI_ST_DEVICE_ID=6 \
  bash experimental/random/bernoulli_mask/tests/st/run.sh
```

默认从 `${ASCEND_HOME_PATH}/opp/vendors/custom_math` 加载已安装的自定义
opapi；非默认安装位置可设置 `BERNOULLI_CUSTOM_VENDOR_ROOT`。测试显式链接
`libcust_opapi.so` 与系统 `libopapi.so`，因为公共 ACLNN 实现复用了系统
`DSAGenBitMask` L0 符号。

Kernel 层的官方 TTK CSV 与 golden 位于相邻的 `../ttk/` 和 `../assets/`
目录。TTK 3.0 的可复现命令见算子顶层 README。
