# IR 提取三途径

> 目标：拿到 Triton-Ascend kernel 各编译阶段的 IR，用于 pass 变换取证。
> 三个层次：`ttir`（前端）→ `ttadapter`（送 bishengir 前）→ `npuir`/HIVM（bishengir 内部各 pass 后）。

## 途径 1：triton 编译缓存（最快，只到 ttadapter）

每次编译的产物在 `/root/.triton/cache/<hash>/`（或 `~/.triton/cache`）：

```
<kernel>.ttir        # Triton IR（前端输出）
<kernel>.ttadapter   # 适配层 IR（bishengir 的输入）
<kernel>.json        # 元数据：options（num_stages/multibuffer/set_workspace_multibuffer...）
<kernel>.npubin      # 最终二进制
```

**变体辨认**（同一 kernel 有多个缓存目录，对应不同 constexpr/选项组合）：

```bash
python3 scripts/find_variants.py <kernel名>
# 或手动：ttir 里的常量暴露 shape/constexpr（如 %c4096_i32 = arith.constant 4096 → N_CTX=4096）
grep -l 'workspace_multibuffer": 2' /root/.triton/cache/*/<kernel>.json   # 按选项过滤
stat -c '%Y %n' ...                                                        # 按 mtime 对应运行批次
```

⚠️ 禁止凭目录 hash 名猜变体。

## 途径 2：TRITON_DEBUG dump（推荐，能拿到 HIVM/npuir）

参考 `/home/w00836812/triton-ascend/docs/zh/debug_guide/debugging.md`：

```bash
TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 python3 your_script.py
# 产出 ~/.triton/dump/<hash>/kernel.ttir.mlir + kernel.ttadapter.mlir
```

**关键：要拿 bishengir 内部 pass 的 IR（npuir.mlir），必须给 kernel 传 `debug=True`**：

```python
kernel[grid](..., debug=True)   # NPUOptions.debug 字段
```

`debug=True` 做两件事（`triton/backends/ascend/compiler.py`）：
1. 给 bishengir-compile 加 `--bishengir-print-ir-after=hivm-graph-sync-solver`
   （**同步结构最终版**，mix 拆分后的 `_mix_aic`/`_mix_aiv` 函数都在里面）
2. 把 bishengir 子进程 stdout/stderr 存为 `~/.triton/dump/<hash>/kernel.npuir.mlir`
   ——不加 debug=True 时子进程输出被 subprocess.PIPE 吞掉，什么都看不到

⚠️ 一次运行可能触发多次 bishengir 调用（aic/aiv 分开编译），dump 文件后者覆盖前者或合并，
若发现内容不全属正常现象；同步结构以 GraphSyncSolver dump 为准。

## 途径 3：手动重放 bishengir-compile（pass 级 before/after 取证，最强）

拿途径 1/2 得到的 ttadapter，手动跑 bishengir-compile，可注入任意 print flags：

```bash
B=/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile
$B <kernel.ttadapter> \
  --target=<SOC，如 Ascend910_9372> \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \          # ⚠️ 必补，triton 驱动自动加，漏了报 linalg.generic 无法 legalize
  --enable-triton-kernel-compile=true \    # ⚠️ 必补，同上
  --mlir-print-ir-before=<pass名> \
  --mlir-print-ir-after=<pass名> \
  -o /dev/null > dump.txt 2>&1
# 退出码 134（尾部崩溃）不影响 dump 有效性，以 IR Dump 是否落盘为准
```

用 `scripts/diff_pass.sh <ttadapter> <pass名>` 一键完成并 diff。

**常用 pass 名**（从 `AscendNPU-IR/bishengir/**/Passes.td` 或 `--help` 查）：

| pass | 作用 | 生效签名 |
|---|---|---|
| `cv-pipelining` | cube/vector 软件流水 | 循环拆成多个纯 cube/纯 vector 子循环（work item 化） |
| `hivm-graph-sync-solver` | 插入 set/wait flag、sync_block | dump 由 debug=True 自动产出 |

**pass 原理文档**：`/home/w00836812/AscendNPU-IR/docs/source/zh_cn/developer_guide/features/`
（cv_pipelining.md、multi_buffer.md、auto_sync.md、plan_memory.md 等）。
⚠️ 文档默认值可能过时——以 `bishengir-compile --help` 的安装版输出为准
（实测：`--set-workspace-multibuffer` 文档默认 2，安装版默认 4）。

## 运行时注入任意 flags（途径 3 的在线版）

若想让**真实 triton 编译流程**带上额外 flags（而非手动重放），monkeypatch 选项构造函数：

```python
import triton.backends.ascend.compiler as asc
_orig = asc.get_common_bishengir_compile_options
def patched(metadata):
    return _orig(metadata) + ["--mlir-print-ir-after=cv-pipelining"]  # 任意 flag
asc.get_common_bishengir_compile_options = patched
# 之后正常跑 kernel（加 debug=True 让输出落盘）
```

⚠️ 注入的 print 输出与其他 dump 共用 stdout，可能被后续编译覆盖；重要结论建议用手动重放（途径 3）复核。

## SOC 版本速查

`--target` 用设备名：`torch_npu.npu.get_device_name(0)`（如 `Ascend910_9372`）。
simulator lib 目录：`$CANN_HOME/aarch64-linux/simulator/<SOC>/lib`。
