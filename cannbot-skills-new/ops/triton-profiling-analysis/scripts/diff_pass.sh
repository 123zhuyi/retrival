#!/bin/bash
# diff_pass.sh — 对 cached/dumped ttadapter 重放 bishengir-compile，
# 输出指定 pass 的 before/after IR 并给出 diff 摘要（判定 pass 是否变换）。
#
# 用法: diff_pass.sh <kernel.ttadapter 路径> <pass名> [额外 bishengir flags...]
# 示例: diff_pass.sh /root/.triton/cache/xxx/_attn_fwd.ttadapter cv-pipelining
#       diff_pass.sh kernel.ttadapter.mlir cv-pipelining --set-workspace-multibuffer=2
set -u

TTADAPTER=$1
PASS=$2
shift 2

B=${BISHENGIR_COMPILE:-/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile}
if ! [ -x "$B" ]; then
    for c in /usr/local/python*/lib/python*/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile; do
        [ -x "$c" ] && B="$c" && break
    done
fi
[ -x "$B" ] || { echo "ERROR: bishengir-compile not found at $B"; exit 1; }
[ -f "$TTADAPTER" ] || { echo "ERROR: ttadapter not found: $TTADAPTER"; exit 1; }

if [ -z "${SOC_VERSION:-}" ]; then
    SOC=$(python3 "$(cd "$(dirname "$0")" && pwd)/detect_soc.py" 2>/dev/null || true)
    SOC=${SOC:-Ascend910_9372}
else
    SOC=$SOC_VERSION
fi
OUT=$(mktemp -t cv_pass_dump_XXXX).txt

# 必备 flags：--enable-hfusion-compile/--enable-triton-kernel-compile（triton 驱动默认带，
# 手动重放漏了会报 "failed to legalize operation 'linalg.generic'"）
timeout 900 "$B" "$TTADAPTER" \
  --target="$SOC" \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \
  --enable-triton-kernel-compile=true \
  "$@" \
  --mlir-print-ir-before="$PASS" \
  --mlir-print-ir-after="$PASS" \
  -o /dev/null > "$OUT" 2>&1
RC=$?
# 退出码非 0（常见 134 尾部崩溃）不影响 dump 有效性，以 IR Dump 是否落盘为准

if ! grep -q "IR Dump Before" "$OUT"; then
  echo "ERROR: 未产出 IR Dump（pass 可能未运行或编译失败）。日志: $OUT"
  head -20 "$OUT"
  exit 2
fi

python3 - "$OUT" "$PASS" <<'PYEOF'
import sys, difflib, re
txt = open(sys.argv[1]).read()
pass_name = sys.argv[2]
# 从日志里提取真实标记（pass 标题大小写不可推断，如 cv-pipelining → CVPipelining）
m = re.search(r"IR Dump Before (\S+) \(" + re.escape(pass_name) + r"\)", txt)
if not m:
    print(f"dump 标记未匹配（pass={pass_name}）。可用标记:")
    for t in re.findall(r"IR Dump \w+ (\S+)", txt):
        print("  ", t)
    sys.exit(2)
title = m.group(1)
bmark = f"IR Dump Before {title} ({pass_name}) //----- //"
amark = f"IR Dump After {title} ({pass_name}) //----- //"
if bmark not in txt or amark not in txt:
    print(f"dump 标记未匹配（pass={pass_name}），请检查 pass 名。可用标记:")
    import re
    for m in re.findall(r"IR Dump \w+ (\S+)", txt):
        print("  ", m)
    sys.exit(2)
def _strip_noise(t):
    # 滤除编译器警告等非 IR 噪声行（曾致 gate 的 cv-pipelining 假阳性：
    # 前后 IR 相同，仅 5 行 warning 混入 after 块）
    return "\n".join(l for l in t.splitlines()
                     if not re.search(r"warning|bisheng:|^\d+ warning", l))

b = txt.split(bmark)[1].split("// -----")[0]
rest = txt.split(amark)[1]
a = rest.split("// -----")[0] if "// -----" in rest else rest
same = _strip_noise(b).strip() == _strip_noise(a).strip()
print(f"pass={pass_name}")
print(f"before: {len(b.splitlines())} 行 | after: {len(a.splitlines())} 行")
print(f"变换: {'无（前后字节级相同，pass 未成功）' if same else '有（结构已改变，pass 成功）'}")
if not same:
    d = [l for l in difflib.unified_diff(b.splitlines(), a.splitlines(), lineterm="")
         if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
    print(f"diff 内容行数: {len(d)}")
    for name, body in [("before", b), ("after", a)]:
        print(f"  {name}: scf.for={body.count('scf.for')} matmul={body.count('matmul')} "
              f"mmad={body.count('mmad')} set_flag={body.count('set_flag')} wait_flag={body.count('wait_flag')}")
print(f"完整日志: {sys.argv[1]}")
PYEOF
