#!/usr/bin/env python3
"""一 次性:将 8_SelfAttention(大 S) 数据集转为 level4_target workload.jsonl 格式,并归档当前小 S 集。

用法: python3 -I convert_selfattn_workload.py <src8> <dst_primary> <dst_smallS_archive>
  src8            8_SelfAttention.json(list-form 50 例)
  dst_primary     主目标 SelfAttention/workload.jsonl(覆盖为大 S)
  dst_smallS_archive  现存小 S workload.jsonl 的归档目标(先复制再覆盖主目标)
"""
import json
import shutil
import sys


def convert(src_path):
    rows = []
    for line in open(src_path, encoding="utf-8-sig"):
        if not line.strip():
            continue
        rec = json.loads(line)
        inp = {i["name"]: i for i in rec["inputs"]}
        x = inp["x"]
        nh = inp["n_heads"]
        rows.append({
            "uuid": f"SelfAttention-{len(rows) + 1:03d}",
            "axes": None,
            "inputs": {
                "x": {
                    "type": "random",
                    "shape": list(x["shape"]),
                    "dtype": x["dtype"],
                    "distribution": "mixed",
                },
                "n_heads": {"type": "scalar", "value": int(nh["value"])},
            },
        })
    return rows


def main():
    src8, dst_primary, dst_archive = sys.argv[1], sys.argv[2], sys.argv[3]
    # 1. 归档现存主目标(小 S)到 secondary 目录
    shutil.copyfile(dst_primary, dst_archive)
    # 2. 转大 S 并覆盖主目标
    rows = convert(src8)
    with open(dst_primary, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. 自检 footprint
    import collections
    dt, hds, ss, bs, dm = collections.Counter(), [], [], [], []
    for r in rows:
        x = r["inputs"]["x"]; nh = r["inputs"]["n_heads"]["value"]
        shape = x["shape"]; hd = shape[-1] // nh
        dt[x["dtype"]] += 1; hds.append(hd); ss.append(shape[1]); bs.append(shape[0]); dm.append(shape[2])
    print(f"wrote {len(rows)} cases -> {dst_primary}")
    print("archived ->", dst_archive)
    print("dtype:", dict(dt))
    print("head_dim distinct:", sorted(set(hds)))
    print(f"S min/max: {min(ss)}/{max(ss)}  全部S>64: {min(ss) > 64}")
    print(f"B min/max: {min(bs)}/{max(bs)}  d_model min/max: {min(dm)}/{max(dm)}")


if __name__ == "__main__":
    main()