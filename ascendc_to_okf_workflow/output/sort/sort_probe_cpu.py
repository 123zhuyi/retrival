# CPU-only 探针：精确确定 torch.sort(stable=True) 对 -0/+0、NaN、inf 的方向行为
# （不碰 NPU，纯 torch，用于修正 _sortable_key 的键映射以逐位匹配 golden）
import torch

def show(name, pat, desc):
    x = torch.tensor(pat, dtype=torch.float32)
    v, i = torch.sort(x, descending=desc, stable=True)
    print(f"{name} desc={desc}:")
    print(f"  values  = {v.tolist()}")
    print(f"  indices = {i.tolist()}")

print("1) signed_zero 模式 [-0.,0.,0.,-0.,1.,-1.,0.,-0.]")
for d in (False, True):
    show("  signed_zero", [-0.0,0.0,0.0,-0.0,1.0,-1.0,0.0,-0.0], d)

print("2) nan 模式 [3.,nan,1.,nan,2.,2.,-1.,0.]")
for d in (False, True):
    show("  nan", [3., float('nan'), 1., float('nan'), 2., 2., -1., 0.], d)

print("3) infinity 模式 [inf,3.,-inf,0.,3.,-0.,1.,-1.]")
for d in (False, True):
    show("  infinity", [float('inf'), 3., float('-inf'), 0., 3., -0., 1., -1.], d)

# 关键问题1：torch 是否把 -0.0 与 +0.0 视为相等(稳定保输入序)？
print("\n4) -0/+0 是否任一顺序先出(有无规律)：")
x = torch.tensor([-0.0, 0.0, -0.0, 0.0])
for d in (False, True):
    v, i = torch.sort(x, descending=d, stable=True)
    print(f"   [-0,0,-0,0] desc={d} -> indices={i.tolist()}")
x = torch.tensor([0.0, -0.0, 0.0, -0.0])
for d in (False, True):
    v, i = torch.sort(x, descending=d, stable=True)
    print(f"   [0,-0,0,-0] desc={d} -> indices={i.tolist()}")

# 关键问题2：nan 在升降序下的排位
print("\n5) NaN 排位：")
for d in (False, True):
    v, i = torch.sort(torch.tensor([1., 2., float('nan'), 3.]), descending=d, stable=True)
    print(f"   [1,2,nan,3] desc={d} -> v={v.tolist()} i={i.tolist()}")

# 关键问题3：+0/-0 与负小/正小的边界
print("\n6) 小正/小负与0：")
x = torch.tensor([-1e-30, -0.0, 0.0, 1e-30])
v, i = torch.sort(x, stable=True)
print(f"   [-1e-30,-0,0,1e-30] asc -> v={v.tolist()} i={i.tolist()}")