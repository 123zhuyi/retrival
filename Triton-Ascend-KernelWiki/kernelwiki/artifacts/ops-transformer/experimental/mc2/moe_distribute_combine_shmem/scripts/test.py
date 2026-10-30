from typing import List
import torch
import torch_npu
import numpy as np
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp
import toolss
from enum import Enum

####################
import shmem as shm
####################

# region 控制参数配置

# region 通信域配置

server_num = 1              # 单机或多机模式
tp_world_size = 1           # 不使能 tp 域通信
local_moe_expert_num = 3    # moe 卡上的专家数
shared_expert_rank_num = 0  # 共享专家卡数

server_index = 0
master_ip = "127.0.0.1"

rank_per_dev = 8                                # 每个 host 有几个 die
world_size = server_num * rank_per_dev          # 通信域总 die 数
ep_world_size = world_size // tp_world_size     # ep 域 die 数
moe_rank_num = ep_world_size - shared_expert_rank_num   # moe 专家卡数
moe_expert_num = moe_rank_num * local_moe_expert_num    # moe 专家数

# endregion

# region 输入输出维度相关

bs = 3
h = 7168
k = 3

global_bs = bs * ep_world_size
A_moe = global_bs * min(local_moe_expert_num, k)

# endregion

# region 产生什么样的输入

out_dtype = 0   # 控制 dispatch 输入及 combine 输出的类型，0 为 bfloat16, 1 为 float16
input_dtype = torch.bfloat16 if out_dtype == 0 else torch.float16

# endregion

# endregion

# region 精度对比常量

class ComparedOutput(Enum):
    EP_RECV_COUNTS = "ep_recv_counts"
    TP_RECV_COUNTS = "tp_recv_counts"
    EXPAND_X = "expand_x"
    EXPAND_IDX = "expand_idx"
    DYNAMIC_SCALES = "dynamic_scales"
    EXPERT_TOKEN_NUMS = "expert_token_nums"
    X = "x"

# endregion

# region 张量处理函数

def work_server_ranks(server_index):
    return slice(server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev)

def chunk_tensor(tensor, chunks):
    return list(tensor.chunk(chunks))

# endregion

# region 测试执行前预备工作相关函数

def print_setting():
    print(f"{server_num=}")
    print(f"{rank_per_dev=}")
    print(f"{tp_world_size=}")
    print(f"{ep_world_size=}")
    print(f"{shared_expert_rank_num=}")
    print(f"{moe_expert_num=}")
    print(f"{moe_rank_num=}")
    print(f"{local_moe_expert_num=}")
    print(f"{bs=}")
    print(f"{h=}")
    print(f"{k=}")

# endregion

# region npu 执行算子相关函数

def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")

def get_dispatch_kwargs(
    shmem_context, x, expert_ids, group_ep, ep_rank_id
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()

    return {
        "shmem_context": shmem_context,
        "x": x,
        "expert_ids": expert_ids,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "global_bs": global_bs,
    }

def get_combine_kwargs(
    shmem_context, expand_x, expert_ids, expand_idx,
    ep_send_counts, expert_scales, group_ep, ep_rank_id
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu()
    ep_send_counts = ep_send_counts.to(torch.int32).npu()
    expert_scales = expert_scales.to(torch.float32).npu()

    return {
        "shmem_context": shmem_context,
        "expand_x": expand_x,
        "expert_ids": expert_ids,
        "assist_info_for_combine": expand_idx,
        "ep_send_counts": ep_send_counts,
        "expert_scales": expert_scales,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "global_bs": global_bs,
    }

def run_cascade_npu(queue, rank, x, expert_ids, expert_scales):
    set_device(rank)

    ##################################################
    ret = shm.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed.")
    # 初始化参数
    init_attr = shm.InitAttr()
    init_attr.my_rank = rank
    init_attr.n_ranks = world_size
    init_attr.ip_port = "tcp://127.0.0.1:50001"
    init_attr.local_mem_size = 1024 * 1024 * 1024
    init_attr.option_attr.data_op_engine_type = shm.OpEngineType.MTE
    # 初始化
    ret = shm.aclshmem_init(init_attr)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init init failed.")
    shmem_context = shm.aclshmem_create_tensor([1, 2], dtype=torch.int8, device_id=rank)
    shmem_context.fill_(0)
    print("[INFO] shmem 初始化结束")
    ##################################################
    
    print(f'[INFO] device_{rank} 构造两算子输入数据')
    expert_scales = expert_scales.to(torch.float32).npu()
    dispatch_kwargs = get_dispatch_kwargs(
        shmem_context=shmem_context,
        x=x,
        expert_ids=expert_ids,
        group_ep="",
        ep_rank_id=rank // tp_world_size,
    )
    combine_kwargs = get_combine_kwargs(
        shmem_context=shmem_context,
        expand_x=torch.empty(0),
        expert_ids=expert_ids,
        expand_idx=torch.empty(0),
        ep_send_counts=torch.empty(0),
        expert_scales=expert_scales,
        group_ep="",
        ep_rank_id=rank // tp_world_size,
    )

    expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
    combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
    x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

    torch.npu.synchronize()
    print(f"[INFO] rank {rank} epid {rank // tp_world_size} npu finished! \n")

    queue.put([
        rank,
        [
            x.cpu()
        ]
    ])
    
def gen_npu(**server_kwargs):
    def parse_rank_input(result_queue, rank, server_kwargs):
        ep_id = rank // tp_world_size
        tp_id = rank % tp_world_size

        return {
            "queue": result_queue,
            "rank": rank,
            "x": server_kwargs["x_list"][tp_id][ep_id],
            "expert_ids": server_kwargs["expert_ids_list"][tp_id][ep_id],
            "expert_scales": server_kwargs["expert_scales_list"][tp_id][ep_id],
        }
    
    print("[INFO] single server scene!!!!")
    rank_list = list(range(world_size))
    print(f"[INFO] rank list is: {rank_list}")

    # 多进程，每个进程调用一张卡执行目标函数
    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)
    for rank in rank_list:
        rank_kwargs = parse_rank_input(result_queue, rank, server_kwargs)
        proc = Process(target=run_cascade_npu, kwargs=rank_kwargs)
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    rank_outputs = [None] * rank_per_dev
    for proc in proc_list:
        rank_id, rank_output = result_queue.get()
        local_rank_id = rank_id - server_index * rank_per_dev
        rank_outputs[local_rank_id] = rank_output
    
    # 将各类输出放入同一个列表中， category_outputs 存储各类输出的列表
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
        category_outputs.append(specific_category_output)
    
    return category_outputs

# endregion

# region 输入构造函数

def gen_x():
    x_list = []
    for _ in range(tp_world_size):
        cur_x = torch.arange(1, global_bs + 1, dtype=input_dtype).view(-1, 1).repeat(1, h)
        x_list.append(cur_x)
    return x_list

def gen_expert_ids():
    ep_expert_ids_list = []
    cards = torch.arange(ep_world_size * k, dtype=torch.int32).view(ep_world_size, k)
    order = list(range(1, ep_world_size)) + [0]
    ep_expert_ids = cards[order].repeat_interleave(bs, dim=0)
    ep_expert_ids_list.append(ep_expert_ids)
    return ep_expert_ids_list

def gen_expert_scales():
    # 固定权重为 1/k，权重和保证为 1
    return [torch.full(size=[global_bs, k], fill_value=1/k, dtype=torch.float32) for _ in range(tp_world_size)]

# endregion

# region golden 值生成函数

def gen_cascade_golden():
    values = torch.arange(1, ep_world_size * bs + 1, dtype=torch.bfloat16).view(ep_world_size, bs)
    golden_x_list = [
        values[i].unsqueeze(1).expand(bs, h)
        for i in range(ep_world_size)
    ]
    return golden_x_list

# endregion

# region 精度对比

def compare_output(compared_output: ComparedOutput, golden_per_card, npu_per_card, compare_dtype, compare_per_rank):
    def compare(golden, npu, dtype, tensor_name, diffThd=0.001, pctThd=0.001):
        np_output_npu = np.array(golden.cpu())
        np_output_golden = np.array(npu.cpu())
        
        max_diff_hd = 0.0
        if dtype == torch.int8:
            max_diff_hd = 1.0
        res_output = toolss.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)
        print(f"[INFO] tensor {tensor_name} output 测试的精度结果为： {res_output}")
        return res_output[0] == "PASS"

    mark_str = compared_output.value.upper().replace('_', ' ')
    result = {}

    if not compare_per_rank:
        print(f"############################ Compare {mark_str} #############################")
        res = compare(
            torch.cat(golden_per_card).to(compare_dtype),
            torch.cat(npu_per_card).to(compare_dtype),
            compare_dtype,
            compared_output.value,
        )
        result[compare_output.value] = res
    else:
        for i in range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev):
            print(f"############################ Compare {mark_str} rank: {i} epId: {i // tp_world_size} #############################")
            local_rank_id = i - server_index * rank_per_dev

            res = compare(
                golden_per_card[local_rank_id].to(compare_dtype),
                npu_per_card[local_rank_id].to(compare_dtype),
                compare_dtype,
                f"{compared_output.value}_{i}",
            )
            result[f"{compared_output.value}_{i}"] = res

    return result

# endregion

if __name__ == "__main__":
    print_setting()

    # 生成各 ep 域的 x
    x_list = gen_x()
    # 生成各 ep 域的 expert_ids
    expert_ids_list = gen_expert_ids()
    # 生成各 ep 域的 expert_scales
    expert_scales_list = gen_expert_scales()

    # 构造 golden 值
    golden_x = gen_cascade_golden()

    print("[INFO] Generate golden success.")
    server_ranks = work_server_ranks(server_index)

    # 执行 npu 任务
    [npu_x] = gen_npu(
        x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
        expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
        expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list],
    )

    print("[INFO] Generate npu success.")

    print("############################ Start Cascade Compare #############################")
    result_cascade = {}
    # 比较输出 x
    res = compare_output(
        ComparedOutput.X,
        golden_x[server_ranks],
        npu_x,
        torch.float32,
        True
    )
    result_cascade.update(res)
    print("############################ End Cascade Compare #############################")

    # 打印最终结果
    print(f"[INFO] result cascade: {result_cascade}")
    all_result = all(result_cascade.values())
    if all_result:
        print("[INFO] cascade test Success.")
    else:
        print("[INFO] cascade test Failed.")
