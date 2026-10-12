#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
from ml_dtypes import bfloat16


__golden__ = {
    "kernel": {
        "stateless_randperm": "stateless_randperm_golden"
    }
}


class RandpermGpuClient:
    def __init__(self, server_ip, server_port=8888):
        self.server_ip = server_ip
        self.server_port = server_port
        self._deps_loaded = False
        self._load_dependencies()

    def _load_dependencies(self):
        if not self._deps_loaded:
            global socket, pickle, struct, torch, np
            import socket
            import pickle
            import struct
            import torch
            import numpy as np
            self._deps_loaded = True

    def _recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _send_msg(self, sock, msg):
        msg = pickle.dumps(msg)
        msg = struct.pack('>Q', len(msg)) + msg
        sock.sendall(msg)

    def _recv_msg(self, sock):
        raw_msglen = self._recv_all(sock, 8)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>Q', raw_msglen)[0]
        return pickle.loads(self._recv_all(sock, msglen))

    def compute_on_gpu(self, seed=42, offset=10, n=100, dtype=9):
        request = {'seed': seed, 'offset': offset, 'n': n, 'dtype': dtype}
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(30)
                s.connect((self.server_ip, self.server_port))
                self._send_msg(s, request)
                result = self._recv_msg(s)
                return result
        except Exception as e:
            print(f"连接错误: {e}")
            return None


def compute_local(seed, offset, n):
    import torch
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    generator.set_offset(offset)
    result = torch.randperm(n, generator=generator, device='cpu')
    return result.numpy()


def stateless_randperm_golden(n, seed, offset, layout=0, dtype=9, **kwargs):
    '''
    Kernel golden for stateless_randperm.
    All the parameters follow @stateless_randperm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import logging
    GPU_SERVER_IP = "x.x.x.x"
    GPU_SERVER_PORT = 32323

    n_val = int(np.array(n).flatten()[0])
    seed_val = int(np.array(seed).flatten()[0])
    offset_val = int(np.array(offset).flatten()[0])

    client = RandpermGpuClient(GPU_SERVER_IP, GPU_SERVER_PORT)
    result = client.compute_on_gpu(seed=seed_val, offset=offset_val, n=n_val, dtype=dtype)
    if dtype == 27:
        result = result.astype(bfloat16)
    logging.info("remote gpu computation is done, the result type is {}.".format(type(result)))

    if result is None:
        logging.warning(f"remote gpu computation failed, switch to local cpu computation.")
        result = compute_local(seed=seed_val, offset=offset_val, n=n_val)

    return result
