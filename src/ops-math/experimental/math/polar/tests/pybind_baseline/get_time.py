#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import csv  
from pathlib import Path  
import numpy as np


def get_time(file_path, time_use_list):
    with open(file_path, 'r', encoding='utf-8') as file:  
        reader = csv.DictReader(file)
        for row in reader:  
            time_use = row['Task Duration(us)']
            
            time_use_list.append(int(float(time_use)* 1000000))


def find_min_time():
    min_time = 0
    time_use_list = []
    directory = Path('./')
    filename_pattern = 'op_summary*.csv'
    
    for file in directory.rglob(filename_pattern):  
        get_time(file, time_use_list)
    
    if len(time_use_list) > 0:
        min_time = np.average(time_use_list[20:40]) 

    print(int(min_time))

if __name__ == '__main__':
    find_min_time()
    


