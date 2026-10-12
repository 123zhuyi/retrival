# win区dump数据解析脚本

## 背景说明

- exception dump：算子在出现aic error后，会触发exception dump功能，将算子输入输出tensor dump成bin文件，其中dispatchv2&combinev2算子还额外注册异常处理回调，会将通信中保存状态位的windows内存dump成bin文件，其中该bin文件中的最后1M的数据称为win区数据。里面包含卡住、aic error等常见问题定位所需数据。

- 在dispatch&combine问题定位中，需要根据多卡数据进行整体分析判断，分析难度大，因此需要开发多卡数据分析工具来快速分析，要求能够根据已有数据，一键式解析数据，按照既定规则分析出可能异常的点，直接打印异常情况描述，部分数据解析完成后需输出csv文件，工具需打印当前正在进行哪一项分析。

## 功能说明

- dump数据解析: 对指定的dump数据进行解析，获得该dump数据的win区数据，获取出每张卡对应的moe专家数、使用核数、执行次数、epworldsize以及0/1标识位，并判断每个核上的这些参数是否一致，并将分析出有异常的点输出，可以用于定位moe专家数、epworldsize输入异常的问题。

- 输入异常分析: 可以用于分析当算子卡死时，是否是因为参数输入异常导致的卡死

- 执行序分析: 可以用于定位因执行次数不匹配导致的卡死问题。

- 状态位分析: 获取每个卡中每个核的执行位置信息，判断哪些核没有等到状态位，并根据对应的dispatchv2&combinev2的0/1标识位，到对应的0/1状态区找出没等到状态的核里面具体是第几个状态位没有等到，用于展示算子卡死后的具体现象。

## 脚本输入输出说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th> 
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>TARGET_PATH</td>
   <td>输入</td>
   <td>dump数据所在的文件路径，调用pytorch接口的日志落盘位置通过ASCEND_WORK_PATH环境变量控制，通过查看日志可以得到dump数据的落盘位置，未设置该环境变量时,dump数据落盘在当前目录下extra-info/data-dump/路径。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>TOOL_PATH</td>
   <td>输入</td>
   <td>装包路径中cann仓所在的文件路径，如:TOOL_PATH=install_pkg/cann-x.x.x/。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SOC_VERSION</td>
   <td>输入</td>
   <td>构造dump数据时对应的版本输入，当前仅支持如:SOC_VERSION=910_93 or 950。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SHARE_EXPERT_CARD_COUNT</td>
   <td>可选输入</td>
   <td>构造dump数据时的输入共享专家卡数,未输入时默认值为0, SHARE_EXPERT_CARD_COUNT <= 总卡数，如:SHARE_EXPERT_CARD_COUNT=1。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>SP_MOE_NUM</td>
   <td>可选输入</td>
   <td>构造dump数据时的输入特殊专家数,未输入时默认值为0, SP_MOE_NUM >0，如:SHARE_EXPERT_CARD_COUNT=1。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>TP_WORLDSIZE</td>
   <td>可选输入</td>
   <td>构造dump数据时的TP_WORLDSIZE,未输入时默认值为1, TP_WORLDSIZE >= 1，如:TP_WORLDSIZE=1。</td>
   <td>INT</td>
  </tr>
  <tr>
    <td>SHARE_EXPERT_NUM</td>
    <td>可选输入</td>
    <td>构造dump数据时的输入共享专家数,未输入时默认值为0, SHARE_EXPERT_NUM > 0，如:SHARE_EXPERT_NUM=1</td>
    <td>INT</td>
  </tr>
  <tr>
   <td>打屏日志</td>
   <td>输出</td>
   <td>Info:分析出的win区数据信息，如：该卡dispatch使用核数：72,combine使用核数：72。该卡dispatch moe专家数：62,combine moe专家数：62<br> Warning:分析出的异常点。如：执行次数dispatch = combine + 1，因此认为该卡挂在dispatch算子上。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>win_status_list</td>
   <td>输出</td>
   <td>存放win区中0/1标识区指定的dispatchv2/combinev2的0/1状态区的所有状态位数据，用于查看没等到状态的核里面具体是第几个状态位没有等到。<br>
       如:解析出该卡dispatchv2的状态区储存在0区，则将win区dump数据中的dispatch 0区状态区[0:64 * 1024]的数据存入win_status_list</td>
   <td>csv表</td>
  </tr>
  <tr>                                                 
   <td>win_all_card_data</td>
   <td>输出</td>
   <td>存放每张卡dispatchv2&combinev2的执行次数、moe专家数以及globalbs,用于对比每张卡的这些参数是否一致。<br> 如:d0 dispatch: 1 指第一张卡dispatchv2算子运行了1次</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_analysis_list</td>
   <td>输出</td>
   <td>存放分析出的异常点及异常位置，即把打屏日志中的Warning信息存储在该csv文件中。<br>如：d0卡dispatchv2算子第35个核的第2个状态位没有等到</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data</td>
   <td>输出</td>
   <td>存放解析出的各卡的moe专家数、使用核数、执行次数、epworldsize、rankid、globalbs、hccl中输入的rankid和globalbs、计算得出的bs、k、0/1标识位数据。<br>如:d0_dispatch_moe专家数: 65</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data_list</td>
   <td>输出</td>
   <td>存放解析出的各卡中每个核的执行位置情况以及0/1标识位数据，<br>如：d0卡dispatchv2的使用核数为72，则将72个核的win区的moe专家数记录为一个长度为72的列表并储存至win_data_list，d0_dispatch_0/1标识区数据:[0:71]。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_all_card_expandidx</td>
   <td>输出</td>
   <td>存放解析出的各卡中输入expertids、dump数据中读取的exoandidx(仅挂在combine算子上的卡，否则为空)、本卡专家数、该卡挂在哪个算子上，<br>如：0卡 本卡专家数: 8。</td>
   <td>csv表</td>
  </tr>
 </tbody>
</table>

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| dump_analysis.sh脚本直调 | bash dump_analysis.sh TARGET_PATH=xxx TOOL_PATH=xxx SP_MOE_NUM=0 TP_WORLDSIZE=1 SOC_VERSION=950 SHARE_EXPERT_CARD_COUNT=1 SHARE_EXPERT_NUM=1| 通过对sh脚本进行入参对指定的TARGET_PATH路径下的dump数据进行dump数据解析。 |

## 样例代码结果说明

[INFO] 开始分析卡1数据<br>
[INFO] 解析文件:/xxx/xxx/xxx/data-dump/mc2_exception_info.xxx<br>
<b>上述结果为当前正在分析的卡的数据信息，会输出当前分析卡的序列号以及对应的dump文件</b><br>

[INFO] 1. 开始进行输入异常分析

[INFO] 1. 该卡dispatch&combine使用核数为 dispatch:72,combine:72<br>
[INFO] 1.1 dispatch各核的rankid:[1,...,1]<br>
[INFO] 1.1 dispatch各核的epworldsize:[2,...,2]<br>
[INFO] 1.1 dispatch各核的hccl中的rankid:[0,...,0]<br>
[INFO] 1.1 dispatch各核的hccl中的epworldsize:[0,...,0]<br>
[INFO] 1.1 dispatch各核的moe专家数:[1,...,1]<br>
[INFO] 1.1 dispatch各核的globalbs:[1,...,1]<br>
[INFO] 1.1 dispatch_rankid:1 dispatch_hccl_rankid:0, dispatch epworldsize:2, dispatch_hccl epworldsize:0, dispatch moe专家数:16, dispatch globalbs:2, 根据dispatch输入计算的bs:1<br>
[WARNING] 1.1 dispatch win区数据中的rankid:1 与 hccl的rankid输入:0 不同<br>
[WARNING] 1.1 dispatch win区数据中的epworldsize:1 与 hccl的epworldsize输入:0 不同<br>

[INFO] 1.1 combine各核的rankid:[1,...,1]<br>
[INFO] 1.1 combine各核的epworldsize:[2,...,2]<br>
[INFO] 1.1 combine各核的hccl中的rankid:[0,...,0]<br>
[INFO] 1.1 combine各核的hccl中的epworldsize:[0,...,0]<br>
[INFO] 1.1 combine各核的moe专家数:[1,...,1]<br>
[INFO] 1.1 combine各核的globalbs:[1,...,1]<br>
[INFO] 1.1 combine_rankid:1 combine_hccl_rankid:0, combine epworldsize:2, combine_hccl epworldsize:0, combine moe专家数:16, combine globalbs:2, 根据combine输入计算的bs:1<br>
[WARNING] 1.1 combine win区数据中的rankid:1 与 hccl的rankid输入:0 不同<br>
[WARNING] 1.1 combine win区数据中的epworldsize:1 与 hccl的epworldsize输入:0 不同<br>
<b>上述结果为对dump数据中的各项数据进行取值以及对比rankid和epworldsize的输入与hccl中的输入是否相同，用于各项参数输入异常导致的卡死问题，并将分析出的异常用warning的形式输出。</b><br><br>

[INFO] 1.2 该卡不为共享专家卡
[INFO] 1.2 本卡专家数为:8
[INFO] 1.2 根据计算得出该卡的bs=1
[INFO] 1.2 根据dump数据文件名MoeDistributeDispatchV2......hote.o，判断该卡挂在dispatch
[INFO] 1.3 该卡的输入expertids为[[ 2 5 ...... 11]]

[INFO] 1.3 未检测到该卡的expertids有输入异常
[INFO] 1.3 根据计算得出该卡的k为:8
[INFO] 1.4 该卡没有挂在combine算子上，不进行epsendcnt分析
[INFO] 1.5 输入异常分析完成
<b>上述结果为对dump数据中的输入expertids、epsendcnt进行取值以及校验是否有非法输入与hccl中的输入是否相同，用于这两个参数输入异常导致的卡死问题，并将分析出的异常用warning的形式输出。</b><br><br>

[INFO] 2. 开始执行序分析<br>
[INFO] 2. dispatch各核执行次数:[7,7,7,7,7,7,.....,7]<br>
[INFO] 2. combine各核执行次数:[6,6,6,6,6,6,6,....,6]<br>
[WARNING] 2. dispatch执行次数:7 = combine执行次数:6 + 1,挂在dispatch上<br>
[INFO] 2. 执行序分析完成<br>
<b>上述结果为执行序分析，会输出当前分析卡的对应文件、当前卡中的dispatch&combine算子的使用核数、各个核的dispatch&combine算子执行次数以及对比执行次数用于定位因执行次数不匹配导致的卡死问题，并将分析出的异常用warning的形式输出。</b><br><br>

[INFO] 3. 开始dispatch状态位分析<br>
[INFO] 3.1 dispatch_epworldsize:2, dispatch moe专家数:65<br>
[INFO] 3.1 dispatch各核执行位置情况:[1,1,1,1,1,...,1]<br>
[INFO] 3.1 该卡不为共享专家卡<br>
[INFO] 3.1 dispatch 总状态位:130<br>
[INFO] 3.2 dispatch 1区状态区数据:int32<br>
[INFO] 3.2 dispatch 1区状态区数据shape:16384<br>
[INFO] 3.2 dispatch 1区状态区数据:[0,3,0,.......,]<br>
[INFO] 3.2 dispatch 中各核分配到的状态位数量:[2,2,2,2,2,....,1,1,1,...,1]<br>
[WARNING] 3.2 dispatch 中有如下下标的核没有等到状态[0,1,2,3,...,71]共72个核<br>
[INFO] 3.3 dispatch状态位分析完成<br>
<b>上述结果为dispatch的状态位分析，会输出当前卡dispatch所使用的epworldsize、moe专家数以及总状态位数量,以及会将各核的执行位置情况、对应状态区的数据信息、各核分配到的状态位数量情况以及没有等到状态的核的下标打印出来，并将分析出的异常用warning的形式输出。</b><br><br>

[INFO] 4. 开始combine状态位分析<br>
[INFO] 4.1 combine_epworldsize:2, combine moe专家数:65<br>
[INFO] 4.1 combine各核执行位置情况:[2,2,2,2,2,2,2,1,...,1]<br>
[INFO] 4.1 combine 总状态位:272<br>
[INFO] 4.2 combine 1区状态区数据:int32<br>
[INFO] 4.2 combine 1区状态区数据shape:81920<br>
[INFO] 4.2 combine 1区状态区数据:[1,3,0,0,0,0,.......,0]<br>
[INFO] 4.2 combine 中各核分配到的状态位数量:[17,17,...0,.......,0]<br>
[WARNING] 4.2 combine 中有如下下标的核没有等到状态[7,8,9,...,71]共65个核<br>
[INFO] 4.3 combine状态位分析完成<br>
<b>上述结果为combine的状态位分析，会输出当前卡combine所使用的epworldsize、moe专家数以及总状态位数量,以及会将各核的执行位置情况、对应状态区的数据信息、各核分配到的状态位数量情况以及没有等到状态的核的下标打印出来，并将分析出的异常用warning的形式输出。</b><br><br>

[INFO] 5. 数据归档<br>
[INFO] 5. 该卡的dispatch&combine的使用核数、epworldsize、moe专家数、0/1标识区数据已归档至win_data.csv<br>
[INFO] 5. 该卡中各核的dispatch&combine的使用核数、执行位置、0/1标识区数据已归档至win_data_list.csv<<br>
[INFO] 5. 该卡所使用的dispatch&combine的状态区数据已归档至win_status_list.csv<br>
[INFO] 5. 分析出的错误详细信息已归档至win_analysis_error.csv<br>
<b>上述结果为分析数据归档，将分析时用到的win区dump数据归档至对应的csv文件中，详细信息可参考脚本输出说明</b><br><br>

[INFO] 6. 开始进行多卡的dispatch&combine数据对比<br>
[INFO] 6. 多卡的dispatch&combine的执行次数完全相同<br>
[INFO] 6. 多卡的dispatch&combine的moe专家数完全相同<br>
[INFO] 6. 多卡的dispatch&combine的globalbs完全相同<br>
[INFO] 6. 各卡的dispatch执行次数:[7,7]<<br>
[INFO] 6. 各卡的combine执行次数:[6,6]<br>
[INFO] 6. 各卡的dispatch的moe专家数:[1616]<<br>
[INFO] 6. 各卡的combine的moe专家数:[16,16]<br>
[INFO] 6. 各卡的dispatch的globalbs:[2,2]<<br>
[INFO] 6. 各卡的combine的globalbs:[2,2]<br>
[INFO] 6. 各卡的dispatch&combine执行次数、moe专家数、globalbs数据已归档至win_all_card_data.csv<br>
<b>上述结果为多卡间的dispatch&combine的执行次数、moe专家数、globalbs对比，当多卡间的dispatch/combine算子的上诉数据不同时，将该异常以warning的形式输出，其中moe专家数的异常以error的形式输出，并将各卡的dispatch&combine执行次数存储至win_all_card_run_num.csv。</b>

[INFO] 7. 开始所有卡的expandidx对比
[INFO] 7. 所有专家的expertidx如下:{0: [(0, 0, 6)],....., 14: [(0, 0, 1), (1, 0, 2)]}
[INFO] 7. 卡0的expandidx没有异常
[INFO] 7. 卡1没有挂在combine不进行异常分析
[INFO] 7. 各卡的expertids、dump数据中读取的expandidx、本卡专家数、该卡挂在哪个算子上已归档至win_all_card_expandidx.csv
[INFO] 7. 所有卡的expandidx对比完成
<b>上述结果为多卡间的dispatch&combine的expandidx输入校验，该功能会计算出所有专家对应的expertidx，当该卡挂在combine算子上时，对该卡的输入expertidx与计算的expertidx进行对比，如果不同，将该异常以warning的形式输出，并将各卡的输入expertids、dump数据中读取的exoandidx(仅挂在combine算子上的卡，否则为空)、本卡专家数、该卡挂在哪个算子上，四项数据存储至win_all_card_run_num.csv。</b>
