### 论文标题：Image Super-Resolution with Taylor Expansion Approximation and Large Field Reception (TMM25) 代码在master里面

### **slurm集群的基本使用：**
#### 1. 查询当前有哪些任务在执行
```bash
squeue
```
#### 结果：
| JOBID | PARTITION |  NAME  | USER | ST |  TIME | NODES | NODELIST(REASON) |
|:-----:|:---------:|:------:|:----:|:--:|:-----:|:-----:|:----------------:|
|  1234 |   g_v100  | python |  fjc |  R | xx:xx |   1   |        gn8       |
|  1235 |   g_v100  | python |  fjc |  R | xx:xx |   1   |       gn10       |
#### 此时，我们可以观察到，有两个任务正在执行，JOBID分别为1234和1235，所使用的节点名分别为gn8和gn10
#### 2. 删除指定任务（例如，删除JOBID为1234的任务）
```bash
scancel 1234
squeue
```
#### 结果：
| JOBID | PARTITION |  NAME  | USER | ST |  TIME | NODES | NODELIST(REASON) |
|:-----:|:---------:|:------:|:----:|:--:|:-----:|:-----:|:----------------:|
|  1235 |   g_v100  | python |  fjc |  R | xx:xx |   1   |       gn10       |
#### 3. 查询节点情况
```bash
sinfo
```
| PARTITION | AVAIL | TIMELIMIT | NODES | STATE |  NODELIST  |
|:---------:|:-----:|:---------:|:-----:|:-----:|:----------:|
|   g_v100  |   up  |  infinite |   3   |  mix  | gn[4,8,10] |
|   g_v100  |   up  |  infinite |   4   | alloc |  gn[1-3,5] |
|   g_v100  |   up  |  infinite |   3   |  idle |  gn[6-7,9] |
#### 其中，STATE为mix代表该节点部分显卡已被使用，alloc代表全部显卡已被使用，idle代表全部显卡空闲。
#### 例如，gn4、gn8和gn10三个节点有部分显卡被使用，gn1、gn2、gn3和gn5所有显卡被使用，gn6、gn7和gn9空闲。
#### 4. 显示节点不正常工作的原因
```bash
sinfo -R
```
|  REASON  | USER | TIMESTAMP | NODELIST |
|:--------:|:----:|:---------:|:--------:|
| for test |  fjc |   xx:xx   |   gn[8]  |
#### 5. 进入到本账号正在使用的节点
```bash
ssh gn8
```
#### 从原本的(base) fjc@ln0:~$ ssh gn8，变为(base) wangymlab03@gn8:~$ 
#### 此时，我们可以查看当前节点显卡使用情况
```bash
watch -n 1 nvidia-smi
```
Thu Feb 20 xx:xx 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-32GB           Off |   00000000:61:00.0 Off |                    0 |
| N/A   59C    P0            205W /  250W |   32468MiB /  32768MiB |     74%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-PCIE-32GB           Off |   00000000:DB:00.0 Off |                    0 |
| N/A   53C    P0             47W /  250W |   22710MiB /  32768MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A   1094399      C   .../anaconda3/envs/fjc/bin/python           32464MiB |
|    1   N/A  N/A   1041358      C   python                                      11396MiB |
|    1   N/A  N/A   1050686      C   python                                      11310MiB |
+-----------------------------------------------------------------------------------------+
