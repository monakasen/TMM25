### 论文标题：Image Super-Resolution with Taylor Expansion Approximation and Large Field Reception (TMM25)，代码在master里面

# **slurm集群的常用命令：**
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
#### 从原本的(base) fjc@ln0:\~\$ ssh gn8，变为(base) fjc@gn8:\~\$ 
#### 其中，ln为登陆节点。此时，我们可以查看当前节点显卡使用情况
```bash
watch -n 1 nvidia-smi
```
#### 6. 退出当前节点（先按住键盘上的ctrl，然后依次按下a和d）
```bash
ctrl+a+d
```
#### 从(base) fjc@gn8:\~\$，变回(base) fjc@ln0:\~\$
#### 7. 执行python脚本
```bash
srun -p g_v100 --nodelist=gn8 --gpus=1 python xxx.py
```
#### 其中，-p指定特定分区名称，--nodelist指定节点名。--gpus指定GPU使用数量。 

## **screen指令创建窗口：**
#### 值得注意的是，当使用srun长时间执行任务时，我们可以结合screen来创建窗口：
```bash
screen -S test
```
#### 其中,-S test表示创建一个名为test的窗口。我们还可以查询有哪些窗口：
```bash
screen -ls
```
#### 退出窗口：
```bash
ctrl+a+d
```
#### 进入某个id对应的窗口（先使用screen -ls查询，得到窗口id）：
```bash
screen -r 窗口id
```
#### 删除窗口：
```bash
screen -S 窗口id -X quit
```
