### 论文标题：Image Super-Resolution with Taylor Expansion Approximation and Large Field Reception (TMM25) 代码在master里面

### **slurm集群的基本使用：**
#### 1. 查询当前有哪些任务在执行
```bash
squeue
```
#### 结果：
| JOBID | PARTITION |  NAME  | USER | ST |  TIME | NODES | NODELIST(REASON) |
|:-----:|:---------:|:------:|:----:|:--:|:-----:|:-----:|:----------------:|
|  1234 |   g_v100  | python |  fjc |  R | xx:xx |   1   |        gn9       |
|  1235 |   g_v100  | python |  fjc |  R | xx:xx |   1   |       gn10       |
#### 此时，我们可以观察到，有两个任务正在执行，JOBID分别为1234和1235，所使用的节点名分别为gn9和gn10
#### 2. 删除指定任务（例如，删除JOBID为1234的任务）
```bash
scancel 1234
squeue
```
#### 结果：
| JOBID | PARTITION |  NAME  | USER | ST |  TIME | NODES | NODELIST(REASON) |
|:-----:|:---------:|:------:|:----:|:--:|:-----:|:-----:|:----------------:|
|  1235 |   g_v100  | python |  fjc |  R | xx:xx |   1   |       gn10       |
