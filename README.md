### 论文标题：Image Super-Resolution with Taylor Expansion Approximation and Large Field Reception (TMM25)

### **代码在master里面**

### **slurm集群的基本使用：**
#### 1. 查询当前有哪些任务在执行
```bash
squeue
```
结果：
| JOBID | PARTITION |  NAME  | USER | ST |  TIME | NODES | NODELIST(REASON) |
|:-----:|:---------:|:------:|:----:|:--:|:-----:|:-----:|:----------------:|
|  1234 |   g_v100  | python |  fjc |  R | xx:xx |   1   |        gn9       |
|  1235 |   g_v100  | python |  fjc |  R | xx:xx |   1   |       gn10       |
#### 2. 查询节点状态
```bash
sinfo
```
