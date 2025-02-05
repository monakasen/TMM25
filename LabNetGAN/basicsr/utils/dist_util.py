# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp




def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)



#def _init_dist_pytorch(backend="nccl", port=None):
#    """Initialize distributed training environment.
#    support both slurm and torch.distributed.launch
#    see torch.distributed.init_process_group() for more details
#    """
#    if mp.get_start_method(allow_none=True) is None:
#        if (
#            mp.get_start_method(allow_none=True) != "spawn"
#        ):  # Return the name of start method used for starting processes
#            mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
#    num_gpus = torch.cuda.device_count()
#
#    if "SLURM_JOB_ID" in os.environ:
#        rank = int(os.environ["SLURM_PROCID"])
#        world_size = int(os.environ["SLURM_NTASKS"])
#        node_list = os.environ["SLURM_NODELIST"]
#        print("当前有多少结点：%s" % node_list)
#        print("当前有多少结点：%s" % node_list)
#        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
#        print("当前地址为：%s" % str(addr))
#        print("当前地址为：%s" % str(addr))
#        # specify master port
#        if port is not None:
#            os.environ["MASTER_PORT"] = str(port)
#        elif "MASTER_PORT" not in os.environ:
#            os.environ["MASTER_PORT"] = "29500"
#        if "MASTER_ADDR" not in os.environ:
#            os.environ["MASTER_ADDR"] = addr
#        os.environ["WORLD_SIZE"] = str(world_size)
#        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
#        os.environ["RANK"] = str(rank)
#    else:
#        rank = int(os.environ["RANK"])
#        world_size = int(os.environ["WORLD_SIZE"])
#
#    print("当前可用gpu个数为：%d" % num_gpus)
#    print("当前可用gpu个数为：%d" % num_gpus)
#    print("当前rank为：%d" % rank)
#    print("当前rank为：%d" % rank)
#    print("world_size为：%d" % world_size)
#    print("world_size为：%d" % world_size)
#    torch.cuda.set_device(rank % num_gpus)
#
#    dist.init_process_group(
#        backend=backend,
#        world_size=world_size,
#        rank=rank,
#    )


#def _init_dist_slurm(backend="nccl", port=None):
#    """Initialize distributed training environment.
#    support both slurm and torch.distributed.launch
#    see torch.distributed.init_process_group() for more details
#    """
#    if mp.get_start_method(allow_none=True) is None:
#        if (
#            mp.get_start_method(allow_none=True) != "spawn"
#        ):  # Return the name of start method used for starting processes
#            mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
#    num_gpus = torch.cuda.device_count()
#
#    if "SLURM_JOB_ID" in os.environ:
#        rank = int(os.environ["SLURM_PROCID"])
#        world_size = int(os.environ["SLURM_NTASKS"])
#        node_list = os.environ["SLURM_NODELIST"]
#        print("当前有多少结点：%s" % node_list)
#        print("当前有多少结点：%s" % node_list)
#        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
#        print("当前地址为：%s" % str(addr))
#        print("当前地址为：%s" % str(addr))
#        # specify master port
#        if port is not None:
#            os.environ["MASTER_PORT"] = str(port)
#        elif "MASTER_PORT" not in os.environ:
#            os.environ["MASTER_PORT"] = "29500"
#        if "MASTER_ADDR" not in os.environ:
#            os.environ["MASTER_ADDR"] = addr
#        os.environ["WORLD_SIZE"] = str(world_size)
#        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
#        os.environ["RANK"] = str(rank)
#    else:
#        rank = int(os.environ["RANK"])
#        world_size = int(os.environ["WORLD_SIZE"])
#
#    print("当前可用gpu个数为：%d" % num_gpus)
#    print("当前可用gpu个数为：%d" % num_gpus)
#    print("当前rank为：%d" % rank)
#    print("当前rank为：%d" % rank)
#    print("world_size为：%d" % world_size)
#    print("world_size为：%d" % world_size)
#    torch.cuda.set_device(rank % num_gpus)
#
#    dist.init_process_group(
#        backend=backend,
#        world_size=world_size,
#        rank=rank,
#    )


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
