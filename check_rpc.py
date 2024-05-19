import torch
import torch.distributed as dist
import torch.nn as nn
import os
from time import sleep
from rpc_container import get_model_params, dict_for_rpc
# torch提供的进行进程间通信的库
import torch.distributed.rpc as rpc


# 初始化分布式训练环境
if __name__ == "__main__":
    # 设置必要的通信环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    # 获取local_rank
    rank = dist.get_rank()
    # 获取进程总数
    world_size = dist.get_world_size()
    # 设置当前进程对应的GPU
    torch.cuda.set_device(rank)
    # 为使用rpc进程间通信所必要的初始化，各个进程的name可以通过rank设置不同的名字
    torch.distributed.rpc.init_rpc("worker" + str(rank), rank=rank, world_size=world_size)

    #创建模型，并且在rank=1的进程上使得模型参数与其他进程不同
    model = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=1,
        bias=False
    ).cuda()
    if rank == 1:
        model.weight.data.fill_(1)
    # 将模型传入全局变量中，使得其可以被其他进程读取
    dict_for_rpc['model'] = model
    print(f'Rank:{rank} Model: {model.weight}')
    # 进程同步，防止某个进程读取另一个进程的模型时，后者模型还没有放入全局的字典容器中
    dist.barrier()    
    
    # rank=0的进程去读取rank=1的进程的模型参数
    if rank == 0:
        # rpc_sync为rpc库提供的通信函数，本调用代表在worker1的进程上运行get_model_params函数
        model_params = torch.distributed.rpc.rpc_sync("worker1", get_model_params)
        print(f"{rank} {model_params}")
        # 将rank=1读取出的模型参数拷贝到rank=0的模型上
        with torch.no_grad():
            for param, param_data in zip(model.parameters(), model_params):
                param.copy_(param_data.cuda())
        print(f"{rank} {list(model.parameters())}")
    sleep(3)
    dist.barrier()    
    print("-----------------------------")
    # 打印模型参数，可以发现rank=0的进程的模型参数已经和rank=1的参数相等了
    print(f'Rank:{rank} Model: {model.weight}')
    # 关闭进程组，结束程序。
    dist.destroy_process_group()

