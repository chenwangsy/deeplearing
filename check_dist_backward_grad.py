import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# 定义一个简单的网络，返回值只有一个数字，就当成损失了。
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return torch.abs(torch.sum(x))

#分布式训练的入口函数，理论上来说这个函数无法接受关键字参数
def train(rank, world_size):
    """
        rank: 通过mp.spawn被自动传入的参数，不需要用户传入
        world_size: 进程总个数
    """
    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    #各个进程基于不同的rank设置对应的GPU
    torch.cuda.set_device(rank)
    #初始化模型
    model = MyNet().cuda().train()
    #将模型转换为ddp模型，此时模型参数自动被同步到rank=0的模型参数，保证了模型的初值相同
    ddp_model = DDP(model, device_ids=[rank])
    #官方教程中，模型先ddp后使用其参数初始化
    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    for i in range(1):
        optimizer.zero_grad()
        #由于未设置随机数种子，不同进程产生的输入是不一样的，模拟了真实的数据输入情况
        input_ft = torch.rand(1, 1, 2, 2).cuda()
        print(f"Rank: {rank}, input: {input_ft}")
        #输入传入ddp模型，得到损失
        loss = ddp_model(input_ft)
        #打印损失可以知道，不同进程中的损失值不相同，对模型产生的local_grad也是不相同的
        print(f"Rank: {rank}, loss: {loss}")
        #ddp模型在backward中注册了回调函数通过其reducer执行了all_reduce，使得各个进程的梯度相同
        loss.backward()
        #同步各个进程，这里是为了使得打印较为同步
        dist.barrier() 
        #打印各个进程的梯度和模型参数，可以发现他们是相同的
        print(f"Before step Rank: {rank}, param: {ddp_model.module.conv1.weight.data}")
        print(f"Before step Rank: {rank}, grad: {ddp_model.module.conv1.weight.grad}")
        optimizer.step()
        #优化器进行梯度下降算法后打印模型参数，可以发现模型参数也是一致的。
        print(f"AFTER Rank: {rank}, param: {ddp_model.module.conv1.weight.data}")    

if __name__ == "__main__":
    #设置环境变量的地址和端口号，不设置初始化进程dist.init_process_group会报错
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    #设置开启两个进程
    world_size = 2
    #pytorch官方推荐的启动多进程的一种方式，传入入口函数，入口函数参数，启动进程个数
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size
    )