import torch
import torch.distributed as dist

# 全局变量用于其他文件中将需要其他进程访问的数据传入该字典内
dict_for_rpc = {}

# 从全局变量中调用需要访问的数据，将其转换到cpu后进行返回
# rpc远程函数需要输入或者输出都转到cpu上进行通信
def get_model_params():
    model = dict_for_rpc['model']
    return [p.detach().cpu() for p in model.parameters()]
