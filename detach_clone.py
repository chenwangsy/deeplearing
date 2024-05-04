import torch
import torch.nn as nn 
import random
import numpy as np
#simulate lidar mmoe [shared det seg]
det_bs = 1
seg_bs = 1
det_input = torch.tensor([1, 2.0]).reshape(1, 2, 1, 1)                                      #模拟det数据输入
seg_input = torch.tensor([5, 6.0]).reshape(1, 2, 1, 1)                                      #模拟seg数据输入
vfe = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False)                   #模型vfe部分
vfe.weight.data = torch.tensor([1, 2.0]).reshape(1, 2, 1, 1)
conv_share = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)            #模型share backbone部分
conv_share.weight.data = torch.tensor([3.0]).reshape(1, 1, 1, 1)
conv_det   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)            #模型det backbone部分
conv_det.weight.data = torch.tensor([2.0]).reshape(1, 1, 1, 1)
conv_seg   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)            #模型seg backbone部分
conv_seg.weight.data = torch.tensor([-1.0]).reshape(1, 1, 1, 1)
fuse_det   = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False)            #模型det fuse部分
fuse_det.weight.data = torch.tensor([-1.0, 1.0]).reshape(1, 2, 1, 1)
fuse_seg   = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False)            #模型seg fuse部分
fuse_seg.weight.data = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1)
#forward det
det_vfe = vfe(det_input)                                                                    #det数据进行vfe
det_vfe_detach = det_vfe.detach().clone()                                                   #防止det的梯度回传入vfe参数, 由于后续不需要这个量的梯度，因此没有设置其requires_grad属性为true
det_backbone_ft = conv_det(det_vfe_detach)                                                  #利用不会回传给vfe参数的中间特征进行det backbone前向传播
#forward seg
seg_vfe = vfe(seg_input)                                                                    #seg数据进行vfe
seg_vfe_detach = seg_vfe.detach().clone()                                                   #防止seg的梯度回传入vfe参数, 由于后续不需要这个量的梯度，因此没有设置其requires_grad属性为true
seg_backbone_ft = conv_seg(seg_vfe_detach)                                                  #利用不会回传给vfe参数的中间特征进行seg backbone前向传播
#forward share
share_backbone_ft = conv_share(torch.cat([det_vfe, seg_vfe], dim=0))                        #将det seg的vfe结果结合起来(注意这里使用的是可以回传进入vfe的那个tensor)进行share backbone推理
                                                                                            #share_backbone_ft内保存了它是怎么从前端计算的公式，因此后续需要使用它来回传 share 和 vfe 的参数梯度
share_backbone_ft_detach = share_backbone_ft.detach().clone()                               #防止使用share_backbone_ft的后续模块将梯度回传给share backbone参数
share_backbone_ft_detach.requires_grad = True                                               #但需要后续模块将梯度回传给这个tensor本身，因此设置其requires_grad属性为true
det_share_backbone_ft = share_backbone_ft_detach[:det_bs]                                   #按照bs将seg det的特征分出来
seg_share_backbone_ft = share_backbone_ft_detach[det_bs:det_bs+seg_bs]                      #按照bs将seg det的特征分出来

loss_det = fuse_det(torch.cat([det_backbone_ft, det_share_backbone_ft], dim=1))             #模拟det检测头的后续网络

loss_seg = fuse_seg(torch.cat([seg_backbone_ft, seg_share_backbone_ft], dim=1))             #模拟seg检测头的后续网络

loss_det.backward()                                                                         #回传所有det任务独占参数的梯度
loss_seg.backward()                                                                         #回传所有seg任务独占参数的梯度
share_backbone_ft.backward(share_backbone_ft_detach.grad)                                   #回传share和vfe参数的梯度, 其中share_backbone_ft_detach的梯度是来源于后端计算的反向传播，share_backbone_ft内记录了vfe 和 share backbone的计算方式
                                                                                            #他们两个一起作用完成了共享参数部分的梯度回传



#unrelate update