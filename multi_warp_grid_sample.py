import torch
import torch.nn as nn
import torch.nn.functional

feats = torch.rand((4, 3, 10, 10)).cuda()

homo_offset = [torch.rand((4, 8, 5, 2)).cuda() for _ in range(5)]

#可以通过两种方式来得到multi_warp进行grid_sample的结果，第一种是通过for循环
#让homo每次都作用feats来得到结果列表最后concat起来，第二种是首先将feats和homo_offset在
#batch维度上堆叠起来，经过一次grid_sample后再在batch维度拆分，然后再将得到的多组特征拼接
#在channel维度上。这两种方式得到的结果相同，hat在spatialTransformerWithOffset采用的是第二种

#way 1 to get output
sample_list = []
for homo in homo_offset:
    sample_feats = torch.nn.functional.grid_sample(feats, homo, align_corners=False)
    sample_list.append(sample_feats)
sample_feats = torch.cat(sample_list, dim=1)
print(sample_feats.shape)

#way 2 to get output
multi_feats = feats.repeat((5, 1, 1, 1))
homo_offset = torch.cat(homo_offset, dim=0)
multi_feats = torch.nn.functional.grid_sample(multi_feats, homo_offset, align_corners=False)
multi_feats = torch.cat(multi_feats.split(4, dim=0), dim=1)
print(multi_feats.shape)

print(torch.equal(sample_feats, multi_feats))

print(torch.max(torch.abs(sample_feats - multi_feats)))
