import torch

#20230106 my impl of spatial transformer network(STN)

#some variables
bs   = 2
c    = 1
h    = 3
w    = 3

in_feat = torch.rand(bs, c, h, w)

#fake theta, because bs is 2, so theta also has the bs 2
theta = torch.tensor([[[1, 0, 1],
                       [0, 1, 1]],
                      [[0, 1, 1],
                       [1, 0, 1]]
                     ], dtype=torch.float32)

t = torch.linspace(start=0, end=h - 1, steps=h, dtype=torch.int32)
loc = torch.stack(torch.meshgrid([t, t]), dim=0).permute(1, 2, 0).reshape(h * w, 2)       #shape: 9 * 2 channel: y, x

#sample_loc = x * theta1 + y * theta2 + theta3
#上面这个等式的乘法其实输入和输出的结果影响都无所谓的，自己确定好乘法的输入输出哪个代表x哪个代表y即可。
sample_loc = + loc[None, :, 1:2] * theta[:, None, :, 0] + loc[None, :, 0:1] * theta[:, None, :, 1] + theta[:, None, :, 2] #channel x, y
sample_loc = torch.flip(sample_loc, dims=[2])                                                                             #channel y, x
sample_loc = torch.clamp(sample_loc, min=0, max=h-1-1e-6)

sample_loc_int = torch.floor(sample_loc).type(torch.long)
sample_loc_fra = sample_loc - sample_loc_int

batch_STN_features = []
#由于不同帧内取到的索引是不同的，为了避免难看的reshape此处采用循环进行操作。将batch维度reshape出去实际也是可以不需要循环就可以实现的
#但是当涉及到部署时，由于bs为1，因此直接不要这个循环就可以理解本段代码了(而不是看懂那些更难理解的reshape和取索引。
for i in range(bs):
    #左上角位置特征
    f1 = in_feat[i, :, sample_loc_int[i, :, 0], sample_loc_int[i, :, 1]] * \
          (1 - sample_loc_fra[i, :, 0])   * \
          (1 - sample_loc_fra[i, :, 1])
    #右上角位置特征
    f2 = in_feat[i, :, sample_loc_int[i, :, 0], sample_loc_int[i, :, 1] + 1] * \
          sample_loc_fra[i, :, 1]      * \
          (1 - sample_loc_fra[i, :, 0])
    #左下角位置特征
    f3 = in_feat[i, :, sample_loc_int[i, :, 0] + 1, sample_loc_int[i, :, 1]] * \
          sample_loc_fra[i, :, 0]  * \
          (1 - sample_loc_fra[i, :, 1])
    #右下角位置特征
    f4 = in_feat[i, :, sample_loc_int[i, :, 0] + 1, sample_loc_int[i, :, 1] + 1] * \
          sample_loc_fra[i, :, 0]  * \
          sample_loc_fra[i, :, 1]
    #四个位置特征相加即可得到一个空间转换后的特征
    STN_feature = (f1 + f2 + f3 + f4).reshape(c, h, w)
    batch_STN_features.append(STN_feature)

STN_output = torch.stack(batch_STN_features, dim=0)

print(in_feat)
print(STN_output)

# tensor([[[[0.9108, 0.5973, 0.4968],
#           [0.8086, 0.3526, 0.1735],
#           [0.1114, 0.3415, 0.9177]]],


#         [[[0.4148, 0.8808, 0.0972],
#           [0.1457, 0.2391, 0.8412],
#           [0.7301, 0.9952, 0.0050]]]])
# tensor([[[[0.3526, 0.1735, 0.1735],
#           [0.3415, 0.9177, 0.9177],
#           [0.3415, 0.9177, 0.9177]]],


#         [[[0.2391, 0.9952, 0.9952],
#           [0.8412, 0.0050, 0.0050],
#           [0.8412, 0.0050, 0.0050]]]])

# Process finished with exit code 0
