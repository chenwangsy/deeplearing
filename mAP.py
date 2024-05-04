import sys

res = {}
classes = ['car', 'person']
confs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
for class_name in classes:
    clsmap = {}
    for conf in [i * 10 for i in range(1, 10)]:
        clsmap[conf] = {'TP': 0, 'TP+FP': 0, 'TP+FN': 0}
    res[class_name] = clsmap
gt = open('./gt.txt').read().split('\n')
gtboxes = [i.split() for i in gt]
dr = open('./dr.txt').read().split('\n')
drboxes = [i.split() for i in dr]

import numpy as np
import torch

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap

def calculate_iou(_box_a, _box_b):
    # -----------------------------------------------------------#
    #   计算真实框的左上角和右下角
    # -----------------------------------------------------------#
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # -----------------------------------------------------------#
    #   计算先验框获得的预测框的左上角和右下角
    # -----------------------------------------------------------#
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

    # -----------------------------------------------------------#
    #   将真实框和预测框都转化成左上角右下角的形式
    # -----------------------------------------------------------#
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

    # -----------------------------------------------------------#
    #   A为真实框的数量，B为先验框的数量
    # -----------------------------------------------------------#
    A = box_a.size(0)
    B = box_b.size(0)

    # -----------------------------------------------------------#
    #   计算交的面积
    # -----------------------------------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    # -----------------------------------------------------------#
    #   计算预测框和真实框各自的面积
    # -----------------------------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # -----------------------------------------------------------#
    #   求IOU
    # -----------------------------------------------------------#
    union = area_a + area_b - inter
    return inter / union  # [A,B]


for cls in classes:
    for conf in confs:
        clsgt = [[int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])] for gt in gtboxes if gt[0] == cls]
        clsdr = [[int(dr[1]), int(dr[2]), int(dr[3]), int(dr[4])] for dr in drboxes if (dr[0] == cls and float(dr[5]) > conf * 0.01)]
        gtarr = torch.tensor(clsgt, requires_grad=False)
        drarr = torch.tensor(clsdr, requires_grad=False)
        res[cls][conf]['TP+FP'] += len(drarr)
        res[cls][conf]['TP+FN'] += len(gtarr)
        if (len(drarr) != 0 and len(gtarr) != 0):
            iou_matrix = calculate_iou(gtarr, drarr).numpy()
            iou_matrix[iou_matrix < 0.5] = 0
            TP = 0
            while np.max(iou_matrix) >= 0.5:
                coord = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                iou_matrix[coord[0], :] = 0
                iou_matrix[:, coord[1]] = 0
                TP += 1
            res[cls][conf]['TP'] += TP



def getClassAP(name):
    prec = []
    reca = []

    for conf in reversed(confs):
        prec.append((res[name][conf]['TP'] + 1) / (res[name][conf]['TP+FP'] + 1))
        reca.append((res[name][conf]['TP'] + 1) / (res[name][conf]['TP+FN'] + 1))
    print(prec)
<<<<<<< HEAD
#    print(reca)
#    return voc_ap(reca, prec)


=======
    print(reca)
    return voc_ap(reca, prec)

print(getClassAP('person'))

print(res)

print(voc_ap([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1, 0.9, 0.8, 0.7, 0.5, 0.6, 0.3, 0.2, 0.2]))
>>>>>>> mainwork
