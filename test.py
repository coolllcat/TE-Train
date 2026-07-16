import os
import time
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from data.dataset import test_data
from model import Video3D
from RATnet import RATNet
from config import AugmentConfig
from segout import segout, segout3D
# from indicator import get_evaluation_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda")

config = AugmentConfig()
model = Video3D(RATNet('basic', [2, 2, 2, 2]), 32, 4, 4)
model.load_state_dict(torch.load(os.path.join(config.path,'best_m.pth.tar')).module.state_dict())
model.to(device)
model.eval()

def Dice(output, target, eps=1e-6):
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice

def cal_dice(output, target):
    """
    0代表背景, 1代表前景
    """
    output = torch.argmax(output,dim=1)
    target = target[:,0,:,:,:]
    dice = Dice((output == 1).float(), (target == 1).float())
    return dice

score_all = 0
score_list = []

for step, (image0, image1, image2, label0, label1, label2, BF) in enumerate(test_data):
    image0 = Variable(image0.to(device, non_blocking=True))
    image1 = Variable(image1.to(device, non_blocking=True))
    image2 = Variable(image2.to(device, non_blocking=True))
    label2 = Variable(label2.to(device, non_blocking=True))
    start = time.time()
    # predict = model([image0, image1, image2])
    # dice = cal_dice(predict[-1],label2)
    predict = model.model(image2)
    logits = F.softmax(predict, dim=1)
    segment = logits.data.max(1)[1]
    dice = cal_dice(predict,label2)
    print(time.time()-start)

    # segout(BF[2], np.array(segment.squeeze().cpu()), np.array(label2.squeeze().cpu()), step)
    segout3D(BF[2], np.array(segment.squeeze().cpu()), np.array(label2.squeeze().cpu()), step)

    score_all += dice
    score_list.append(dice)

score_mean = score_all/test_data.__len__()
print(score_mean)
with open("test.txt","w") as f:
    for i in range(len(score_list)):
        f.write(test_data.files[i]['image'][-1].split('/')[-1] + ' ' + str(score_list[i]) + '\n')
