""" Training augmented model """
import os
import numpy as np
#from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils
from config import AugmentConfig
from model import Video3D
from RATnet import RATNet
from data.dataset import train_data,test_data
from loss import Segloss

config = AugmentConfig()

os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda")

# logger 
#writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
#writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.workers,
                                            pin_memory=True)

def Dice(output, target, eps=1e-3):
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

def main():
    logger.info("Logger is set - training start")
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    # initial
    model = Video3D(RATNet('basic', [2, 2, 2, 2]), 32, 4, 4)
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    losser = Segloss(n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    best_dice = 0.

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # training loop
    for epoch in range(config.epochs):
        # training
        train(train_loader, model, optimizer, losser, epoch)
        # validation
        val_dice = validate(test_data, model, epoch)
        # save
        if best_dice < val_dice:
            best_dice = val_dice
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        lr_scheduler.step()
        print("")

    logger.info("Final best Prec@1 = {:.3f}".format(best_dice))


def train(train_loader, model, optimizer, loss1, epoch):
    losses1 = utils.AverageMeter()
    losses_total = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    #writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (image0, image1, image2, label0, label1, label2) in enumerate(train_loader):
        image0 = Variable(image0.to(device, non_blocking=True))
        image1 = Variable(image1.to(device, non_blocking=True))
        image2 = Variable(image2.to(device, non_blocking=True))
        label0 = Variable(label0.to(device, non_blocking=True))
        label1 = Variable(label1.to(device, non_blocking=True))
        label2 = Variable(label2.to(device, non_blocking=True))
        N = label2.size(0)
        optimizer.zero_grad()

        predict = model([image0, image1, image2])
        seg_loss = (loss1(predict[0], label0) + loss1(predict[1], label1) + loss1(predict[2], label2) + loss1(predict[3], label0) + loss1(predict[4], label1) + loss1(predict[5], label2)) / 6

        loss_total = seg_loss
        loss_total.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        losses1.update(seg_loss.item(), N)
        losses_total.update(loss_total.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Seg_loss {:.3f}"
                .format(epoch+1, config.epochs, step, len(train_loader)-1, losses1.avg))

        #writer.add_scalar('train/loss', loss.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Loss@1 {:.3f}".format(epoch+1, config.epochs, losses1.avg))


def validate(test_data, model, epoch):
    dicees = utils.AverageMeter()
    
    model.eval()

    with torch.no_grad():
        for step, (image0, image1, image2, label0, label1, label2, BF) in enumerate(test_data):
            image0 = Variable(image0.to(device, non_blocking=True))
            image1 = Variable(image1.to(device, non_blocking=True))
            image2 = Variable(image2.to(device, non_blocking=True))
            label2 = Variable(label2.to(device, non_blocking=True))
            N = label2.size(0)

            predict = model([image0, image1, image2])
            dice = cal_dice(predict[-1],label2)

            dicees.update(dice.item(), N)

    #writer.add_scalar('val/loss', losses.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Dice: {:.3f}".format(epoch+1, config.epochs, dicees.avg))

    return dicees.avg
           
if __name__ == "__main__":
    main()
