#coding:utf8
import math
import os

import torch

from UNet import *
from sets import *
from dataset_3d import Longi15
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time

############################################################################
def val(model,dataloader):
    model.eval()
    val_losses, dcs = [], []
    for ii, data in enumerate(dataloader): # AverageMeter

        input, label = data
        label[label < 1] = 0
        label[label >= 1] = 1
        val_input = Variable(input.cuda())
        val_label = Variable(label.cuda())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            model = model.cuda()
        outputs=model(val_input)
        pred = outputs.data.max(1)[1].cpu().numpy().squeeze()
        gt = val_label.data.cpu().numpy().squeeze()

        for i in range(gt.shape[0]):
            dc,val_loss=calc_dice(gt[i,:,:,:],pred[i,:,:,:])
            dcs.append(dc)
            val_losses.append(val_loss)

    model.train()
    return np.mean(dcs),np.mean(val_losses)
############################################################################




############################################################################
print('train:')
lr = 0.001
batch_size = 3
print('batch_size:',batch_size,'lr:',lr)

plt_list = []

model = AttU_Net3Dtest()

if opt.use_gpu:
    model.cuda()
train_data=Longi15(opt.train_data_root,train=True)
val_data=Longi15(opt.train_data_root,train=False,val=True)
val_dataloader = DataLoader(val_data,4,shuffle=False,num_workers=opt.num_workers)

criterion = t.nn.CrossEntropyLoss()
if opt.use_gpu: 
    criterion = criterion.cuda()

loss_meter=AverageMeter()
previous_loss = 1e+20

train_dataloader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=opt.num_workers)
optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)

# train
for epoch in range(opt.max_epoch):
    print(epoch)
    loss_meter.reset()
    
    for ii,(data,label) in tqdm(enumerate(train_dataloader),total=math.ceil(len(train_data)/batch_size)):

        input = Variable(data)
        label[label < 1] = 0
        label[label >= 1] = 1
        target = Variable(label)
            
        if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        score = model(input)
        loss = criterion(score,target)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        if ii%50==1:
            plt_list.append(loss_meter.val)
        if ii%50==1:
            print('train-loss-avg:', loss_meter.avg,'train-loss-each:', loss_meter.val)
            
    if epoch == 200 or epoch == 300:
        if 1 > 0:
            acc = 0
            val_loss = 0
            prefix = './code2/check/pth/' + str(acc)+'_4444_'+str(val_loss) + '_'+str(lr)+'_'+str(batch_size)+'_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            t.save(model.state_dict(), name)
            
            name1 = time.strftime('./code2/check/plt/' + '%m%d_%H:%M:%S.npy')
            numpy.save(name1, plt_list)

    print('batch_size:',batch_size,'lr:',lr)

############################################################################




