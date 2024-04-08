#coding:utf8
from UNet import *
from sets import *
import torch as t
from tqdm import tqdm
import numpy
import time
import os
import nibabel as nib
from findcpoint import findcpoint_3D

### python userhome/GUOXUTAO/2021_13/NET00/KDself/16/test_timee.py --path userhome/GUOXUTAO/2021_13/NET00/KDself/16/ --check_list check05 check06 check07

import argparse

def get_parser():
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('--path', default = "default", help='save path')
    parser.add_argument('--check_list', nargs='+')
    args = parser.parse_args()
    
    return args

# python ./code2/test_time_3D_Attcenter.py --path ./code2/ --check_list check

config = get_parser()

root_path = config.path
check_list = config.check_list
print(root_path)
print(check_list)

###
###
###


for check_path in check_list:
    path = root_path + check_path + '/'

    if os.path.exists(path+'dice_time'):
        print('ok')
    else:
        os.mkdir(path+'dice_time')

    val_dice = []
    val_std = []
    check_ing_path = path + 'pth/'

    check_list = sorted(os.listdir(check_ing_path),key=lambda x: os.path.getmtime(os.path.join(check_ing_path, x)))
    check_list.reverse() 
    print(len(check_list))
    read_list = os.listdir(path + 'dice_time/')

    for index,checkname in enumerate(check_list):
        print(index,checkname)

        if checkname != 'un' and checkname != '01':

            model = AttU_Net3Dtest()
            #model.eval()
            model.load_state_dict(torch.load(check_ing_path+checkname))
            model.eval()

            if opt.use_gpu: model.cuda()


            with torch.no_grad():   # if 1 > 0:

                testpath = './DATA/test/'
                path_img = os.path.join(testpath, "images")
                path_label = os.path.join(testpath, "mask")
                path_surf = os.path.join(testpath, "surf_gr")
                folderlist1 = os.listdir(path_img)
                folderlist2 = os.listdir(path_label)
                folderlist3 = os.listdir(path_surf)
                folderlist1.sort()
                folderlist2.sort()
                folderlist3.sort()

                WT_dice_3Dlist = []

                for fodernum in range(len(folderlist1)):

                    a = path_img+'/'+folderlist1[fodernum]
                    b = path_label+'/'+folderlist2[fodernum]
                    c = path_surf+'/'+folderlist3[fodernum]
                    data_image = nib.load(a).get_fdata()
                    data_label = nib.load(b).get_fdata()
                    data_surf = nib.load(c).get_fdata()
                    affine = nib.load(b).affine
                    hdr = nib.load(b).header
                    data = list()
                    data.append(data_image[np.newaxis, :])
                    data.append(data_label[np.newaxis, :])
                    data.append(data_surf[np.newaxis, :])
                    data = np.concatenate(data, axis=0)
                    data = np.asarray(data)
                    pre = np.zeros(data.shape)
                    pre = pre[0]
                    print(pre.shape)

                    prob = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))
                    g = 0
                    s0 = 32
                    s1 = 48
                    ss = 144
                    sss = 144

                    vector = data[[0,2], :, :, :].astype(float)

                    xc, yc, zc = findcpoint_3D(data[[0, 2], :, :, :])
                    print(zc)
                    if xc < 72 or xc + 72 > data.shape[1]:
                        xc = int(data.shape[1] / 2)
                    if yc < 72 or yc + 72 > data.shape[2]:
                        yc = int(data.shape[2] / 2)
                    if zc < 72 or zc + 72 > data.shape[3]:
                        zc = int(data.shape[3] / 2)
                    img_out = vector[:, xc-72:xc+72, yc-72:yc+72, zc-72:zc+72]
                    img = torch.from_numpy(img_out).float()
                    img = torch.unsqueeze(img, 0)
                    with torch.no_grad():
                        input = t.autograd.Variable(img)
                    if True: input = input.cuda()

                    score = model(input)
                    score = torch.nn.Softmax(dim=1)(score).squeeze().detach().cpu().numpy()

                    prob[:, xc-72:xc+72, yc-72:yc+72, zc-72:zc+72] = prob[:, xc-72:xc+72, yc-72:yc+72, zc-72:zc+72] + score

                    label = np.argmax((prob).astype(float),axis=0)
                    print(label.shape)
                    pre[:, :, :] = label
                    print(np.sum(label))

                    tru = data[1, :, :, :]
                    preg = pre
                    trug = tru

                    pre = np.zeros(preg.shape)
                    tru = np.zeros(trug.shape)
                    print(np.sum(preg))
                    pre[preg>=1] = 1
                    tru[trug>=1] = 1
                    a1 = np.sum(pre==1)
                    print(a1)
                    a2 = np.sum(tru==1)
                    print(a2)
                    a3 = np.sum(np.multiply(pre,tru)==1)
                    if a1+a2 > 0:
                        WT_Dice = (2.0*a3)/(a1 + a2)

                    print(WT_Dice)
                    WT_dice_3Dlist.append(WT_Dice)

                    name = os.path.basename(a)
                    name = name.split(".")[0]
                    print(name)
                    pre_int16 = pre.astype(np.uint16)
                    img_nii = nib.Nifti1Image(pre_int16, affine, hdr)
                    nii_path = path + 'output2/'
                    out1_nii_name = os.path.join(nii_path, str(name) + "_out" + ".nii.gz")
                    nib.save(img_nii, out1_nii_name)

                ### mean
                mean_WT_dice = np.mean(WT_dice_3Dlist)

                print('mean  ', 'WT:', mean_WT_dice)

                ### std
                std_WT_dice = np.std(WT_dice_3Dlist)

                print('std  ', 'WT:', std_WT_dice)

                val_dice.append(mean_WT_dice)
                val_std.append(std_WT_dice)

                if not os.path.exists(path + 'dice_time/'+checkname+'/'):
                    os.mkdir(path + 'dice_time/'+checkname+'/')
                np.save(path + 'dice_time/'+checkname+'/'+'dice.npy',WT_dice_3Dlist)

        np.save(path + 'dice_time/val_dice.npy',val_dice)
        np.save(path + 'dice_time/val_std.npy',val_std)
            
print('over!')
