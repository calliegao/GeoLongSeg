#!/usr/bin/env python
#coding:utf8
import nibabel as nib
import os
#import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import  transforms as T
from PIL import Image
from torch.utils import data
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from findcpoint import findcpoint_3D

###
class Longi15(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False, val = False):
        self.test = test
        self.train = train
        self.val = val

        if self.train:
            self.root = './DATA/train'

            self.folderlist1 = os.listdir(os.path.join(self.root, "images"))
            self.folderlist2 = os.listdir(os.path.join(self.root, "mask"))
            self.folderlist3 = os.listdir(os.path.join(self.root, "mask_surf"))
        elif self.val:
            self.root = ''
        elif self.test:
            self.root = './DATA/test/'
            self.folderlist = os.listdir(os.path.join(self.root))

    def __getitem__(self,index):
          
        if self.train:                            
            if 1 > 0 :
                ss = 72
                sss = 72
                ssss = 72
                path_img = os.path.join(self.root, "images")
                path_label = os.path.join(self.root, "mask")
                path_surf = os.path.join(self.root, "mask_surf")
                self.folderlist1.sort()
                self.folderlist2.sort()
                self.folderlist3.sort()
                a = os.path.join(path_img,self.folderlist1[index])
                b = os.path.join(path_label,self.folderlist2[index])
                c = os.path.join(path_surf, self.folderlist3[index])
                data_image = nib.load(a).get_fdata()
                data_label = nib.load(b).get_fdata()
                data_surf = nib.load(c).get_fdata()

                # 升维合并
                data_temp = list()
                data_temp.append(data_image[np.newaxis, :])
                data_temp.append(data_label[np.newaxis, :])
                data_temp.append(data_surf[np.newaxis, :])

                img = np.concatenate(data_temp, axis=0)
                img = np.asarray(img)

                index_x = np.random.randint(ss,img.shape[1]-ss,size=1)
                index_y = np.random.randint(sss,img.shape[2]-sss,size=1)
                index_z = np.random.randint(ssss,img.shape[3]-ssss,size=1)

                img_in = img[:,index_x[0]-ss:index_x[0]+ss,index_y[0]-sss:index_y[0]+sss,index_z[0]-ssss:index_z[0]+ssss]
                img_out = img_in[[0,2],:,:,:].astype(float)
                label_out = img_in[1,:,:,:].astype(float)

                img = torch.from_numpy(img_out).float()
                label = torch.from_numpy(label_out).long()
        else:
            print('###$$$$$$$$$$$$$$$$$$$^^^^^^^^^^^^^')     

        return img, label

    def __len__(self):
        return len(self.folderlist1) if len(self.folderlist1) == len(self.folderlist2) == len(self.folderlist3) else -1







