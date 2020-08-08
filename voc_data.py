'''
voc 2007 数据集的处理
'''

import collections
import os
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils.data import Dataset
import matplotlib.image as im
import matplotlib.pyplot as plt
import random
import cv2
import torch
import torchvision.transforms as transforms

class VOCClassSegBase(Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self,root,split='train',transform = False):

        self.root = root
        self.split = split
        self._transform = transform
        self.image_mean = np.array([0.485,0.456,0.406])
        self.image_std = np.array([0.229,0.224,0.225])
        self.tsfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(self.image_mean,self.image_std)])
        # 建立数据集文件路径
        dataset_dir = os.path.join(self.root,'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007')
        self.files = collections.defaultdict(list)
        for split in ['train','val']:
            imageset_file = os.path.join(dataset_dir,'ImageSets/Segmentation/%s.txt' % split)
            with open(imageset_file,'r') as f:
                ids = f.read().splitlines()
            for id in ids:
                img_file = os.path.join(dataset_dir,'JPEGImages/%s.jpg' % id)
                label_file = os.path.join(dataset_dir,'SegmentationClass/%s.png' % id)
                self.files[split].append({'img':img_file,'label':label_file})

    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, item):
        data_file = self.files[self.split][item]
        # load image
        img_file = data_file['img']
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['label']
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl[lbl == 255] = 0

        # 数据增强
        img,lbl = self.augmentation(img,lbl)

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
    # 矩阵左右翻转
    def randomFilp(self,img,label):
        if random.random() < 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        return img,label

    # 线性插值改变大小
    def resize(self,img,label,s = 320):
        img = cv2.resize(img,(s,s),interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label,(s,s),interpolation=cv2.INTER_NEAREST)
        return img,label

    def randomCrop(self,img,label):
        h,w,_ = img.shape
        short_size = min(w,h)
        rand_size = random.randrange(int(0.7*short_size),short_size)
        x = random.randrange(0,w-rand_size)
        y = random.randrange(0,h-rand_size)

        return img[y:y+rand_size,x:x+rand_size],label[y:y+rand_size,x:x+rand_size]

    # 数据增强
    def augmentation(self,img,label):
        img,label = self.randomFilp(img,label)
        img,label = self.randomCrop(img,label)
        img,label = self.resize(img,label)
        return img,label

    def imshow(self,inp,title=None):
        inp = inp.numpy().transpose((1,2,0))
        inp = inp*self.image_std+self.image_mean
        inp = np.clip(inp,0,1)
        plt.imshow(inp)
        plt.pause(0.001)


def VOCClassSegClass(root,transform):
    voc = VOCClassSegBase(root=root,transform = transform)
    dataset = []
    for i in range(voc.__len__()):
        img,label = voc.__getitem__(i)
        dataset.append((img,label))
    return dataset

if __name__ == '__main__':
    root = 'd:/input_data/'
    voc = VOCClassSegBase(root,transform=True)
    img,label = voc.__getitem__(0)
    print("image.shape = ",img.size())
    print("label.shape = ",label.size())
    uimg,ulabel = voc.untransform(img,label)
    for i in range(1,256):
        print(label[label == i])
    plt.imshow(uimg)
    plt.show()
    plt.imshow(ulabel)
    plt.show()









