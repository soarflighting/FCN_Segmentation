'''
工具类
'''
import os
import numpy as np
from PIL import Image

def getPalette():
    pal = np.array([[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]], dtype='uint8').flatten()
    return pal

def colorize_mask(mask):
    '''

    :param mask: 图片大小的数值，代表不同的颜色
    :return:
    '''
    new_mask = Image.fromarray(mask.astype(np.uint8),'P')    # 将二维数组转化为图像

    pal = getPalette()
    new_mask.putpalette(pal)

    return new_mask


def getFileName(file_path):
    '''
    get file_path name from path+name+'test.jpg'
    :param file_path:
    :return:
    '''
    full_name = file_path.split('/')[-1]
    name = os.path.splitext(full_name)[0]

    return name


def label2png(label,img_name):
    '''
    转换label 到 png 图片
    :param label:
    :param img_name:
    :return:
    '''
    label = label.numpy()
    label_pil = colorize_mask(label)
    label_pil.save(img_name)
    return label_pil


def label2img(label):
    label = label.numpy()
    label_pil = colorize_mask(label)
    return label_pil


def _fast_hist(label_true,label_pred,n_class):
    mask = (label_true > 0) & (label_true < n_class)
    hist = np.bincount(n_class*label_true[mask].astype(int)+label_pred[mask],
                       minlength=n_class**2).reshape(n_class,n_class)

    return hist

def accuracy_score(label_trues,label_preds,n_class = 21):
    '''

    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    hist = np.zeros((n_class,n_class))
    for lt,lp in zip(label_trues,label_preds):
        hist += _fast_hist(lt.flatten(),lp.flatten(),n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist)/ hist.sum(axis = 1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis = 1) + hist.sum(axis = 0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis = 1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc,acc_cls,mean_iu,fwavacc






