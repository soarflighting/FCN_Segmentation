'''
对训练结果进行评估
'''

import os
import torch
import fcn_model
import fcn_utils
import voc_data
import numpy as np


n_class = 21
def evaluate(root):

    val_data = voc_data.VOCClassSegBase(root = root,split = 'val',transform = True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size = 64,shuffle =True,nun_workers = 1)

    vgg_model = fcn_model.VGGNET()
    f_model = fcn_model.FCN8S(pretrained_net=vgg_model,n_class=n_class)
    f_model.load_state_dict(torch.load('./pretrained_models/modelXXX.pth',map_location='cpu'))

    f_model.eval()
    label_trues = []
    label_preds = []
    for idx,(imgs,labels) in enumerate(val_loader):

        out = f_model(imgs)
        pred = out.data.max(1)[1].squeeze_(1).squeeze_(0)

        label_preds.append(pred.numpy)
        label_trues.append(labels.numpy())


    metrics = fcn_utils.accuracy_score(label_trues,label_preds)
    metrics.np.array(metrics)
    print('Accuracy: {0}\t Accuracy Class: {1} \t Mean IU: {2} \t FWAV Accuracy: {3}'.format(*metrics))