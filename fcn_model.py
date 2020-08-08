'''
fcn 模型
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
import numpy as np
from config import ranges,vgg

def get_upsample_weight(in_channels,out_channels,kernel_size):
    '''
    给上采样卷积初始化 二维线性核
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :return:
    '''
    factor = (kernel_size + 1)//2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size,:kernel_size]   # 二维数组 （1x64) (64x1)
    filt = (1-abs(og[0] - center)/factor)* (1-abs(og[1]-center)/factor) # 64x64
    weight = np.zeros((in_channels,out_channels,kernel_size,kernel_size),dtype=np.float64)
    weight[range(in_channels),range(out_channels),:,:] = filt

    return torch.from_numpy(weight).float()


class FCN32S(nn.Module):
    def __init__(self,pretrained_net,n_class):
        super(FCN32S,self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.Relu(inplace = True)

        self.dconv1 = nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,
                                          padding=1,dilation=1,output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.dconv2 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,
                                          padding=1,dilation=1,output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.dconv3 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,
                                         padding=1,dilation=1,output_padding=1)
        self.bn3 =nn.BatchNorm2d(128)

        self.dconv4 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,
                                         padding=1,dilation=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.dconv5 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,
                                        padding=1,dilation=1,output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32,n_class,kernel_size=1)

        # 参数初始化
        nn.init.xavier_uniform_(self.dconv1.weight)
        nn.init.xavier_uniform_(self.dconv2.weight)
        nn.init.xavier_uniform_(self.dconv3.weight)
        nn.init.xavier_uniform_(self.dconv4.weight)
        nn.init.xavier_uniform_(self.dconv5.weight)
        nn.init.xavier_uniform_(self.classifier.weight)


    def forward(self, input):
        output = self.pretrained.forward(input)
        x5 = output['x5']    #size = [n,512,h/32,w/2]
        score = self.bn1(self.relu(self.dconv1(x5)))
        score = self.bn2(self.relu(self.dconv2(score)))
        score = self.bn3(self.relu(self.dconv3(score)))
        score = self.bn4(self.relu(self.dconv4(score)))
        score = self.bn5(self.relu(self.dconv5(score)))
        score = self.classifier(score)

        return score


class FCN16S(nn.Module):
    def __init__(self,pretrained_net,n_class):
        super(FCN16S,self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.dconv1 = nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,
                                          padding=1,dilation=1,output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.dconv2 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,
                                          padding=1,dilation=1,output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.dconv3 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,
                                         padding=1,dilation=1,output_padding=1)
        self.bn3 =nn.BatchNorm2d(128)

        self.dconv4 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,
                                         padding=1,dilation=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.dconv5 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,
                                        padding=1,dilation=1,output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32,n_class,kernel_size=1)

        # 参数初始化
        nn.init.xavier_uniform_(self.dconv1.weight)
        nn.init.xavier_uniform_(self.dconv2.weight)
        nn.init.xavier_uniform_(self.dconv3.weight)
        nn.init.xavier_uniform_(self.dconv4.weight)
        nn.init.xavier_uniform_(self.dconv5.weight)
        nn.init.xavier_uniform_(self.classifier.weight)


    def forward(self, input):
        output = self.pretrained_net.forward(input)
        x5 = output['x5']    # size = [n,512,h/32,w/32]
        x4 = output['x4']    # size = [n,512,h/16,w/16]

        score = self.relu(self.dconv1(x5))
        score = self.bn1(score+x4)
        score = self.bn2(self.relu(self.dconv2(score)))
        score = self.bn3(self.relu(self.dconv3(score)))
        score = self.bn4(self.relu(self.dconv4(score)))
        score = self.bn5(self.relu(self.dconv5(score)))
        score = self.classifier(score)

        return score



class FCN8S(nn.Module):
    def __init__(self,pretrained_net,n_class):
        super(FCN8S,self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace = True)
        self.dconv1 = nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,
                                          padding=1,dilation=1,output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.dconv2 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,
                                          padding=1,dilation=1,output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.dconv3 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,
                                         padding=1,dilation=1,output_padding=1)
        self.bn3 =nn.BatchNorm2d(128)

        self.dconv4 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,
                                         padding=1,dilation=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.dconv5 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,
                                        padding=1,dilation=1,output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32,n_class,kernel_size=1)

        # 参数初始化
        nn.init.xavier_uniform_(self.dconv1.weight)
        nn.init.xavier_uniform_(self.dconv2.weight)
        nn.init.xavier_uniform_(self.dconv3.weight)
        nn.init.xavier_uniform_(self.dconv4.weight)
        nn.init.xavier_uniform_(self.dconv5.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input):
        output = self.pretrained_net.forward(input)
        x5 = output['x5']         # [n,512,h/32,w/32]
        x4 = output['x4']         # [n,512,h/16,w/16]
        x3 = output['x3']         # [n,512,h/8,w/8]

        score = self.relu(self.dconv1(x5))
        score = self.bn1(score+x4)
        score = self.relu(self.dconv2(score))
        score = self.bn2(score+x3)
        score = self.bn3(self.relu(self.dconv3(score)))
        score = self.bn4(self.relu(self.dconv4(score)))
        score = self.bn5(self.relu(self.dconv5(score)))
        score = self.classifier(score)

        return score


def make_layers(cfg,batch_norm = False):
    '''
    vgg 模型建模
    :param cfg:
    :param batch_norm:
    :return:
    '''
    layers = []
    in_channels = 3
    for v in cfg:
        if v =='M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNET(VGG):
    def __init__(self,pretrained = True,model = 'vgg16',requires_grad = True, remove_fc=True, show_params=False):
        super(VGGNET,self).__init__(make_layers(vgg[model]))
        self.ranges = ranges[model]

        if pretrained:
            vgg16 = models.vgg16(pretrained=True)

        if not requires_grad:
            for param in super.parameters():
                param.requires_grad = False

        if remove_fc:
            del self.classifier

        if show_params:
            for name,param in self.named_parameters():
                print(name,param.size())

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0],self.ranges[idx][1]):
                x = self.features[layer](x)
            output['x%d'%(idx+1)] = x
        return output

if __name__ == '__main__':
    VGGNET(show_params=True)







