import torch
import torchvision

def VGG16(pretrained= True):
    vgg16_model = torchvision.models.vgg16(pretrained)
    if not pretrained:
        return vgg16_model
    return vgg16_model