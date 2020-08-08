'''
预测语义分割
'''
import os
import torch
import cv2
import voc_data
import fcn_model
import random
import fcn_utils


n_class = 21


def main(root):
    '''
    预测主函数
    :param root:  图片路径
    :return:
    '''
    path = os.getcwd() + "/images/"
    dataset = voc_data.VOCClassSegClass(root,transform=True)
    vgg_model = fcn_model.VGGNET()
    f_model = fcn_model.FCN8S(pretrained_net=vgg_model,n_class = n_class)
    f_model.load_state_dict(torch.load('./pretrained_models/modelXXX.pth',map_location='cpu'))

    f_model.eval()

    for i in range(len(dataset)):
        idx = random.randrange(0,len(dataset))
        img,label = dataset[idx]
        img_name = str(i)

        img_src,_ = dataset.untransform(img,label)
        cv2.imwrite(path+'image%s_src.jpg'%img_name,img_src)
        fcn_utils.label2png(label,path+'image/%s_label.png'%img_name)

        out = f_model(img)

        net_out = out.data.max(1)[1].squeeze_(0)

        fcn_utils.label2png(net_out,path + 'image/%s_out.png'%img_name)

        if i == 20:
            break


if __name__ == '__main__':
    print(os.getcwd()+'/images/')
