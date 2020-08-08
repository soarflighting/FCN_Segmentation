'''
训练模型
'''
import os
import torch
import torch.utils.data.dataloader
import numpy as np
import torchvision
import voc_data
from torch.optim import Adam
import fcn_model
import loss_function
from tqdm import tqdm



batch_size = 64
root = 'd:/input_data'
n_class = 21
learning_rate =1e-4
epoch_num = 40

vgg_model = fcn_model.VGGNET()
f_model = fcn_model.FCN8S(pretrained_net=vgg_model,n_class=n_class)

# 损失函数
criterion = loss_function.CrossEntropyLoss_2d()
optimizer = Adam(f_model.parameters(),lr=learning_rate)

def train(epoch):
    f_model.train()
    total_loss = 0
    loss = 0
    for batch_idx,(imgs,labels) in tqdm(enumerate(train_loader),total = len(train_loader),desc='Train epoch=%d' % epoch, ncols=80, leave=False):

        out = f_model(imgs)

        loss = criterion(out,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print('test epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch,
                                                                       epoch_num, batch_idx, len(val_loader),
                                                                       total_loss / (batch_idx + 1)))



    torch.save(f_model.state_dict(), './pretrained_models/model%d.pth'%epoch)  # save for 5 epochs
    total_loss /= len(train_loader)
    print('train epoch [%d] average_loss %.5f' % (epoch, total_loss))


def test(epoch):
    f_model.eval()
    total_loss = 0
    for batch_idx,(imgs,labels) in tqdm(
                enumerate(val_loader), total=len(val_loader),
                desc='Valid iteration=%d' % epoch, ncols=80,leave=False):
        out = f_model(imgs)
        loss = criterion(out,labels)
        total_loss += loss.item()


        if (batch_idx + 1) % 3 == 0:
            print('test epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch,
                                                                       epoch_num, batch_idx, len(val_loader),
                                                                       total_loss / (batch_idx + 1)))

    total_loss /= len(val_loader)
    print('test epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))

    global best_test_loss
    if best_test_loss > total_loss:
        best_test_loss = total_loss




if __name__ == '__main__':
    print('正在加载数据集...')
    train_data = voc_data.VOCClassSegBase(root,split='train',transform = True)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=16,shuffle=True,num_workers=1)

    val_data = voc_data.VOCClassSegBase(root,split='val',transform=True)

    val_loader = torch.utils.data.DataLoader(val_data,batch_size = 64,shuffle=True,num_workers=1)
    print("数据加载完毕...")
    print("开始训练...")
    # print(train_loader.dataset.class_names)
    train(1)



