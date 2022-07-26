import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from assets import data
from model.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据集路径
train_annotation_dir = './assets/lfpw/trainset/annotation.csv'
test_annotation_dir = './assets/lfpw/testset/annotation_test.csv'
train_img_dir = './assets/lfpw/trainset/'
test_img_dir = './assets/lfpw/testset/'

# 简单图像预处理
transformer = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((256,256)),
    # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



#训练、验证和测试数据集
train_dataset = data.myDataset(train_annotation_dir, train_img_dir, transform=transformer,
                               target_width=256, target_height=256)
test_dataset = data.myDataset(test_annotation_dir, test_img_dir, transform=transformer,
                               target_width=256, target_height=256)

#转换为dataloader 
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)



# 训练模型
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss




if __name__ == "__main__":
    #定义模型、优化器和损失函数
    model = FaceNet(num_keypoints=68).cuda()
    print(summary(model, input_size=(3,256,256)))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    epoch = 10
    writer = SummaryWriter('./logs')
    #训练&保存模型
    for i in range(epoch):
        print('epoch:', i)
        loss_train = train(train_dataloader, model, loss, optimizer)
        writer.add_scalar('loss/train', loss_train, i)
        torch.save(model.state_dict(), './weights/model_' + str(i) + '.pth')
        

