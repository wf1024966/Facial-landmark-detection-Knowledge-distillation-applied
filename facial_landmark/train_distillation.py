import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

import numpy as np 
import pandas as pd 

from assets import data
from model.model import *
from model.distillater import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#训练、验证和测试数据集
def load_data(train_annotation_dir, test_annotation_dir, train_img_dir, test_img_dir, transformer):
    train_dataset = data.myDataset(train_annotation_dir, train_img_dir, transform=transformer,
                                target_width=256, target_height=256)
    test_dataset = data.myDataset(test_annotation_dir, test_img_dir, transform=transformer,
                                target_width=256, target_height=256)

    #转换为dataloader 
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    return train_dataloader, test_dataloader

# 计算soft_loss, hard_loss，并回传distller的总loss函数
def distilling_loss(student_model, teacher_model,X,y, alpha):
    student_output = student_model(X)
    teacher_output = teacher_model(X)
    distiller = Distiller(gt_label=y, student_output = student_output, 
                          teacher_output=teacher_output, alpha=alpha)
    soft_loss = distiller.soft_loss(student_output=student_output, 
                                    teacher_output=teacher_output)
    hard_loss = distiller.hard_loss(student_output=student_output,
                                    gt_label=y)
    return soft_loss, hard_loss, distiller.loss_fn

# 训练模型
def train(dataloader, student_model, teacher_model, optimizer):
    size = len(dataloader.dataset)
    student_model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        # pred = model(X)
        soft_loss, hard_loss, loss_fn = distilling_loss(student_model, teacher_model, 
                                                        X, y, alpha=0.5)
        loss = loss_fn(soft_loss=soft_loss, hard_loss=hard_loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss



if __name__ == '__main__':
    # 加载数据集
    # 数据集路径
    train_annotation_dir = './facial_landmark/assets/lfpw/trainset/annotation.csv'
    test_annotation_dir = './facial_landmark/assets/lfpw/testset/annotation_test.csv'
    train_img_dir = './facial_landmark/assets/lfpw/trainset/'
    test_img_dir = './facial_landmark/assets/lfpw/testset/'

    # 简单图像预处理
    transformer = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((256,256)),
        # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataloader, test_dataloader = load_data(train_annotation_dir, test_annotation_dir, 
                                                  train_img_dir, test_img_dir, transformer)
    
    
    student_model = FaceNet(num_keypoints=68).to(device)
    teacher_model = FaceNet(num_keypoints=68).to(device)
    teacher_model.load_state_dict(torch.load('./facial_landmark/weights/model_10.pth'))
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    epoch = 10
    for i in range(epoch):
        loss = train(train_dataloader, student_model, teacher_model, optimizer)
        print(f"epoch: {i+1}/{epoch} loss: {loss:>7f}")

    