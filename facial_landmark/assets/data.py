

# class ImgTransforms(object):
#     """
#     图像预处理工具，用于将图像进行升维(96, 96) => (96, 96, 3)，
#     并对图像的维度进行转换从HWC变为CHW
#     """
#     def __init__(self, fmt):
#         self.format = fmt

#     def __call__(self, img):
#         if len(img.shape) == 2:
#             img = np.expand_dims(img, axis=2)
#         img =  img.transpose(self.format)
#         if img.shape[0] == 1:
#             img = np.repeat(img, 3, axis=0)
#         return img

# class FaceDataset(Dataset):
#     def __init__(self, data_path, mode='train', val_split=0.2):
#         self.mode = mode
#         assert self.mode in ['train', 'val', 'test'], \
#             "mode should be 'train' or 'test', but got {}".format(self.mode)
#         self.data_source = pd.read_csv(data_path)
#         # 清洗数据, 数据集中有很多样本只标注了部分关键点, 这里有两种策略
#         # 第一种, 将未标注的位置从上一个样本对应的关键点复制过来
#         # self.data_source.fillna(method = 'ffill',inplace = True)
#         # 第二种, 将包含有未标注的样本从数据集中移除
#         self.data_source.dropna(how="any", inplace=True)  
#         self.data_label_all = self.data_source.drop('Image', axis = 1)
        
#         # 划分训练集和验证集合
#         if self.mode in ['train', 'val']:
#             np.random.seed(43)
#             data_len = len(self.data_source)
 
#             # 随机划分
#             shuffled_indices = np.random.permutation(data_len)
#             # 顺序划分
#             # shuffled_indices = np.arange(data_len)
#             self.shuffled_indices = shuffled_indices
#             val_set_size = int(data_len*val_split)
#             if self.mode == 'val':
#                 val_indices = shuffled_indices[:val_set_size]
#                 self.data_img = self.data_source.reindex().iloc[val_indices]
#                 self.data_label = self.data_label_all.reindex().iloc[val_indices]
#             elif self.mode == 'train':
#                 train_indices = shuffled_indices[val_set_size:]
#                 self.data_img = self.data_source.reindex().iloc[train_indices]
#                 self.data_label = self.data_label_all.reindex().iloc[train_indices]
#         elif self.mode == 'test':
#             self.data_img = self.data_source
#             self.data_label = self.data_label_all

#         self.transforms = transforms.Compose([
#             ImgTransforms((2, 0, 1))
#         ])

#     # 每次迭代时返回数据和对应的标签
#     def __getitem__(self, idx):

#         img = self.data_img['Image'].iloc[idx].split(' ')
#         img = ['0' if x == '' else x for x in img]
#         img = np.array(img, dtype = 'float32').reshape(96, 96)
#         img = self.transforms(img)
#         label = np.array(self.data_label.iloc[idx,:],dtype = 'float32')/96
#         return img, label

#     # 返回整个数据集的总数
#     def __len__(self):
#         return len(self.data_img)
    
    
# # Train_Dir = './assets/data/data60/training.csv'
# # Test_Dir = './assets/data/data60/test.csv'
# # lookid_dir = './assets/data/data60/IdLookupTable.csv'
# # train_dataset = FaceDataset(Train_Dir, mode='train')
# # val_dataset = FaceDataset(Train_Dir, mode='val')
# # test_dataset = FaceDataset(Test_Dir,  mode='test')
# # print(train_dataset.__len__())

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class myDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=True,
                 target_width = None, target_height = None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_width = target_width
        self.target_height = target_height

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0])+'.png')
        image = Image.open(img_path).convert('RGB')
        # print(image.shape)
        label = self.img_labels.iloc[idx, 1:]
        label = np.array([item.split(' ') for item in label], dtype = 'float32')
        

        if self.target_transform:
            for idx, item in enumerate(label):
                item[0] = (self.target_width/image.width)*item[0]
                item[1] = (self.target_height/image.height)*item[1]
                label[idx] = item
            label = torch.tensor(label.ravel())
            # print(label)
        if self.transform:
            image = self.transform(image)
        return image, label

# annotation_file = './assets/lfpw/trainset/annotation.csv'
# img_dir = './assets/lfpw/trainset'
# data = myDataset(annotation_file, img_dir)
# data.__getitem__(0)