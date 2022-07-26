# 读取图片
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch 
from model.model import *

 
# 读取对应的pts文件
def getLandmark(pts_dir):
    with open(pts_dir) as file_obj:
        contents = file_obj.readlines()
    
    i = 0
    landmarks = []
    for line in contents:
        TT = line.strip("\n")  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        if i > 2 and i < 71:
            # print TT
            TT_temp = TT.split(" ")
            x = float(TT_temp[0])
            y = float(TT_temp[1].strip("\r"))  # \r :回车
            landmarks.append((x, y))
        i += 1
    return landmarks
 
 
#  将关键点标在图片上
def drawLandmarks(landmarks, image, video = False):
    m = 0  # 标号初始为0
    for point in landmarks:
        # print(point[0],point[1])
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)  # 颜色顺序：BGR (0, 255, 0)绿色,-1 实心圆
        m += 1
        cv2.putText(image, str(m), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255),
                    1)  # 每个关键点上标号
    #     plt.scatter(np.transpose(point)[0], np.transpose(point)[1])  # 散点图
    # plt.show()
    return image
        


if __name__ == '__main__':
    img_path = r'D:\VSCodeFiles\LipDetect\knowledge_distillation\pic.jpg'
    
    #    image = cv2.imread(img_path, 1)  # 1 彩色，0 灰色
    cap = cv2.VideoCapture(0)
    while 1:
        ret, image = cap.read()
        image_draw = image    
        model = FaceNet(68).cuda()
        model.load_state_dict(torch.load('./weights/model_9.pth'))
        
        image = cv2.resize(image, (256, 256))
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image).unsqueeze(0).float()
        image = image.cuda()
        pred = model(image)
        pred = pred.cpu().detach().numpy().ravel()
        
        coord = []
    #    image_draw = cv2.imread(img_path, 1)  # 1 彩色，0 灰色
        for (x,y) in zip(pred[0::2], pred[1::2]):
            x = x*image_draw.shape[1]/256
            y = y*image_draw.shape[0]/256
            coord.append([x,y])
        result = drawLandmarks(coord, image_draw, video=True)
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break