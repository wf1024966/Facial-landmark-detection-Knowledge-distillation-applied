import os # 读取图片
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
image = cv2.imread('./assets/lfpw/trainset/image_0001.png', 1)  # 1 彩色，0 灰色
 
 
 
df = pd.DataFrame(index=None)

# 读取对应的pts文件
for j in range(1,240):
    if os.path.exists('./assets/lfpw/testset/image_%04d.pts'%j):
        with open('./assets/lfpw/testset/image_%04d.pts'%j) as file_obj:
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
        landmarks = [str(item).replace(',', '')[1:-1] for item in landmarks]#将(x,y)转换为字符串x y
        df['image_%04d'%j] = landmarks
    else:
        continue
df = df.T
print(df.head())
df.to_csv('./assets/lfpw/testset/annotation_test.csv',  header=False)

# print(landmarks[1])  # (83.8168, 220.22)
# # 测试读出的点有多少个
# c=0
# for b in landmarks:
#     c+=1;
# print(c) #68
 
 
#  将关键点标在图片上
# '''
# cv2.circle(image, center_coordinates, radius, color, thickness)
# '''
# m = 0  # 标号初始为0
# for point in landmarks:
#     # print(point[0],point[1])
#     cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)  # 颜色顺序：BGR (0, 255, 0)绿色,-1 实心圆
#     m += 1
#     cv2.putText(image, str(m), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255),
#                 1)  # 每个关键点上标号
# #     plt.scatter(np.transpose(point)[0], np.transpose(point)[1])  # 散点图
# # plt.show()
# cv2.imshow("pointImg", image)
# cv2.waitKey()