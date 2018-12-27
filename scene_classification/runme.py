import pandas as pd
train_csv = pd.read_csv('train.csv', delimiter= ",", encoding='utf-8', header=None)
train_csv = train_csv[1:]
 train_csv.head()
train_csv[1].head()
from matplotlib import pyplot as plt
from collections import Counter
Counter(train_csv[1])
count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
train_final = []
train_csv.shape
for i in range(17034):
    if(train_csv.iloc[i][1] == '0' and count_0<200):
        count_0+=1
        train_final.append([train_csv.iloc[i][0], train_csv.iloc[i][1]])
    if(train_csv.iloc[i][1] == '1' and count_1<200):
        count_1+=1
        train_final.append([train_csv.iloc[i][0], train_csv.iloc[i][1]])
    if(train_csv.iloc[i][1] == '2' and count_2<200):
        count_2+=1
        train_final.append([train_csv.iloc[i][0], train_csv.iloc[i][1]])
    if(train_csv.iloc[i][1] == '3' and count_3<200):
        count_3+=1
        train_final.append([train_csv.iloc[i][0], train_csv.iloc[i][1]])
    if(train_csv.iloc[i][1] == '4' and count_4<200):
        count_4+=1
        train_final.append([train_csv.iloc[i][0], train_csv.iloc[i][1]])
    if(train_csv.iloc[i][1] == '5' and count_5<200):
        count_5+=1
        train_final.append([train_csv.iloc[i][0], train_csv.iloc[i][1]])
import numpy as np
train_final = np.array(train_final)
train_final[:,1]
Counter(train_final[:,1])
import cv2
img = cv2.imread('data/data/0.jpg')
plt.imshow(img)
