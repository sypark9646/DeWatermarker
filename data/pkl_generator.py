import cv2
import os
import pickle

mylist = list()

mydictionary = {}
mydictionary['watermarked'] = cv2.imread("C:/Users/soyeon/Desktop/DeWatermarker-master/data/new/watermarked/2.tif", 0)
mydictionary['original'] = cv2.imread("C:/Users/soyeon/Desktop/DeWatermarker-master/data/new/original/2.tif", 0)

mydictionary['watermarked']=mydictionary['watermarked'].reshape(2306, 1728, 1)
mydictionary['original']=mydictionary['original'].reshape(2306, 1728, 1)
#print(mydictionary['watermarked'].shape) #(2306, 1728, 1)

mylist.append(mydictionary)

with open('C:/Users/soyeon/Desktop/DeWatermarker-master/data/training/set.pkl', 'wb') as f:
    pickle.dump(mylist, f)