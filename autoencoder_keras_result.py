import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import pickle

# 모델 load
model = load_model('./model/dewatermark.h5')

# 결과 확인
watermarked_img = cv2.imread("C:/Users/soyeon/Desktop/DeWatermarker-master/data/new/watermarked/1.tiff" ,0)
dewatermarked_img = model.predict(watermarked_img)
dewatermarked_img = cv2.cvtcolor(dewatermarked_img cv2.color_gray2bgr)
cv2.imwrite("C:/Users/soyeon/Desktop/DeWatermarker-master/data/new/watermarked/1_dewatermarked.tiff", dewatermarked_img)
