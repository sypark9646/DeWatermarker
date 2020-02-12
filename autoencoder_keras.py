import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping


with open("C:/Users/soyeon/Desktop/DeWatermarker-master/data/new/pickled_data.pkl", "br") as fh:
    data = pickle.load(fh)

watermarked_images = data[0].reshape(7, 3984768)
original_images = data[1].reshape(7, 3984768) 
x_train_noisy, x_test_noisy, x_train, x_test = train_test_split(watermarked_images, original_images, test_size=0.2)

print("watermarked shape", watermarked_images.shape)
print("original shape", original_images.shape)

#모형 구축
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=3984768))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3984768, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

#체크포인트
ckpt = ModelCheckpoint('./model/dewatermark.h5', save_best_only=True, monitor = 'val_loss', mode = 'min')
#학습 자동 중단
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

#학습
model.fit(x_train_noisy, x_train, 
          epochs=200,
          batch_size=1,
          shuffle=True,
          callbacks=[ckpt, early_stopping],
          validation_data=(x_test_noisy, x_test))

model.save('./model/dewatermark.h5', save_format = 'h5')
