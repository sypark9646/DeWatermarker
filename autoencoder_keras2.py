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

#모형 구축 https://keraskorea.github.io/posts/2018-10-23-keras_autoencoder/
input_img = Input(shape=(2306, 1728, 1))  # 'channels_first'이미지 데이터 형식을 사용하는 경우 이를 적용

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 이 시점에서 표현(representation)은 (7,7,32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#체크포인트
ckpt = ModelCheckpoint('./model/dewatermark2.h5', save_best_only=True, monitor = 'val_loss', mode = 'min')
#학습 자동 중단
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
print(autoencoder.summary())

autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=1,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[ckpt, early_stopping])

autoencoder.save('./model/dewatermark2.h5', save_format = 'h5')