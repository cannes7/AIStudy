'''
MNIST는 손글씨 숫자 데이터셋으로, 6만 개의 훈련 이미지와 1만 개의 테스트 이미지로 구성되어 있다.

- 데이터 로딩 및 전처리: 데이터셋 로드, 탐색 및 훈련/테스트 데이터 준비
- 신경망 모델: 이미지 분류를 위한 Sequential 모델 구성
- 모델 컴파일 및 훈련: 손실 함수와 옵티마이저를 설정하여 모델 컴파일하고 훈련
- 평가: 훈련 및 테스트 데이터셋에서 모델 성능 평가
- 모델 저장 및 로드: 학습된 모델을 파일에 저장하고 필요할 때 로드
- 예측: 새 이미지에 대해 학습된 모델을 사용하여 숫자 예측

'''
# Import necessary libraries
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from PIL import Image
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("Number of training images: %d" % (X_train.shape[0]))
print("Number of test images: %d" % (X_test.shape[0]))

# Display the first image in the training dataset in grayscale
plt.imshow(X_train[0], cmap='Greys')
plt.show()

# Print all the pixel values of the first image in the training dataset
for x in X_train[0]:
    for i in x:
        print('%d  ' % i, end='')
    print()

# Prepare data for training
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# Define the neural network model
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=2)

# Print training accuracy and validation accuracy
print('\nAccuracy: {:.4f}'.format(model.evaluate(X_train, Y_train)[1]))
print('\nVal_Accuracy: {:.4f}'.format(model.evaluate(X_test, Y_test)[1]))

# Save the model
model.save('Predict_Model.h5')

# Load the model
model = load_model('Predict_Model.h5')

# Load and preprocess the test image (test.png)
img = Image.open("test.png").convert("L")
img = np.resize(img, (1, 784))
test_data = ((np.array(img) / 255) - 1) * -1

# Predict the class of the test image
res = np.argmax(model.predict(test_data), axis=-1)
print(res)
