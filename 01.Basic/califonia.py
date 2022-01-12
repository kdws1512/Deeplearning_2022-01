# import part
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 상수값 설정 등 변수 초기화
seed = 2022
warnings.filterwarnings('ignore')
np.random.seed(seed)
tf.random.set_seed(seed)
house = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    house.data, house.target, test_size=0.1, random_state=seed
)

# 메인 모델 만들기
model = Sequential([
    Dense(20, input_dim=8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
hist = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=128)

# 입력값 받기