import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 생성할 데이터 포인트의 수 정의
num_points = 1200

# y = mx + c 방정식 기반으로 데이터 생성
data = []
m = 0.2
c = 0.5
for i in range(num_points):
    # 'x' 생성
    x = np.random.normal(0.0, 0.8)

    # 약간의 노이즈 생성
    noise = np.random.normal(0.0, 0.04)

    # 'y' 계산
    y = m * x + c + noise

    data.append([x, y])

# x와 y를 분리
x_data = np.array([d[0] for d in data])
y_data = np.array([d[1] for d in data])

# 생성된 데이터 플로팅
plt.plot(x_data, y_data, 'ro')
plt.title('입력 데이터')
plt.show()

# 가중치와 편향 정의
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# 최적화기 정의
optimizer = tf.optimizers.SGD(learning_rate=0.5)

# 학습 단계 함수
def train_step():
    with tf.GradientTape() as tape:
        y_pred = W * x_data + b
        loss = tf.reduce_mean(tf.square(y_pred - y_data))
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss

# 학습 루프
num_iterations = 10
for step in range(num_iterations):
    loss_value = train_step()

    # 진행 상황 출력
    print('\n반복', step + 1)
    print('W =', W.numpy()[0])
    print('b =', b.numpy()[0])
    print('손실 =', loss_value.numpy())

    # 입력 데이터 플로팅
    plt.plot(x_data, y_data, 'ro')

    # 예측된 출력 라인 플로팅
    plt.plot(x_data, W.numpy() * x_data + b.numpy())

    # 플로팅 매개변수 설정
    plt.xlabel('차원 0')
    plt.ylabel('차원 1')
    plt.title('반복 ' + str(step + 1) + ' / ' + str(num_iterations))
    plt.show()
