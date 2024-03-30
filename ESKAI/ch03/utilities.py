import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=''):
    # X와 Y의 최소 및 최대값 정의
    # 메시 그리드에 사용될 최소 및 최대값
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # 플롯에 사용할 메시 그리드의 단계 크기 정의
    mesh_step_size = 0.01

    # X 및 Y 값의 메시 그리드 정의
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # 메시 그리드에서 분류기 실행
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # 출력 배열의 모양을 변경
    output = output.reshape(x_vals.shape)

    # 플롯 생성
    plt.figure()

    # 제목 지정
    plt.title(title)

    # 플롯에 대한 색상 체계 선택
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # 플롯 위에 훈련 포인트를 오버레이함...
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # 플롯의 경계 지정
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # X 및 Y 축의 눈금 지정
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()
