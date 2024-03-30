import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn import datasets
from sklearn.mixture import GMM
from sklearn.cross_validation import StratifiedKFold

# 붓꽃 데이터셋 로드
iris = datasets.load_iris()

# 데이터를 학습과 테스트로 나누기 (80/20 비율)
indices = StratifiedKFold(iris.target, n_folds=5)
train_index, test_index = next(iter(indices))

# 학습 데이터와 레이블 추출
X_train = iris.data[train_index]
y_train = iris.target[train_index]

# 테스트 데이터와 레이블 추출
X_test = iris.data[test_index]
y_test = iris.target[test_index]

# 클래스의 수 추출
num_classes = len(np.unique(y_train))

# GMM 모델 생성
classifier = GMM(n_components=num_classes, covariance_type='full', init_params='wc', n_iter=20)

# GMM 모델의 평균 초기화
classifier.means_ = np.array([X_train[y_train == i].mean(axis=0) for i in range(num_classes)])

# GMM 분류기 학습
classifier.fit(X_train)

# 경계 그리기
plt.figure()
colors = 'bgr'
for i, color in enumerate(colors):
    # 고유값과 고유벡터 추출
    eigenvalues, eigenvectors = np.linalg.eigh(classifier._get_covars()[i][:2, :2])

    # 첫 번째 고유벡터를 정규화
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])

    # 기울기 각도 추출
    angle = np.arctan2(norm_vec[1], norm_vec[0])
    angle = 180 * angle / np.pi

    # 타원체를 확대하기 위한 스케일링 요소
    # (우리의 필요에 맞는 임의의 값 선택)
    scaling_factor = 8
    eigenvalues *= scaling_factor

    # 타원 그리기
    ellipse = patches.Ellipse(classifier.means_[i, :2], eigenvalues[0], eigenvalues[1], 180 + angle, color=color)
    axis_handle = plt.subplot(1, 1, 1)
    ellipse.set_clip_box(axis_handle.bbox)
    ellipse.set_alpha(0.6)
    axis_handle.add_artist(ellipse)

# 데이터 플롯
colors = 'bgr'
for i, color in enumerate(colors):
    cur_data = iris.data[iris.target == i]
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker='o', facecolors='none', edgecolors='black', s=40,
                label=iris.target_names[i])
    test_data = X_test[y_test == i]
    plt.scatter(test_data[:, 0], test_data[:, 1], marker='s', facecolors='black', edgecolors='black', s=40,
                label=iris.target_names[i])

# 학습 및 테스트 데이터에 대한 예측 계산
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('학습 데이터의 정확도 =', accuracy_training)

y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print('테스트 데이터의 정확도 =', accuracy_testing)

plt.title('GMM 분류기')
plt.xticks(())
plt.yticks(())

plt.show()
