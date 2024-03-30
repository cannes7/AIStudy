# 의사 결정 트리 모델 구축을 위한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier  # 분류기 시각화 유틸리티

# 입력 데이터 가져오기
input_file = 'data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 레이블에 따라 클래스를 나누기
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

# 입력 데이터 시각화
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
            edgecolors='black', linewidth=1, marker='x', label='Class-0')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='o', label='Class-1')
plt.title('Input data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 데이터 세트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# 의사 결정 트리 모델 생성 및 학습
params = {'random_state': 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)

# 학습 데이터로 분류기 시각화
plt.figure()
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

# 테스트 데이터로 분류기 시각화
plt.figure()
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# 분류기의 성능 평가 및 출력
class_names = ['Class-0', 'Class-1']
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, classifier.predict(X_test), target_names=class_names))
print("#" * 40 + "\n")

plt.show()
