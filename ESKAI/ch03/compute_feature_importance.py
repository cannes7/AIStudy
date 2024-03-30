import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import cross_validation
from sklearn.utils import shuffle

# 주택 데이터 로드
housing_data = datasets.load_boston()

# 데이터 셔플
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# 데이터를 훈련 및 테스트 데이터로 분할
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=7)

# 아다부스트 회귀 분류기 모델 생성
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                              n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# 아다부스트 회귀 분류기 성능 평가
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# 특성 중요도 추출
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# 중요도 값 정규화
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# 값 정렬 및 뒤집기
index_sorted = np.flipud(np.argsort(feature_importances))

# X축 틱 정렬
pos = np.arange(index_sorted.shape[0]) + 0.5

# 막대 그래프 플롯
plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Feature importance using AdaBoost regressor')
plt.show()
