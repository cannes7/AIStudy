import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# 입력 데이터 로드
X = np.loadtxt('data_clustering.txt', delimiter=',')

# 클러스터 개수 설정
num_clusters = 5

# 입력 데이터 플로팅
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none',
            edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# KMeans 객체 생성
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

# KMeans 클러스터링 모델 학습
kmeans.fit(X)

# 메쉬의 단계 크기 설정
step_size = 0.01

# 경계를 플로팅할 그리드 포인트 정의
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

# 그리드 상의 모든 포인트에 대한 출력 레이블 예측
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

# 다른 영역을 플로팅하고 색칠
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
                   y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')

# 입력 점 겹쳐 플로팅
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none',
            edgecolors='black', s=80)

# 클러스터 중심 플로팅
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
            marker='o', s=210, linewidths=4, color='black',
            zorder=12, facecolors='black')

# 플롯 제목 및 축 범위 설정
plt.title('Boundaries of clusters')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
