import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# 입력 파일로부터 데이터 불러오기
X = np.loadtxt('data_clustering.txt', delimiter=',')

# X의 대역폭(데이터의 너비) 추정
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# MeanShift를 사용하여 데이터 클러스터링
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# 클러스터 중심점 추출
cluster_centers = meanshift_model.cluster_centers_
print('\n클러스터 중심점:\n', cluster_centers)

# 클러스터 수 추정
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\n입력 데이터의 클러스터 수 =", num_clusters)

# 데이터 포인트와 클러스터 중심점 플롯
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # 현재 클러스터에 속하는 데이터 포인트 플롯
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')

    # 클러스터 중심점 플롯
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black',
             markersize=15)

plt.title('클러스터')
plt.show()
