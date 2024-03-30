score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))

print("\n클러스터 수 =", num_clusters)
print("실루엣 점수 =", score)

scores.append(score)

# 실루엣 점수 플롯
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('클러스터 수 대 실루엣 점수')

# 최적의 점수와 클러스터 수 추출
num_clusters = np.argmax(scores) + values[0]
print('\n최적의 클러스터 수 =', num_clusters)

# 데이터 플롯
plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('입력 데이터')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
