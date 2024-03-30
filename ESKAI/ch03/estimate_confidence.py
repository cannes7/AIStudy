import numpy as np

# Test datapoints 정의
test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

# 각 데이터 포인트의 신뢰도 계산 및 출력
print("\nConfidence measure:")
for datapoint in test_datapoints:
    probabilities = classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print('\nDatapoint:', datapoint)
    print('Predicted class:', predicted_class)

# 데이터포인트 시각화
visualize_classifier(classifier, test_datapoints,
                     [0] * len(test_datapoints),
                     'Test datapoints')

plt.show()
