'''
조건부 랜덤 필드로 알파벳 문자열 예측하기
- 조건부 랜덤 필드(CRF, Conditional Random Field):
    구조화된 데이터 분석에 자주 사용되는 확률 모델
    다양한 형태의 순차적 데이터를 레이블링하고 분류함
    주요 특징: 판별 모델(discriminative model)임
    <- 생성 모델(generative model)인 HMM과는 대조적(그리고 조건부 확률 분포가 아닌 결합 분포를 이용함)
    레이블 값이 주어진 일련의 데이터가 있다면 조건부 확률 분포 정의 가능
    조건부적인 모델이기에 연속된 예측 결과 간의 독립성을 가정하지 않음
    실제 환경의 데이터가 시간적으로 의존성을 가지는 경우에 더 적합한 모델
    자연어 처리, 음성 인식, 생명공학 등 다양한 응용 분야에서 HMM보다 우수한 성능을 보이는 경향이 있음
    e.g., OCR(광학 문자 인식):
    CRF를 사용하여 알파벳 문자열을 예측하는 방법
    이 데이터셋은 문자열의 첫 글자가 모두 삭제되어 있으며,
    이는 OCR 데이터에서 첫 글자가 대문자로 표현되어 소문자와 다른 형태를 가지기 때문에 제외된 것
'''

import os
import argparse 
import string
import pickle 

import numpy as np
import matplotlib.pyplot as plt
# pystruct 임포트 오류 문제...
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF 
from pystruct.learners import FrankWolfeSSVM

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains a Conditional\
            Random Field classifier')
    parser.add_argument("--C", dest="c_val", required=False, type=float,
            default=1.0, help='C value to be used for training')
    return parser

# Class to model the CRF
class CRFModel(object):
    def __init__(self, c_val=1.0):
        self.clf = FrankWolfeSSVM(model=ChainCRF(), 
                C=c_val, max_iter=50) 

    # Load the training data
    def load_data(self):
        alphabets = load_letters()
        X = np.array(alphabets['data'])
        y = np.array(alphabets['labels'])
        folds = alphabets['folds']

        return X, y, folds

    # Train the CRF
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    # Evaluate the accuracy of the CRF
    def evaluate(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    # Run the CRF on unknown data
    def classify(self, input_data):
        return self.clf.predict(input_data)[0]

# Convert indices to alphabets
def convert_to_letters(indices):
    # Create a numpy array of all alphabets
    alphabets = np.array(list(string.ascii_lowercase))

    # Extract the letters based on input indices
    output = np.take(alphabets, indices)
    output = ''.join(output)

    return output

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    c_val = args.c_val

    # Create the CRF model
    crf = CRFModel(c_val)

    # Load the train and test data
    X, y, folds = crf.load_data()
    X_train, X_test = X[folds == 1], X[folds != 1]
    y_train, y_test = y[folds == 1], y[folds != 1]

    # Train the CRF model
    print('\nTraining the CRF model...')
    crf.train(X_train, y_train)

    # Evaluate the accuracy
    score = crf.evaluate(X_test, y_test)
    print('\nAccuracy score =', str(round(score*100, 2)) + '%')

    indices = range(3000, len(y_test), 200)
    for index in indices:
        print("\nOriginal  =", convert_to_letters(y_test[index]))
        predicted = crf.classify([X_test[index]])
        print("Predicted =", convert_to_letters(predicted))