'''
실행이 안 됨 mnist_data 다운 받던 중 문제가 생기는 것 같음
'''
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Build a classifier using MNIST data')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='/Users/eunseo-ko/ECC/AIStudy/ESKAI/ch16/mnist_data', help='Directory for storing data')
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    # MNIST 데이터셋 로드
    mnist, info = tfds.load('mnist', data_dir=args.input_dir, with_info=True, as_supervised=True)
    mnist_train, mnist_test = mnist['train'], mnist['test']

    # 데이터 전처리 함수
    def preprocess(image, label):
        image = tf.reshape(image, [-1]) / 255.0
        label = tf.one_hot(label, depth=10)
        return image, label

    # 배치 및 전처리
    batch_size = 90
    train_dataset = mnist_train.map(preprocess).shuffle(10000).batch(batch_size)
    test_dataset = mnist_test.map(preprocess).batch(batch_size)

    # 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(784,), activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 학습 시작
    model.fit(train_dataset, epochs=10)

    # 정확도 계산
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print('\nAccuracy =', test_accuracy)

