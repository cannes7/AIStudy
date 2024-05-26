'''
실행이 안 됨
'''
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Build a CNN classifier \
            using MNIST data')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
            default='./mnist_data', help='Directory for storing data')
    return parser

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    # MNIST 데이터셋 로드
    mnist, info = tfds.load('mnist', data_dir=args.input_dir, with_info=True, as_supervised=True)
    mnist_train, mnist_test = mnist['train'], mnist['test']

    # 데이터 전처리 함수
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return tf.image.resize(image, (28, 28)), tf.one_hot(label, depth=10)

    # 배치 및 전처리
    batch_size = 75
    train_dataset = mnist_train.map(preprocess).shuffle(10000).batch(batch_size)
    test_dataset = mnist_test.map(preprocess).batch(batch_size)

    # 모델 생성
    model = create_model()

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 학습 시작
    model.fit(train_dataset, epochs=10)

    # 정확도 계산
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print('\nTest Accuracy =', test_accuracy)
