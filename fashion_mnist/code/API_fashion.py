import API_base
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def net(input_shape, output_classes, compiled=False):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(128, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(256, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_classes, activation='softmax'))
    if compiled:
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                      metrics=['accuracy'])
    return model


class Model_Wrapper(API_base.Model_Wrapper_Base):

    def add_reserve_class(self):
        # the model must support adding one new output class.
        # the added output serves as reserve_class

        model = self.model

        # copy the original weights, to keep the predicting function same
        weights_bak = model.layers[-1].get_weights()
        num_classes = model.layers[-1].output_shape[-1]
        model.pop()
        model.add(Dense(num_classes + 1, activation='softmax'))
        # model.summary()
        weights_new = model.layers[-1].get_weights()
        weights_new[0][:, :-1] = weights_bak[0]
        weights_new[1][:-1] = weights_bak[1]

        # use the average weight to init the last. This suppress its output, while keeping performance.
        weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
        weights_new[1][-1] = np.mean(weights_bak[1])

        model.layers[-1].set_weights(weights_new)

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                      metrics=['accuracy'])
        self.reserved = True
        self.model = model

    def load_from_file(self, path):

        self.num_classes = len(self.class_name_list)
        self.model = net((28, 28, 1), self.num_classes)
        self.model.load_weights(path)
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                           metrics=['accuracy'])

    def fit(self, X, y):
        # lazy add
        if self.reserved is False:
            if y.shape[1] > self.num_classes:
                self.add_reserve_class()
        self.model.fit(X, y, batch_size=128, epochs=1, verbose=0)


def load_fashion(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class Data_Wrapper(API_base.Data_Wrapper_Base):

    def load(self, data_path, class_list, class_ratio=None):
        self.data = {}
        X, Y = load_fashion(data_path, kind='train')
        Xt, Yt = load_fashion(data_path, kind='t10k')

        X, Y = self.slice_by_class(X, Y, class_list, class_ratio)
        Xt, Yt = self.slice_by_class(Xt, Yt, class_list, class_ratio)

        unique_labels = list(set(Y))
        n_classes = len(unique_labels)
        from collections import Counter
        print(Counter(Y))
        Y = self.to_categorical(Y, n_classes)
        Yt = self.to_categorical(Yt, n_classes)
        X = X.astype('float32') / 255.
        Xt = Xt.astype('float32') / 255.
        X = np.reshape(X, (len(X), 28, 28, 1))
        Xt = np.reshape(Xt, (len(Xt), 28, 28, 1))
        self.data['X_train'] = X
        self.data['X_test'] = Xt
        self.data['y_train'] = Y
        self.data['y_test'] = Yt


if __name__ == "__main__":
    model = Model_Wrapper('fashion_0123', ['0', '1', '2', '3'])
    import os

    base_path = os.getcwd()
    path = os.path.join(base_path, '..', 'models', 'fashion_0123.h5')
    model.load_from_file(path)
    model.model.summary()
