import numpy as np


class Model_Wrapper_Base():

    def __init__(self, model_name, class_name_list):
        self.name = model_name
        self.class_name_list = class_name_list
        self.num_classes = len(class_name_list)
        self.reserved = False
        self.model = None

    def add_reserve_class(self):
        """ the model must support adding one new output class.
            the added output serves as reserve_class."""
        pass

    def load_from_file(self, path):
        """ load the model from file into self.model
            should add reserve class at the end """
        pass
        self.add_reserve_class()

    def predict_proba(self, X, remove_reserved_class=True):
        """ predict probability 
            self.model.predict() should predict probability"""
        if self.reserved and remove_reserved_class:
            return self.model.predict(X)[:, :-1]
        else:
            return self.model.predict(X)

    def predict_class(self, X, remove_reserved_class=True):
        predicted_value = self.predict_proba(X, remove_reserved_class)
        return np.argmax(predicted_value, axis=1)

    def evaluate(self, X, y):
        predicted_label = self.predict_class(X)
        if len(y.shape) == 2:  # one-hot encoded label
            y_multiclass = np.argmax(y, axis=1)
            return np.sum(predicted_label == y_multiclass) / len(y_multiclass)
        else:  # multi-class label
            return np.sum(predicted_label == y) / len(y)

    def loss_per_instance(self, y_pred, y_true):
        """ cross entropy loss per instance """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        ce = -np.sum(y_true * np.log(y_pred + 1e-9), axis=1)
        return ce

    def check_to_add_reserved_class(self, y):
        if self.reserved == False:
            if y.shape[1] > self.num_classes:
                self.add_reserve_class()

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, kwargs)


class Data_Wrapper_Base():

    def slice_by_class(self, X, y, class_list, class_ratio=None):
        """ supporting function, for slice by class"""
        idx = np.full((len(y)), False)
        if class_ratio == None:
            for e in class_list:
                idx[(y == e)] = True
        else:
            for i, e in enumerate(class_list):
                ratio = class_ratio[i]
                idx_selected = np.where(y == e)[0]
                start = int(np.round(idx_selected.shape[0] * ratio[0]))
                end = int(np.round(idx_selected.shape[0] * ratio[1]))
                idx[idx_selected[start:end]] = True

        y_temp = y[idx]
        X_part = X[idx, :]

        y_part = np.zeros((len(y_temp)))
        class_idx = list(np.unique(y_temp))
        class_idx.sort()
        for class_name in class_idx:
            y_part[y_temp == class_name] = class_idx.index(class_name)

        return X_part, y_part

    def to_categorical(self, Y, n_classes):
        from keras.utils import to_categorical
        return to_categorical(Y, n_classes)

    def load(self, data_path, class_list):
        """
        Should load data into a dict like this
        self.data['X_train'] = X
        self.data['X_test'] = Xt
        self.data['y_train'] = Y
        self.data['y_test'] = Yt
        """
        pass

    def __init__(self, data_path, class_list, class_ratio_list=None):
        self.data_path = data_path
        self.class_list = class_list
        self.data = {}
        self.load(data_path, class_list, class_ratio_list)
