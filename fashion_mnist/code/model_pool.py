import numpy as np
from importlib import import_module
from config_parser import parse_config


class Model_Pool():
    """
    This class works as an full functionable unit,
    wrapping the low-level models and datasets,
    to support the CalibRe procedure
    """

    def __init__(self, config_filename):
        """initialize from config file"""
        self.construct_from_config(config_filename)

    def construct_from_config(self, config_filename):
        """
        Read and parse the config file,
        load the models and datasets
        """
        config = parse_config(config_filename)
        self.API_list = []
        for API_name in config['API_list']:
            API = import_module(API_name)
            self.API_list.append(API)

        self.names = config['names']

        self.class_name_list = config['class_name_list']
        self.num_classes = len(self.class_name_list)
        self.class_name_lists = config['class_name_lists']
        self.local_class_name_lists = config['local_class_name_lists']
        self.class_ratio_list = config['class_ratio_list']

        self.load(config['model_path_list'], config['data_path_list'])
        self.full_label_space_mapping()
        self.data_size = []
        for name in self.names:
            self.data_size.append(self.data_pool[name]['y_train'].shape[0])
        self.full_data_size = sum(self.data_size)
        self.model_weight = np.array(self.data_size) / self.full_data_size
        print(self.data_size)

    def random_model(self):
        return np.random.choice(self.names, p=self.model_weight)

    def full_label_space_mapping(self):
        """
        Count the overlapping classes among different models,
        and create the mapping from local label space to global label space.
        """
        self.mapping = []
        for name in self.names:
            self.mapping.append(self.model_pool[name].class_name_list)

    def local_to_local(self, sender, receiver, Y_sender):
        sender_class = self.model_pool[sender].class_name_list
        receiver_class = self.model_pool[receiver].class_name_list
        Y_receiver = np.zeros((Y_sender.shape[0], len(
            receiver_class) + 1))  # add reserved class
        for idx, class_name in enumerate(sender_class):
            if class_name not in receiver_class:  # should be reserved class
                Y_receiver[:, -1] += Y_sender[:, idx]
            else:
                receiver_idx = receiver_class.index(class_name)
                Y_receiver[:, receiver_idx] = Y_sender[:, idx]
        Y_receiver[Y_receiver[:, -1] >= 1, -1] = 1
        return Y_receiver

    def load(self, model_path_list, data_path_list):
        self.model_pool = {}
        self.data_pool = {}
        for idx, name in enumerate(self.names):
            model = self.API_list[idx].Model_Wrapper(
                name, self.class_name_lists[idx])
            model.load_from_file(model_path_list[idx])
            if self.class_ratio_list is None:
                data = self.API_list[idx].Data_Wrapper(data_path_list[idx],
                                                       self.local_class_name_lists[idx])
            else:
                data = self.API_list[idx].Data_Wrapper(data_path_list[idx],
                                                       self.local_class_name_lists[
                                                           idx],
                                                       self.class_ratio_list[idx])
            self.model_pool[name] = model
            self.data_pool[name] = data.data

    def evaluate_locally(self):
        score = {}
        for name in self.names:
            X_test = self.data_pool[name]['X_test']
            y_test = self.data_pool[name]['y_test']
            accuracy = self.model_pool[name].evaluate(X_test, y_test)
            score[name] = accuracy
        return score

    def local_to_global(self, name, y_local):
        """ transform one-hot local label space matrix to global, padding with 0"""
        predict_value = np.zeros((y_local.shape[0], self.num_classes))
        idx = self.names.index(name)
        predict_value[:, self.mapping[idx]] = y_local
        return predict_value

    def global_class_to_local_class(self, name, y_global_class):
        y_local_class = np.zeros(y_global_class.shape[0], dtype=np.int)
        class_list = self.model_pool[name].class_name_list
        reserved_class = len(class_list)
        reserved_flag = False
        for i, e in enumerate(y_global_class):
            if e in class_list:
                y_local_class[i] = class_list.index(e)
            else:
                reserved_flag = True
                y_local_class[i] = reserved_class
        return y_local_class, reserved_flag

    def predict_proba_each(self, X):
        separated_predict_value = {}
        for idx, name in enumerate(self.names):
            y_local = self.model_pool[name].predict_proba(X)
            y_global = self.local_to_global(name, y_local)
            separated_predict_value[name] = y_global
        return separated_predict_value

    def MPMC_margin(self, x, y):
        """ 
        comput MPMC margin on (x,y)
        y is the correct multi-class label
        """
        i_pos = None
        correct_max = 0
        i_neg = None
        incorrect_max = 0
        x = np.expand_dims(x, axis=0)
        for idx, name in enumerate(self.names):
            y_local = self.model_pool[name].predict_proba(x)
            y_global = np.ravel(self.local_to_global(name, y_local))
            if y_global[y] > correct_max:
                i_pos = name
                correct_max = y_global[y]
            y_global[y] = 0
            max_class = np.argmax(y_global)
            max_proba = y_global[max_class]
            if max_proba > incorrect_max:
                i_neg = name
                incorrect_max = max_proba
        margin = correct_max - incorrect_max
        return (margin, i_pos, i_neg)

    def MPMC_margin_batch(self, X, Y):
        n = Y.shape[0]
        margin = np.zeros(n)
        i_pos = [None] * n
        i_neg = [None] * n
        for i in range(n):
            margin[i], i_pos[i], i_neg[i] = self.MPMC_margin(X[i, :], Y[i])
        return (margin, i_pos, i_neg)

    def predict_proba(self, X):
        n = X.shape[0]
        predict_value = np.zeros((len(self.names), n, self.num_classes))

        for idx, name in enumerate(self.names):
            y_local = self.model_pool[name].predict_proba(X)
            y_global = self.local_to_global(name, y_local)
            predict_value[idx, :, :] = y_global
        # predict_vector = np.zeros((n, self.num_classes))
        predict_vector = np.max(predict_value, axis=0)
        return predict_vector

    def predict(self, X):
        predict_value = self.predict_proba(X)
        return np.argmax(predict_value, axis=1)

    def evaluate(self):
        score_each = {}
        all_correct_cnt = 0
        all_cnt = 0
        for idx, name in enumerate(self.names):
            X_test = self.data_pool[name]['X_test']
            y_test = self.data_pool[name]['y_test']
            # print('test on %s: %d' % (name, len(y_test)))
            y_test_temp = np.argmax(y_test, axis=1)

            y_test_multiclass = np.zeros((len(y_test_temp)))
            for i, e in enumerate(y_test_temp):
                y_test_multiclass[i] = self.mapping[idx][e]
            predict_label = self.predict(X_test)
            correct_cnt = np.sum(predict_label == y_test_multiclass)
            accuracy = correct_cnt / len(y_test_multiclass)
            score_each[name] = accuracy
            all_correct_cnt += correct_cnt
            all_cnt += len(y_test_multiclass)
        score_all = all_correct_cnt / all_cnt
        return score_each, score_all

    def evaluate_on(self, Xt, yt):
        y_pred = self.predict(Xt)
        if len(yt.shape) > 1:
            yt = np.argmax(yt, axis=1)
        accuracy = np.sum(y_pred == yt) / yt.shape[0]
        return accuracy


class Tunnel():

    def __init__(self, names):
        self.names = names
        self.data = {}
        for sender in names:
            self.data[sender] = {}
            for receiver in names:
                self.data[sender][receiver] = []

    def send(self, sender, receiver, x, y):
        self.data[sender][receiver].append((x, y))

    def receive(self, receiver):
        received_X = []
        received_y = []
        for sender in self.names:
            received_X.extend([e[0] for e in self.data[sender][receiver]])
            received_y.extend([e[1] for e in self.data[sender][receiver]])
        received_X = np.array(received_X)
        received_y = np.array(received_y)
        return received_X, received_y


if __name__ == "__main__":
    # demo usage
    # load fashion_mnist model and data, then evaluate them.

    hyper_model = Model_Pool('fashion_config.txt')
    score, score_all = hyper_model.evaluate()
    print('accuracy on each:')
    for name in hyper_model.names:
        print('%20s: %.4f' % (name, score[name]))
    print('accuracy before calibration:')
    print('%20s: %.4f' % ('hyper model', score_all))
