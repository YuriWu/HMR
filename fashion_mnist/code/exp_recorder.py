import pickle
import copy
import time
import os


class Exp_Recorder():
    """ record the experimental results
        for pretty prints and save function"""

    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in names:
            self.record[name] = []
        self.record['global'] = []
        self.start_time = time.time()

    def add_record(self, score_local, score):
        for name in self.names:
            self.record[name].append(score_local[name])

        self.record['global'].append(score)

    def print(self, last=None):
        columns = len(self.names) + 1
        steps = len(self.record['global'])
        string_format = 'step ' + '%-12.11s' * columns

        names_print = copy.copy(self.names)
        names_print.extend(['global acc'])
        print(string_format % tuple(names_print))
        if last is None:
            last = steps
        for i in range(max(steps - last, 0), steps):
            temp = []
            for name in self.names:
                temp.append(self.record[name][i])
            temp.append(self.record['global'][i])
            print('%-5d' % i, end='')
            print('%-12.4f' * columns % tuple(temp))

        elapse = time.time() - self.start_time
        time_str = '%.2f min' % (elapse / 60)
        return time_str

    def load(self, filepath):
        tmp_dict = pickle.load(open(filepath, 'rb'))
        self.__dict__.update(tmp_dict)

    def save(self, filepath):
        pickle.dump(self.__dict__, open(filepath, 'wb'))
