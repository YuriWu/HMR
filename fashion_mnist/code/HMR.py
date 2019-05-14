from model_pool import Model_Pool, Tunnel
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from API_fashion import Data_Wrapper
from exp_recorder import Exp_Recorder
import sys
import os

setting = sys.argv[1]
loop_cnt = int(sys.argv[2])
np.random.seed(loop_cnt)
tf.set_random_seed(loop_cnt)

# load local models from configs
hyper_model = Model_Pool(os.path.join('..', 'config', '%s.txt' % setting))
save_filepath = os.path.join('..', 'exp', '%s_%d.pkl' % (setting, loop_cnt))

names = hyper_model.names
print('Following local models are loaded: ', end='')
print(','.join(names))

# show the performance before calibration
score_local = hyper_model.evaluate_locally()
print('accuracy on local data:')
for name in names:
    print('%20s: %.4f' % (name, score_local[name]))


fashion_data = Data_Wrapper(os.path.join('..', 'data', 'fashion_mnist'), list(range(10)))
fashion_data = fashion_data.data

# recorder to save the experimental results
recorder = Exp_Recorder(names)
Xt = fashion_data['X_test']
Yt = fashion_data['y_test']
score_global = hyper_model.evaluate_on(Xt, Yt)
recorder.add_record(score_local, score_global)
recorder.print()

# tunnel stores exchanged examples
tunnel = Tunnel(names)


def calibrate(hyper_model, tunnel, receiver):
    """ the calibrate operation """
    X = hyper_model.data_pool[receiver]['X_train']
    y = hyper_model.data_pool[receiver]['y_train']
    n, m = y.shape
    received_X, received_y = tunnel.receive(receiver)
    received_y, reserved_flag = hyper_model.global_class_to_local_class(receiver, received_y)
    if reserved_flag:
        received_y = to_categorical(received_y, m + 1)
        # pad zeros at last
        y = np.concatenate((y, np.zeros((n, 1))), axis=1)
    else:
        received_y = to_categorical(received_y, m)
    X_augmented = np.concatenate((X, received_X), axis=0)
    y_augmented = np.concatenate((y, received_y), axis=0)
    hyper_model.model_pool[receiver].fit(X_augmented, y_augmented)


recorder.step_saver = []
counter = 0
time_str = '0.00 min'
for step in range(0, 10000):
    if counter >= 105:
        break
    random_sender = hyper_model.random_model()
    sender_local_data = hyper_model.data_pool[random_sender]
    X = sender_local_data['X_train']
    Y = sender_local_data['y_train']
    random_example_idx = np.random.randint(X.shape[0])
    x = X[random_example_idx, :]
    y = hyper_model.local_to_global(random_sender, Y[random_example_idx, :])
    y = np.argmax(y)  # one-hot to multi-class
    margin, pos_receiver, neg_receiver = hyper_model.MPMC_margin(x, y)
    if margin <= 0:  # violated
        print('sender: %-6s, pos: %-6s, neg: %-6s, class: %d, margin: %.6f, step:%d/%d, time: %s' %
              (random_sender, pos_receiver, neg_receiver, y, margin, counter, step, time_str))

        tunnel.send(random_sender, pos_receiver, x, y)
        tunnel.send(random_sender, neg_receiver, x, y)
        calibrate(hyper_model, tunnel, pos_receiver)
        calibrate(hyper_model, tunnel, neg_receiver)
        score_local = hyper_model.evaluate_locally()
        score_global = hyper_model.evaluate_on(Xt, Yt)
        recorder.add_record(score_local, score_global)
        time_str = recorder.print(1)
        recorder.step_saver.append((counter, step))
        recorder.save(save_filepath)
        counter += 1
    else:
        continue
