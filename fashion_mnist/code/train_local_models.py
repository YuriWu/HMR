import os
import sys
from config_parser import parse_config
from importlib import import_module

setting = sys.argv[1]
config = parse_config(os.path.join('..', 'config', '%s.txt' % setting))

for idx, name in enumerate(config['names']):
    data_path = config['data_path_list'][idx]
    model_path = config['model_path_list'][idx]
    class_list = config['local_class_name_lists'][idx]
    API = import_module(config['API_list'][idx])
    if config['class_ratio_list'] is None:
        data = API.Data_Wrapper(data_path, class_list)
    else:
        class_ratio_list = config['class_ratio_list'][idx]
        data = API.Data_Wrapper(data_path, class_list, class_ratio_list)

    X = data.data['X_train']
    Xt = data.data['X_test']
    y = data.data['y_train']
    yt = data.data['y_test']
    model = API.net((28, 28, 1), len(class_list), compiled=True)
    model.fit(X, y, epochs=20, batch_size=64, validation_data=(Xt, yt))
    model.save_weights(model_path)
    print('%s trained and saved' % name)
