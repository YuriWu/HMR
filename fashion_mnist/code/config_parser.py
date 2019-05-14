import ast
import os


def parse_config(config_filename):
    config = {}
    file_content = {}
    with open(config_filename) as f:
        for line in f:
            line = line.strip()
            line = line.replace(' ', '')
            idx = line.find(':')
            file_content[line[:idx]] = line[idx + 1:]

    config['API_list'] = file_content['API_list'].split(',')
    config['names'] = file_content['model_names'].split(',')
    config['model_path_list'] = []
    base_path = file_content['base_path']
    for path in file_content['model_files'].split(','):
        config['model_path_list'].append(os.path.join(base_path, 'model', path))

    config['data_path_list'] = []
    for path in file_content['data_files'].split(','):
        config['data_path_list'].append(os.path.join(base_path, 'data', path))

    config['class_name_list'] = ast.literal_eval(file_content['all_class_names'])
    config['num_classes'] = len(config['class_name_list'])
    config['class_name_lists'] = ast.literal_eval(file_content['class_name_lists'])
    config['local_class_name_lists'] = ast.literal_eval(file_content['local_class_name_lists'])
    if 'local_class_ratio_lists' in file_content.keys():
        config['class_ratio_list'] = ast.literal_eval(
            file_content['local_class_ratio_lists'])
    else:
        config['class_ratio_list'] = None
    return config


if __name__ == "__main__":
    c = parse_config(os.path.join('..', 'config', '2p_A.txt'))
