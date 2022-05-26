"""
Authors: Miriam Rateike, Ayan Majumdar
"""

import argparse
import ast
import os
import pickle
import yaml


def list_intersection(l1, l2):
    out = list(set(l1) & set(l2))
    if len(out) > 0:
        my_type = type(out[0])
        assert all(isinstance(x, my_type) for x in out)
    return out


def list_union(l1, l2):
    out = list(set(l1) | set(l2))
    if len(out) > 0:
        my_type = type(out[0])
        assert all(isinstance(x, my_type) for x in out)
    return out


def list_subtract(l, l_substact):
    out = list(set(l) - set(l_substact))
    if len(out) > 0:
        my_type = type(out[0])
        assert all(isinstance(x, my_type) for x in out)
    return out


def to_str(elem):
    if isinstance(elem, list):
        return '_'.join([str(s) for s in elem])
    else:
        return str(elem)


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split("+"):
            k, v = kv.split("=")
            if isinstance(v, str):
                if k not in ['missing_perc', 'features_s', 'features_e', 'features_l']:
                    try:
                        v = ast.literal_eval(v)
                    except:
                        pass
                else:
                    try:
                        vi = ast.literal_eval(v)
                    except:
                        vi = 2
                        pass
                    v = vi if vi < 1.0 else v

            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def newest(path):
    if not os.path.exists(path):
        return None
    files = os.listdir(path)
    if len(files) == 0:
        return None
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def save_yaml(yaml_object, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(yaml_object, yaml_file, default_flow_style=False)

    print(f'Saving yaml: {file_path}')
    return


def save_obj(filename_no_extension, obj, ext='.pkl'):
    with open(filename_no_extension + ext, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def parse_args(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return cfg


def flatten_cfg(cfg):
    cfg_flat = {}
    for key, value in cfg.items():
        if not isinstance(value, dict):
            cfg_flat[key] = value
        else:
            for key2, value2 in value.items():
                if not isinstance(value2, dict):
                    cfg_flat[f'{key}_{key2}'] = value2
                else:
                    for key3, value3 in value2.items():
                        cfg_flat[f'{key}_{key2}_{key3}'] = value3

    return cfg_flat


def get_experiment_folder(cfg):
    dataset_params = '_'.join([f"{to_str(v)}" for k, v in cfg['dataset']['params2'].items()])

    _cfg = cfg['model']['params'].copy()
    if 'phase1-ckpt' in _cfg.keys():
        del _cfg['phase1-ckpt']
    if 'phase1-hparams' in _cfg.keys():
        del _cfg['phase1-hparams']

    model_params = '_'.join([f"{to_str(v)}" for k, v in _cfg.items()])
    optim_params = '_'.join([f"{to_str(v)}" for k, v in cfg['optimizer']['params'].items()])
    trainer1_params = '_'.join([f"{to_str(v)}" for k, v in cfg['trainer1'].items()])
    trainer2_params = '_'.join([f"{to_str(v)}" for k, v in cfg['trainer2'].items()])

    save_dir = os.path.join(f"{cfg['dataset']['name']}_{dataset_params}",
                            cfg['model']['name'] + '_' +
                            model_params, cfg['optimizer']['name'] + '_' +
                            optim_params + '_phase1_' + trainer1_params + '_phase2_' + trainer2_params)

    return save_dir
