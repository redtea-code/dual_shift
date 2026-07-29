"""
V3 配置加载工具。
"""
import yaml
import inspect


def load_config(file_path):
    """加载 YAML 配置，将列表转为元组。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        for key in config.keys():
            if type(config[key]) == list:
                config[key] = tuple(config[key])
        return config


def get_parameters(fn, original_dict):
    """过滤 dict 中与函数签名匹配的键。"""
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict


def is_cross_val_enabled(config: dict) -> bool:
    """检查 config 中是否启用了交叉验证。"""
    return config.get('cross_val', {}).get('enabled', False)
