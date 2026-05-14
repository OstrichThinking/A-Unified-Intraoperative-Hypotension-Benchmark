import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
import argparse

def round_data_to_two_decimals(data):
    return np.where(np.isnan(data), data, np.round(data, 2))

def timestamp_to_datetime(timestamp: int) -> str:
    """将秒级时间戳转换为ISO格式时间字符串 (YYYY-MM-DDTHH:MM:SS.000Z)。"""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def datetime_to_timestamp(date_str: str) -> int:
    """将ISO格式时间字符串 (YYYY-MM-DDTHH:MM:SS.000Z) 转换为秒级时间戳。"""
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.000Z")
    return int(dt.timestamp())

def load_config(dataset_name):
    """
    加载配置文件。

    参数:
    dataset_name (str): 数据集名称。如果为 "all"，则加载所有配置文件；否则，加载与数据集名称匹配的配置文件。

    返回:
    dict: 包含配置文件内容的字典，其中键是配置文件的名称（不含扩展名），值是配置文件的内容。
    """
    config_dir = Path(__file__).parent / "config"
    config_dic = {}
    if dataset_name == "all":
        # 读取config目录下的所有配置文件
        for config_file in config_dir.glob("*.yaml"):
            with open(config_file, 'r', encoding='utf-8') as file:
                config_dic[config_file.stem] = yaml.safe_load(file)
    else:
        # 搜索并读取具有dataset_name字符的配置文件
        for config_file in config_dir.glob(f"*{dataset_name}*.yaml"):
            with open(config_file, 'r', encoding='utf-8') as file:
                config_dic[config_file.stem] = yaml.safe_load(file)
    return config_dic

def get_config(dataset_name):
    config_dic = load_config(dataset_name)
    return config_dic[f'{dataset_name}']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process VitalDB IOH dataset.')
    parser.add_argument('--dataset_name', type=str, default='invasive_group_test', help='Name of the dataset')
    return parser.parse_args()