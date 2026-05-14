import json
import jsonlines


def convert_json_to_jsonl(json_file, jsonl_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding="utf-8") as file:
        data = json.load(file)  # 加载 JSON 数据（假设是一个列表）

    # 写入 JSONL 文件
    with jsonlines.open(jsonl_file, 'w') as writer:
        for item in data:  # 遍历数组中的每个对象
            writer.write(item)  # 将每个对象写入一行


# 定义文件路径

json_file_path= "/home/share/datasets/ioh/VitalDB/processed/timeseries_by_caseids_prerisk/ANS2E_IOH/vitals_ioh_timeline.json"
jsonl_file_path= "/home/share/datasets/ioh/VitalDB/processed/timeseries_by_caseids_prerisk/ANS2E_IOH/vitals_ioh_timeline.jsonl"

# 调用函数进行转换
convert_json_to_jsonl(json_file_path, jsonl_file_path)