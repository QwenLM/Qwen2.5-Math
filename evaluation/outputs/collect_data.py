import os
import json

def collect_json_files(directory, output_file):
    collected_data = {}

    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                print(file_path)
                
                # 读取 JSON 文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 将文件路径作为键，内容作为值
                collected_data[file_path] = data

    # 将收集的数据写入一个大的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=4)

# 调用函数，传入 outputs 文件夹路径和输出文件路径
collect_json_files('./', 'collected_data.json')