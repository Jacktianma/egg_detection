import os
import json

def modify_labels_in_json(json_folder, new_label='egg'):
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)

            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 修改 shapes 中每个标注的 label 字段
            if 'shapes' in data:
                for shape in data['shapes']:
                    shape['label'] = new_label

            # 保存修改后的 JSON 文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"已处理: {filename}")

# 修改为你的实际路径
json_folder_path = 'photo'  # 👉 替换为你的 JSON 文件夹路径
modify_labels_in_json(json_folder_path)
