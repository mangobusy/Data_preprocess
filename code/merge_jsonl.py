import json
import re
import os

# 1. 定义输入和输出文件
input_files = ['/data/Shizihui/Data_preprocess/LJSpeech/other/train/ljspeech_train_audio_tokens.jsonl', 
                '/data/Shizihui/Data_preprocess/LJSpeech/other/val/ljspeech_val_audio_tokens.jsonl', 
                '/data/Shizihui/Data_preprocess/LJSpeech/other/test/ljspeech_test_audio_tokens.jsonl']
output_file = '/data/Shizihui/Data_preprocess/LJSpeech/LJSpeech_data.jsonl'

data_list = []

print("正在读取数据...")

# 2. 读取所有文件
for file_path in input_files:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    data_list.append(item)
                except json.JSONDecodeError:
                    print(f"警告: 跳过无法解析的行 in {file_path}")
    else:
        print(f"警告: 文件 {file_path} 不存在，跳过。")

print(f"共读取到 {len(data_list)} 条数据。正在排序...")

# 3. 定义排序函数
def sort_key_func(item):
    # 提取 key，例如 "LJ050-0161"
    key_str = item.get('key', '')
    
    # 使用正则提取故事编号和句子编号
    # 匹配 LJ(数字)-(数字)
    match = re.match(r"LJ(\d+)-(\d+)", key_str)
    
    if match:
        story_num = int(match.group(1)) # 50
        sent_num = int(match.group(2))  # 161
        return (story_num, sent_num)
    else:
        # 如果格式不对，这就放到最后，或者按字符串排序
        return (999999, key_str)

# 4. 执行排序
# 这会按照 (故事ID, 句子ID) 的数字大小顺序排列
data_list.sort(key=sort_key_func)

# 5. 写入结果
print(f"正在写入 {output_file} ...")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data_list:
        # ensure_ascii=False 保证非ASCII字符（如果有）原样输出，不转义
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("完成！")