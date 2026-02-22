
import json
import os

# 配置文件路径
input_file = '/data/Shizihui/Data_preprocess/Blizzard/Blizzard_test.jsonl'       
output_file = '/data/Shizihui/Data_preprocess/Blizzard/Blizzard_test1.jsonl'  

def process_dataset():
    # 打开输入文件和输出文件
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        print(f"开始处理 {input_file} ...")
        
        count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                # 1. 将每一行 JSON 字符串转为 Python 字典
                data = json.loads(line)
                # 2. 获取 key 的值
                key_value = data.get('key', '')
                # 3. 拼接字符串
                # 注意：通常路径层级之间需要斜杠。
                # 如果你的意图是 .../EN/文件名.wav，请保留下面的斜杠 '/'
                # 如果你的意图是 .../EN文件名.wav (EN是前缀)，请去掉下面的斜杠
                path_prefix = "/Data_preprocess/audio/EN/" 
                
                data['target_wav'] = f"{path_prefix}{key_value}.wav"
                
                # 4. 将处理后的字典转回 JSON 字符串并写入文件
                # ensure_ascii=False 保证中文或其他字符正常显示
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
                
            except json.JSONDecodeError:
                print(f"跳过格式错误的行: {line[:50]}...")
                continue

    print(f"处理完成！共处理了 {count} 条数据。")
    print(f"结果已保存至: {output_file}")

if __name__ == '__main__':
    # 确保你把 data.jsonl 放在和脚本同一个目录下，或者修改上面的 input_file 路径
    process_dataset()
"""

import json

input_path = "/data/Shizihui/Data_preprocess/Total/EN/librispeech_test_clean.jsonl"
output_path = "/data/Shizihui/Data_preprocess/Total/EN/librispeech_test_clean1.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)

        # 新增字段（如果不存在）
        if "answer_cosyvoice_speech_token" not in item:
            item["answer_cosyvoice_speech_token"] = []

        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
"""