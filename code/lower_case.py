import json
import os

def to_lowercase(input_file, output_file):
    print(f"正在处理: {input_file} ...")
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 1. 解析 JSON
                data = json.loads(line)
                
                # 2. 修改 source_text 为小写
                if "source_text" in data and isinstance(data["source_text"], str):
                    data["source_text"] = data["source_text"].lower()
                    
                # 3. 修改 target_text 为小写
                if "target_text" in data and isinstance(data["target_text"], str):
                    data["target_text"] = data["target_text"].lower()
                
                # 4. 写入新文件
                # ensure_ascii=False 保证如果里面有中文或其他字符，不会被转义成 \uXXXX
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
                
            except json.JSONDecodeError:
                print(f"警告: 跳过无法解析的行: {line[:50]}...")

    print(f"处理完成！已保存到: {output_file}")
    print(f"共处理 {count} 条数据。")

if __name__ == "__main__":
    # --- 请在这里修改你的文件路径 ---
    input_path = "/data/Shizihui/dataset/LibriSpeech/clean/validation/librispeech-val.jsonl"      # 你的原始文件名
    output_path = "/data/Shizihui/Data_preprocess/LibriSpeech/librispeech-val.jsonl" # 转换后输出的文件名
    
    # 检查文件是否存在
    if os.path.exists(input_path):
        to_lowercase(input_path, output_path)
    else:
        print(f"错误: 找不到文件 {input_path}，请修改脚本中的 input_path 路径。")