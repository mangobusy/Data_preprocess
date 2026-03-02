import json
import os

def extract_story_id(key):
    """
    智能提取故事的唯一 ID (Context Group ID)
    
    支持格式:
    1. 路径格式: "6097_clean/14411/nada_lily_00_haggard_0024.wav" 
       -> 提取: "6097_clean/14411/nada_lily_00_haggard" (去掉序号和后缀)
       
    2. LibriTTS 格式: "27-124992-0035.wav"
       -> 提取: "27-124992" (同一说话人的同一章节视为一个故事)
       
    3. LJSpeech 格式: "LJ031-0021"
       -> 提取: "LJ031" (同一章节)
       
    4. 其他下划线格式: "xxx_0024"
       -> 提取: "xxx"
    """
    
    # 1. 预处理：去掉 .wav 后缀
    if key.endswith('.wav'):
        clean_key = key[:-4]
    else:
        clean_key = key
        
    # 2. 处理路径格式 (包含斜杠 /)
    # 例如: 6097_clean/14411/nada_lily_00_haggard_0024
    if '/' in clean_key:
        # 去掉最后的 _xxxx 序号部分
        # 假设文件名最后一段总是 _数字 结尾
        if '_' in clean_key:
            return clean_key.rsplit('_', 1)[0]
        return clean_key

    # 3. 处理 LibriTTS 格式 (27-124992-0035)
    # 特征：包含两个连字符 '-'
    parts = clean_key.split('-')
    if len(parts) == 3:
        # 返回 "27-124992" (BookID-ChapterID)
        return f"{parts[0]}-{parts[1]}"

    # 4. 处理 LJSpeech 格式 (LJ031-0021)
    # 特征：包含一个连字符 '-'
    if len(parts) == 2:
        return parts[0]

    # 5. 处理下划线格式 (xxx_0024)
    if '_' in clean_key:
        return clean_key.rsplit('_', 1)[0]

    # 6. 兜底：如果没有特征，就认为它自己就是 ID
    return clean_key

def add_context_to_dataset(input_file, output_file):
    print(f"开始处理文件: {input_file} ...")
    
    prev_story_id = None
    prev_text = ""
    count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析当前行
                data = json.loads(line)
                current_key = data.get('key', '')
                current_text = data.get('source_text', '')
                
                # 提取当前的故事 ID
                current_story_id = extract_story_id(current_key)
                
                # 判断是否属于同一个故事
                # 注意：这里假设输入文件是已经按 Story 排序过的！
                if current_story_id == prev_story_id and current_story_id is not None:
                    data['context'] = prev_text
                else:
                    # 新故事的第一句话，或者无法识别 ID，没有上下文
                    data['context'] = ""
                    
                # 将处理后的数据写入新文件
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # 更新 prev 变量，供下一句使用
                prev_story_id = current_story_id
                prev_text = current_text
                count += 1
                
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无法解析的行: {line[:50]}...")
            except Exception as e:
                print(f"⚠️ 处理出错: {e}, 行内容: {line[:50]}...")

    print(f"✅ 处理完成！共 {count} 条数据。")
    print(f"新数据集已保存至: {output_file}")

# === 使用示例 ===
if __name__ == "__main__":

    # ================= 运行 =================
    # 替换成你实际的文件路径
    input_path = "/data/Shizihui/Data_preprocess/LibriSpeech/LibriSpeech-test.jsonl"   
    output_path = "/data/Shizihui/Data_preprocess/LibriSpeech/LibriSpeech-test-context.jsonl" 

    add_context_to_dataset(input_path, output_path)