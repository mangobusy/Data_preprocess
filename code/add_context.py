import json

def extract_story_id(key):
    """
    智能提取故事的唯一 ID
    支持格式1: 6097_clean/14411/nada_lily_00_haggard_0024.wav -> 提取出 6097_clean/14411/nada_lily_00_haggard
    支持格式2: LJ031-0021 -> 提取出 LJ031
    支持格式3: xxx_0024 -> 提取出 xxx
    """
    # 1. 去掉可能的 .wav 后缀
    if key.endswith('.wav'):
        key = key[:-4]
        
    # 2. 如果是 LJ031-0021 这种格式（有连字符且没有路径斜杠）
    if '-' in key and '/' not in key:
        return key.rsplit('-', 1)[0]
        
    # 3. 如果是 xxx_0024 这种格式（用下划线分隔序列号）
    if '_' in key:
        return key.rsplit('_', 1)[0]
        
    return key # 兜底返回


def add_context_to_dataset(input_file, output_file):
    prev_story_id = None
    # 使用一个列表来存储当前故事的历史句子
    history_sentences =[]

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            # 解析当前行
            data = json.loads(line)
            current_key = data['key']
            
            # 提取当前的故事 ID
            current_story_id = extract_story_id(current_key)
            
            # 判断是否属于同一个故事
            if current_story_id == prev_story_id:
                # 是同一个故事，context 为历史句子列表的副本（使用 list() 复制以防引用问题）
                data['context'] = list(history_sentences)
            else:
                # 换了新故事，清空历史句子队列，当前 context 为空列表
                history_sentences =[]
                data['context'] =[]
                
            # 将处理后的数据写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # ========== 更新变量供下一句使用 ==========
            prev_story_id = current_story_id
            
            # 将当前句子的文本加入历史队列
            history_sentences.append(data['source_text'])
            
            # 限制历史队列最大长度为5
            # 如果超过5句，就弹出最早的一句（索引0）
            if len(history_sentences) > 5:
                history_sentences.pop(0)

    print(f"✅ 处理完成！带有 context 的新数据集已保存至: {output_file}")


# ================= 运行 =================
# 替换成你实际的文件路径
input_path = "/root/autodl-tmp/data/Data_preprocess/Total/train.jsonl"   
output_path = "/root/autodl-tmp/data/Data_preprocess/Total/train-context.jsonl" 

add_context_to_dataset(input_path, output_path)