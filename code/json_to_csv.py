import json
import csv
import re

def convert_jsonl_to_csv(input_file, output_file, dataset):
    # 打开输入和输出文件
    # newline='' 是为了防止在 Windows 上出现空行
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'a', encoding='utf-8', newline='') as f_out:
        
        # 定义 CSV 的列名
        # ============audio_token=============
        # header = ['ID', 'story', 'text','key','audio_token']
        # ==============emotion=============
        header = ['ID', 'story', 'text']

        # 初始化 CSV writer
        # 如果你想要制表符分隔 (看起来像 Excel 复制出来的样子)，可以把 delimiter=',' 改为 delimiter='\t'
        writer = csv.writer(f_out, delimiter=',') 
        # 写入表头
        writer.writerow(header)
        # 初始化 ID 计数器
        row_id = 1
        print(f"开始转换: {input_file} -> {output_file}")
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            # 1. 解析 JSON
            data = json.loads(line)
            raw_key = data.get('key', '')     # e.g., "LJ001-0001"
            text = data.get('source_text', '') # e.g., "Printing..."
            audio_token = data.get('answer_cosyvoice_speech_token', []) # e.g., [1,2,3,...]
            # 2. 处理 'story' 字段
            # 逻辑：获取 "-" 前面的部分 (LJ001)，提取数字
            try:
                if dataset == 'ljspeech':
                    story_part = raw_key.split('-')[0] # 拿到 "LJ001"
                    # 使用正则提取数字，或者切片去掉 'LJ'
                    # 这里假设格式固定为 LJ 开头
                    story_num_str = story_part.replace('LJ', '') # 拿到 "001"
                    story_num = int(story_num_str) # 变成数字 1 (去掉了前导0)
                    story_val = f"story{story_num}" # 变成 "story1"
                elif dataset == 'storytts':
                    story_num_str = re.search(r"episode(\d+)", raw_key)
                    story_num = story_num_str.group(1)
                    story_val = f"story{story_num}"
                elif dataset == 'hifitts':
                    story_num = raw_key.split('/')[1]
                    if story_num == '14411':
                        story_val = "story1"
                    elif story_num == '15326':
                        story_val = "story2"
                    elif story_num == '15277':
                        story_val = "story3"
                    elif story_num == '11271':
                        story_val = "story4"
                    elif story_num == '13657':
                        story_val = "story5"
                    elif story_num == '8992':
                        story_val = "story6"
                    elif story_num == '9575':
                        story_val = "story7"
                    elif story_num == '14764':
                        story_val = "story8"
                    elif story_num == '14843':
                        story_val = "story9"
                    elif story_num == '13418':
                        story_val = "story10"
                    elif story_num == '13885':
                        story_val = "story11"
                    elif story_num == '14261':
                        story_val = "story12"
                    elif story_num == '15752':
                        story_val = "story13"
                    elif story_num == '15482':
                        story_val = "story14"
                    elif story_num == '13130':
                        story_val = "story15"
                    elif story_num == '13089':
                        story_val = "story16"
                    elif story_num == '11216':
                        story_val = "story17"
                    elif story_num == '10425':
                        story_val = "story18"
                    elif story_num == '9288':
                        story_val = "story19"
                    elif story_num == '13594':
                        story_val = "story20"
                    elif story_num == '10633':
                        story_val = "story21"
                    elif story_num == '6973':
                        story_val = "story22"
                    elif story_num == '7967':
                        story_val = "story23"
                    elif story_num == '9783':
                        story_val = "story24"
            except Exception as e:
                # 防止格式不匹配报错
                print(f"Key 格式警告: {raw_key}")
                story_val = raw_key

            # 3. 写入 CSV 行
            writer.writerow([row_id, story_val, text])
            
            row_id += 1

    print(f"转换完成！共处理 {row_id - 1} 行数据。")

if __name__ == "__main__":
    # 设置你的文件名
    INPUT_JSONL = "/data/Shizihui/Data_preprocess/HiFi_TTS/other/audio_token/hifitts_audio_tokens_4.jsonl"  # 你的源文件路径
    OUTPUT_CSV = "/data/Shizihui/Data_preprocess/HiFi_TTS/other/hifitts.csv"    # 结果文件路径
    dataset = ""  # 数据集名称，可以根据需要修改
    # 执行转换
    convert_jsonl_to_csv(INPUT_JSONL, OUTPUT_CSV, dataset)