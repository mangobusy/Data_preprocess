
# import argparse
# from datasets import load_dataset, load_from_disk
# from tqdm import tqdm
# import json
# import numpy as np

import os
import json
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def extract_parquet_to_flac(
    parquet_path, 
    jsonl_output_path, 
    audio_output_dir
):
    # 1. 创建音频输出目录
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)
        print(f"创建音频目录: {audio_output_dir}")

    # 2. 加载 Parquet 文件
    print(f"正在加载数据集: {parquet_path} ...")
    ds = load_dataset("parquet", data_files=parquet_path, split="train")

    print("开始处理数据...")
    
    with open(jsonl_output_path, 'a', encoding='utf-8') as f_out:
        for row in tqdm(ds):
            # === 1. 处理文件名 ===
            original_filename = row['file']
            if original_filename.startswith("audio/"):
                relative_path_flac = original_filename[6:]  # 结果: '6097_clean/14411/nada.flac'
            else:
                relative_path_flac = original_filename

            base_path = os.path.splitext(relative_path_flac)[0] 
            relative_path = f"{base_path}.wav"

            save_full_path = os.path.join(audio_output_dir, relative_path)
            # breakpoint()
            # 获取文件所在的目录路径 (例如 .../6097_clean/14411)
            save_dir = os.path.dirname(save_full_path)
            # 如果目录不存在，递归创建 (exist_ok=True 表示如果目录已存在不报错)
            os.makedirs(save_dir, exist_ok=True)

            # === 3. 保存音频 ===
            audio_data = row['audio']
            # 直接写入，soundfile 会根据后缀 .flac 自动处理
            sf.write(save_full_path, audio_data['array'], audio_data['sampling_rate'])
            
            

            # === 3. 处理 Metadata ===
            # 将 metadata 写入 jsonl
            # 注意：这里的 file 字段我们通常更新为实际保存的新文件名，或者保留原始文件名
            # 这里我示范更新为新的文件名，以便后续训练代码能找到它
            meta_entry = {
                'key': relative_path,                    
                'source_text': row['text_normalized'],
                'target_text': row['text_normalized']
            }

            f_out.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

    print(f"处理完成！")
    print(f"Metadata (JSONL) 已保存至: {jsonl_output_path}")
    print(f"Audio (wav) 已保存至: {audio_output_dir}")

if __name__ == "__main__":
    # 配置路径
  
    PARQUET_FILE = "/data/Shizihui/dataset/HiFi-tts/data/train.clean-00024-of-00035-019b9d9e9771173c.parquet"   # 修改这里
    JSONL_OUTPUT = "/data/Shizihui/dataset/HiFi-tts/hifi-tts_train_3.jsonl"
    AUDIO_DIR = "/data/Shizihui/dataset/HiFi-tts/audio/train_clean"

    extract_parquet_to_flac(PARQUET_FILE, JSONL_OUTPUT, AUDIO_DIR)

'''

def process_data(id, load_from_cache_file=True, seed=42, split_size=0.0001):
    file_id_str = f"{id:05d}" 
    train_data_path = f"/root/autodl-tmp/data/Libritts_R/train.clean.100/train.clean.100-{file_id_str}-of-00018.parquet"
    # train_data_path = "/data/Shizihui/dataset/HiFi-tts/data/test.clean-00000-of-00001-ad9def4d041642e2.parquet"
    print(f"Processing file: {train_data_path}")
    if load_from_cache_file:       
        ds = load_dataset("parquet", data_files=train_data_path)
        
    else:
        ds = load_from_disk(train_data_path)
        
    breakpoint()
    train_val_split = ds['train'].train_test_split(test_size=split_size, seed=seed)
    train_data_list = train_val_split["train"]
    val_data_list = train_val_split['test']
    train_json_path = f"/root/autodl-tmp/data/Libritts_R/train.clean.100_{id}.jsonl"
    val_json_path = f"/root/autodl-tmp/data/Libritts_R/val.clean.100_{id}.jsonl"



    with open(val_json_path, 'w') as out_f:
        for data in tqdm(val_data_list, total=len(val_data_list)):
    # with open(test_json_path, 'w') as out_f:
    #     for data in tqdm(val_data_list, total=len(val_data_list)):
            data_dict = {
                'key': data['file'],
                'speaker': data['speaker'],
                'chapter_id': data['chapter_id'],
                'text_normalized': data['text_normalized'],
                
            }
            out_f.write(json.dumps(data_dict,ensure_ascii=False) + '\n')

    with open(train_json_path, 'w') as out_f:
        for data in tqdm(train_data_list, total=len(train_data_list)):
            data_dict = {
                'key': data['id'],
                'chapter_id': data['chapter_id'],
                'text_original': data['text_original'],
                'text_normalized': data['text_normalized'],
             
            }
            out_f.write(json.dumps(data_dict,ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data files based on the given ID.")
    parser.add_argument('id', type=int, help='The ID of the data part to process')
    args = parser.parse_args()


    process_data(args.id)

'''