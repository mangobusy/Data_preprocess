import json
import os
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort
import numpy as np
from tqdm import tqdm

# 强制使用 soundfile 后端
try:
    torchaudio.set_audio_backend('soundfile')
except Exception:
    pass

def generate_custom_jsonl(
    json_path, 
    output_jsonl_path, 
    onnx_model_path, 
    target_sr=16000,
    device='cuda'
):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到输入文件: {json_path}")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"找不到模型文件: {onnx_model_path}")

    # 初始化 ONNX
    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(onnx_model_path, providers=providers)
    except Exception as e:
        print(f"CUDA加载失败，切换回CPU: {e}")
        sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    input_nodes = sess.get_inputs()
    speech_name = input_nodes[0].name
    length_name = input_nodes[1].name
    
    print(f"模型输入节点: 1.{speech_name}, 2.{length_name}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"开始处理，共 {len(data)} 条数据...")
    
    # 缓存重采样器
    resamplers = {}

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for key, item in tqdm(data.items()):
            # 获取wav路径和文本会有不一样
            wav_path_raw = item.get('wav')
            # 路径拼接逻辑
            wav_path = os.path.join("/data/Shizihui/dataset/LJSpeech/wavs", os.path.basename(wav_path_raw))
            text_content = item.get('char', '')
            
            if not os.path.exists(wav_path):
                print(f"\n[跳过] 音频文件缺失: {wav_path}")
                continue

            try:
                # --- A. 音频加载 ---
                wav, sr = torchaudio.load(wav_path)
                
                # 转单声道
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                
                # 重采样
                if sr != target_sr:
                    if sr not in resamplers:
                        resamplers[sr] = torchaudio.transforms.Resample(sr, target_sr)
                    wav = resamplers[sr](wav)
                
                # 音频归一化 (防止爆音)
                max_val = wav.abs().max()
                if max_val > 1:
                    wav /= max_val

                # --- B. 特征提取 ---
                # Kaldi Fbank 需要 int16 范围的数值
                wav = wav * (1 << 15)
                
                # 提取 Fbank: [Time, 128]
                feat = kaldi.fbank(wav, num_mel_bins=128, sample_frequency=16000, dither=0.0)

                # --- C. [核心修复] 特征归一化 (Instance Norm) ---
                # 这一步非常重要！如果不做，输入全是负数，导致 Token 固定
                # 对每个样本的特征做 (x - mean) / std
                feat_mean = feat.mean(0, keepdim=True)
                feat_std = feat.std(0, keepdim=True)
                # 防止除以0
                feat = (feat - feat_mean) / (feat_std + 1e-5)

                # --- D. 维度调整 ---
                # [Time, 128] -> [1, 128, Time]
                feat = feat.transpose(0, 1)
                feat = feat.unsqueeze(0)
                
                feat_numpy = feat.numpy()
                
                # --- E. 长度 ---
                feat_len = np.array([feat_numpy.shape[2]], dtype=np.int32)

                # 构造输入
                inputs = {
                    speech_name: feat_numpy,
                    length_name: feat_len
                }

                # --- F. 推理 ---
                outputs = sess.run(None, inputs)
                
                tokens = outputs[0].squeeze().tolist()
                # =========================================================
                # print(tokens)
                # break
                # =========================================================
                if isinstance(tokens, int):
                    tokens = [tokens]
                elif not tokens:
                    tokens = []

                new_item = {
                    "key": key,
                    "source_text": text_content,
                    "target_text": text_content,
                    "answer_cosyvoice_speech_token": tokens
                }
                
                out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                
            except Exception as e:
                # 打印第一个错误以便调试，后续静默
                # print(f"\n[错误] {key}: {e}")
                pass

    print(f"\n处理完成！文件已保存至: {output_jsonl_path}")

if __name__ == "__main__":
    INPUT_JSON = "/data/Shizihui/dataset/LJSpeech/ljspeech_train.json"           
    OUTPUT_JSONL = "/data/Shizihui/Data_preprocess/LJSpeech/LJSpeech_with_audio_token_fixed.jsonl" 
    ONNX_MODEL = "/data/Shizihui/Data_preprocess/ckp/speech_tokenizer_v1.onnx" 
    
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    generate_custom_jsonl(INPUT_JSON, OUTPUT_JSONL, ONNX_MODEL)