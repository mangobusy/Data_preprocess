#!/usr/bin/env python3
from __future__ import annotations
"""Generate CosyVoice speech tokens for JSONL records.

Example:
    python cosyvoice1_tokenizer.py \
        --input-jsonl /data/Shizihui/dataset/LJSpeech/ljspeech_valid.json \
        --output-jsonl /data/Shizihui/Data_preprocess/LJSpeech/ljspeech_val_audio_tokens.jsonl \
        --audio-dir /data/Shizihui/dataset/LJSpeech/wavs \
        --model-dir /data/Shizihui/Data_preprocess/ckp/CosyVoice-300M
"""
'''
python cosyvoice1_tokenizer.py \
        --input-jsonl /data/Shizihui/dataset/StroyTTS/transcript.txt \
        --output-jsonl /data/Shizihui/Data_preprocess/StoryTTS/storytts_audio_tokens.jsonl \
        --audio-dir /data/Shizihui/dataset/StroyTTS/LianLiru_ZSDFS/llr \
        --model-dir /data/Shizihui/Data_preprocess/ckp/CosyVoice-300M
'''
'''
python cosyvoice1_tokenizer.py \
        --input-jsonl /data/Shizihui/Data_preprocess/HiFi_TTS/other/hifi-tts_val.jsonl \
        --output-jsonl /data/Shizihui/Data_preprocess/HiFi_TTS/hifitts_audio_tokens_val.jsonl \
        --audio-dir /data/Shizihui/dataset/HiFi-tts/audio/val_clean \
        --model-dir /data/Shizihui/Data_preprocess/ckp/CosyVoice-300M
'''
'''
python cosyvoice1_tokenizer.py \
        --input-jsonl /data/Shizihui/Data_preprocess/Blizzard/raw.jsonl \
        --output-jsonl /data/Shizihui/Data_preprocess/Blizzard/blizzard_audio_tokens.jsonl \
        --audio-dir /data/Shizihui/dataset/blizzard_release_2017/audio \
        --model-dir /data/Shizihui/Data_preprocess/ckp/CosyVoice-300M
'''
'''
python cosyvoice1_tokenizer.py \
        --input-jsonl /data/Shizihui/Data_preprocess/LibriSpeech/librispeech-val.jsonl \
        --output-jsonl /data/Shizihui/Data_preprocess/LibriSpeech/librispeech-val_audio_tokens.jsonl \
        --audio-dir /data/Shizihui/dataset/LibriSpeech/clean/validation/audio \
        --model-dir /data/Shizihui/Data_preprocess/ckp/CosyVoice-300M
'''

import argparse
import json
import os
import sys
from pathlib import Path
import re
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from tqdm import tqdm
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COSYVOICE_ROOT = REPO_ROOT / "CosyVoice"
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"

sys.path.append(str(COSYVOICE_ROOT))
sys.path.append(str(MATCHA_ROOT))

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CosyVoice')))

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add CosyVoice speech tokens to JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output-jsonl", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model-dir", required=True, help="CosyVoice model directory or ModelScope ID.")
    parser.add_argument("--cosyvoice-version", type=int, default=1, choices=[1, 2], help="CosyVoice model version.")
    parser.add_argument("--audio-dir", help="Directory containing audio files named as <key><audio-ext>.")
    parser.add_argument("--audio-ext", default=".wav", help="Audio extension when using --audio-dir.")
    parser.add_argument("--audio-field", default="source_wav", help="JSON field name holding the audio path.")
    parser.add_argument("--audio-root", help="Base directory to resolve relative audio paths.")
    parser.add_argument("--key-field", default="key", help="JSON field name used with --audio-dir.")
    parser.add_argument("--device", default="cuda", help="Torch device used for tokenization.")
    return parser.parse_args()


def resolve_model_dir(model_dir: str) -> str:
    if os.path.exists(model_dir):
        return model_dir
    return snapshot_download(model_dir)


def load_frontend(model_dir: str, cosyvoice_version: int) -> CosyVoiceFrontEnd:
    config_path = os.path.join(model_dir, "cosyvoice.yaml")
    with open(config_path, "r", encoding="utf-8") as config_file:
        overrides = {
            "llm": None,
            "flow": None,
            "hift": None,
        }
        if cosyvoice_version == 2:
            overrides["qwen_pretrain_path"] = os.path.join(model_dir, "CosyVoice-BlankEN")
            speech_tokenizer_model = os.path.join(model_dir, "speech_tokenizer_v2.onnx")
        else:
            
            speech_tokenizer_model = os.path.join(model_dir, "speech_tokenizer_v1.onnx")
        configs = load_hyperpyyaml(config_file, overrides=overrides)

    return CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        os.path.join(model_dir, "campplus.onnx"),
        speech_tokenizer_model,
        os.path.join(model_dir, "spk2info.pt"),
        configs["allowed_special"],
    )


def resolve_audio_path(entry: dict, args: argparse.Namespace, input_path: Path, dataset) -> Path:
    if args.audio_dir:  # train_clean
        key = entry.get(args.key_field) # "key": 6097_clean/8992/presentpictureofnsw_03_mann_0786.wav
        if dataset == 'storytts':
            episode = key.split("-")[2]
            return Path(args.audio_dir) / episode /f"{key}{args.audio_ext}"
        elif dataset == 'hifitts':
            return Path(args.audio_dir) / f"{key}"
        elif dataset == 'LJSpeech':
            return Path(args.audio_dir) / f"{key}{args.audio_ext}"
        elif dataset == "Blizzard":
            return Path(args.audio_dir) / f"{key}{args.audio_ext}"  # /data/Shizihui/dataset/blizzard_release_2017/audio 
        elif dataset == 'librispeech':
            return Path(args.audio_dir) / f"{key}"

        if not key:
            raise KeyError(f"Missing key field: {args.key_field}")
        

    audio_path = entry.get(args.audio_field)
    if not audio_path:
        raise KeyError(f"Missing audio field: {args.audio_field}")

    audio_path = Path(audio_path)  # ?
    if audio_path.is_absolute():
        return audio_path

    return audio_path

def file_Format_Adaptation(input_file, clean_file, dataset):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")
    if not os.path.exists(clean_file):
        raise FileNotFoundError(f"找不到输出文件: {clean_file}")

    with open(input_file, 'r', encoding='utf-8') as f:  

        if dataset=='LJSpeech':
            data = json.load(f)
            print(f"转换文件格式中，共 {len(data)} 条数据...")
            with open(clean_file, 'w', encoding='utf-8') as out_f:
                for key, item in tqdm(data.items()):
                    text_content = item.get('char', '')
                    new_item = {
                            "key": key,
                            "source_text": text_content,
                            "target_text": text_content,
                        }
                    out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        
        elif dataset=='storytts':
            with open(clean_file, 'w', encoding='utf-8') as out_f:
                for line in tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        print(f"警告：跳过格式错误的行: {line}")
                        continue
                    key = parts[0]
                    text_content = parts[1]
                    new_key = re.sub(r'episode0*(\d+)', r'episode\1', key)

                    new_item = {
                            "key": new_key,
                            "source_text": text_content,
                            "target_text": text_content,
                        }
                    out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
    print(f"文件格式转换完成，已保存到: {clean_file}")

def extract_speech_tokens(frontend: CosyVoiceFrontEnd, audio_path: Path, device: torch.device) -> list[int]:
    try:
        speech = load_wav(str(audio_path), 16000)
        speech = speech.to(device)
        speech_token, _ = frontend._extract_speech_token(speech)
        return speech_token.squeeze(0).tolist()
    except TypeError as exc:
        if "Invalid file" not in str(exc):
            raise
        speech_token, _ = frontend._extract_speech_token(str(audio_path))
        return speech_token.squeeze(0).tolist()

def main() -> None:
    dataset = "librispeech"  # "hifitts"
    args = parse_args()
    raw_file = Path(args.input_jsonl)
    if dataset == 'storytts':
        clean_file = Path("/data/Shizihui/Data_preprocess/StoryTTS/other/storytts_train.jsonl")
        file_Format_Adaptation(raw_file, clean_file, dataset)
    elif dataset == 'hifitts':
        clean_file = raw_file
    elif dataset == 'LJSpeech':
        clean_file = Path("/data/Shizihui/Data_preprocess/LJSpeech/other/val/ljspeech_val.jsonl")
        file_Format_Adaptation(raw_file, clean_file, dataset)
    elif dataset == "Blizzard":
        clean_file = raw_file
    elif dataset == "librispeech":
        clean_file = raw_file
    output_path = Path(args.output_jsonl)

    model_dir = resolve_model_dir(args.model_dir)
    frontend = load_frontend(model_dir, args.cosyvoice_version)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    

    with clean_file.open("r", encoding="utf-8") as clean_file, output_path.open(
        "a", encoding="utf-8"
    ) as output_file:
        # for line in tqdm(clean_file, desc="Tokenizing", unit="line"):
        for idx, line in enumerate(tqdm(clean_file, desc="Tokenizing", unit="line"), start=1):
            if idx <= 2528:
                continue
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_path = resolve_audio_path(entry, args, clean_file, dataset)
            # breakpoint()
            entry["answer_cosyvoice_speech_token"] = extract_speech_tokens(frontend, audio_path, device)
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
