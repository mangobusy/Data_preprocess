
from huggingface_hub import snapshot_download
'''
# 直接将仓库文件下载到本地目录
local_dir = "/data/Shizihui/dataset/LJSpeech"
snapshot_download(
    repo_id="flexthink/ljspeech",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 确保下载的是实体文件而不是链接
)

print(f"下载完成，数据保存在 {local_dir}")
'''

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="model-scope/CosyVoice-300M",
    filename="speech_tokenizer_v1.onnx",
    repo_type="model",
    local_dir="/data/Shizihui/Data_preprocess/ckp",
    local_dir_use_symlinks=False,
)
print("Downloaded to:", path)



# # !pip install openai-whisper

# import os
# import torch
# import whisper

# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# model = whisper.load_model(
#     "large-v3",
#     device=device,
#     download_root="/root/autodl-tmp/EmoVoice/checkpoint"
# )


# snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/root/autodl-tmp/CosyVoice/ckp/CosyVoice2-0.5B')
# snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='/root/autodl-tmp/CosyVoice/ckp/CosyVoice-ttsfrd')
