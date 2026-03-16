import os
import sys

def download_models():
    from modelscope import snapshot_download
    
    vision_model_dir = './model/vision_model/internvit-300m'
    llm_model_dir = './model/llm/qwen2-7b'
    
    if not os.path.exists(vision_model_dir):
        print("正在下载 InternViT-300M-448px 模型...")
        snapshot_download(
            'OpenGVLab/InternViT-300M-448px',
            cache_dir=vision_model_dir,
            revision='master'
        )
        print(f"InternViT-300M-448px 已下载到 {vision_model_dir}")
    else:
        print(f"InternViT-300M-448px 已存在于 {vision_model_dir}")
    
    if not os.path.exists(llm_model_dir):
        print("正在下载 Qwen2-7B 模型...")
        snapshot_download(
            'qwen/Qwen2-7B',
            cache_dir=llm_model_dir,
            revision='master'
        )
        print(f"Qwen2-7B 已下载到 {llm_model_dir}")
    else:
        print(f"Qwen2-7B 已存在于 {llm_model_dir}")
    
    print("\n模型下载完成！")

if __name__ == '__main__':
    download_models()
