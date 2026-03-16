import time
import argparse
import os
import warnings
import torch
from PIL import Image
from transformers import AutoTokenizer, TextStreamer
from model.model_tiny_vlm import TinyVLM, VLMConfig
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    
    config = VLMConfig(
        llm_model_path=args.llm_path,
        vision_model_path=args.vision_path,
    )
    model = TinyVLM(config)
    
    if args.weight and os.path.exists(f'{args.save_dir}/{args.weight}.pth'):
        state_dict = torch.load(f'{args.save_dir}/{args.weight}.pth', map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded weights from {args.save_dir}/{args.weight}.pth')
    
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def main():
    parser = argparse.ArgumentParser(description="Tiny-VLM Chat")
    parser.add_argument('--llm_path', default='./model/llm/qwen2-7b', type=str, help="LLM模型路径")
    parser.add_argument('--vision_path', default='./model/vision_model/internvit-300m', type=str, help="视觉模型路径")
    parser.add_argument('--save_dir', default='./out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值")
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help="测试图像目录")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt = "仔细看一下这张图：\n\n<image>\n\n描述一下这个图像的内容。"
    
    if os.path.exists(args.image_dir):
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        print(f"图像目录 {args.image_dir} 不存在")
        return
    
    for image_file in sorted(image_files):
        setup_seed(2026)
        image_path = os.path.join(args.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        pixel_values = TinyVLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)
        
        messages = [{"role": "user", "content": prompt.replace('<image>', model.config.image_special_token)}]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            inputs_text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
        
        print(f'[图像]: {image_file}')
        print(f'💬: {prompt.replace(chr(10), "\\n")}')
        print('🤖: ', end='')
        st = time.time()
        
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            pixel_values=pixel_values
        )
        
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')
        else:
            print('\n\n')


if __name__ == "__main__":
    main()
