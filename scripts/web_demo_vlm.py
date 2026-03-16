import os
import sys
import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer, TextStreamer
from model.model_tiny_vlm import TinyVLM, VLMConfig

def load_model():
    config = VLMConfig(
        llm_model_path='./model/llm/qwen2-7b',
        vision_model_path='./model/vision_model/internvit-300m',
    )
    model = TinyVLM(config)
    
    if os.path.exists('./out/sft_vlm.pth'):
        state_dict = torch.load('./out/sft_vlm.pth', map_location='cuda')
        model.load_state_dict(state_dict, strict=False)
        print("Loaded SFT weights")
    
    return model.eval().cuda(), model.tokenizer, model.processor

model, tokenizer, processor = None, None, None

def chat(image, message, history):
    global model, tokenizer, processor
    
    if model is None:
        model, tokenizer, processor = load_model()
    
    if image is not None:
        image = image.convert('RGB')
        pixel_values = TinyVLM.image2tensor(image, processor).cuda().unsqueeze(0)
    else:
        pixel_values = None
    
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    if image is not None:
        content = message.replace('<image>', model.config.image_special_token)
        if '<image>' not in message and model.config.image_special_token not in message:
            content = model.config.image_special_token + '\n' + message
    else:
        content = message
    
    messages.append({"role": "user", "content": content})
    
    if hasattr(tokenizer, 'apply_chat_template'):
        inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        inputs_text = ""
        for msg in messages:
            if msg['role'] == 'user':
                inputs_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            else:
                inputs_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        inputs_text += "<|im_start|>assistant\n"
    
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pixel_values=pixel_values
        )
    
    response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    history.append((message, response))
    return history, history

def main():
    with gr.Blocks(title="Tiny-VLM Chat") as demo:
        gr.Markdown("# Tiny-VLM 多模态对话系统")
        gr.Markdown("基于 InternViT-300M + Qwen2-7B 的视觉语言模型")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传图片")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话")
                message = gr.Textbox(label="输入消息", placeholder="输入您的问题...")
                with gr.Row():
                    submit = gr.Button("发送")
                    clear = gr.Button("清空对话")
        
        state = gr.State([])
        
        submit.click(chat, [image_input, message, state], [chatbot, state])
        message.submit(chat, [image_input, message, state], [chatbot, state])
        clear.click(lambda: ([], []), None, [chatbot, state])
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
