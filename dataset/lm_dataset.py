import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import io
from PIL import Image
from torch.utils.data import Dataset
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TinyVLMDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        processor=None,
        max_length: int = 2048,
        image_special_token: str = '<image>'
    ):
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = processor
        self.image_token = image_special_token
        
        self.bos_token = tokenizer.bos_token if tokenizer.bos_token else "<|im_start|>"
        self.eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|im_end|>"
        
    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            content = turn['content'].replace('<image>', self.image_token)
            messages.append({"role": role, "content": content})
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                if msg['role'] == 'user':
                    prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        return prompt

    def generate_labels(self, input_ids, prompt_text):
        labels = [-100] * len(input_ids)
        
        assistant_start = prompt_text.find("<|im_start|>assistant")
        if assistant_start == -1:
            assistant_start = prompt_text.find("assistant")
        
        if assistant_start != -1:
            prefix_text = prompt_text[:assistant_start]
            prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False).input_ids
            start_idx = len(prefix_tokens)
            
            for i in range(start_idx, len(input_ids)):
                labels[i] = input_ids[i]
        
        return labels

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        
        labels = self.generate_labels(input_ids, prompt)
        
        image_tensors = []
        for img_bytes in image_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            if self.processor is not None:
                img_tensor = self.processor(images=image, return_tensors="pt")['pixel_values']
                image_tensors.append(img_tensor)
        
        if len(image_tensors) > 0:
            image_tensor = torch.cat(image_tensors, dim=0)
        else:
            image_tensor = torch.zeros(1, 3, 448, 448)
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            image_tensor
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        if os.path.exists(path):
            t = pq.read_table(path)
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            for i in range(min(5, len(t))):
                ax[i].imshow(Image.open(io.BytesIO(t['image_bytes'][i].as_py())))
                ax[i].axis('off')
                conv = json.loads(t['conversations'][i].as_py())
                title = conv[1]['content'][:30] if len(conv) > 1 else "N/A"
                ax[i].set_title(title, fontsize=8)
            out = path.replace('.parquet', '_preview.png')
            plt.savefig(out)
            print(f'已保存{out}, 共{len(t)}条')
