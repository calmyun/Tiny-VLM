import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_tiny_vlm import TinyVLM, VLMConfig


def get_model_params(model, config, ignore_patterns=['vision_encoder', 'llm']):
    def should_count(n):
        return not any(p in n for p in ignore_patterns)
    
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and should_count(n)) / 1e6
    
    Logger(f'Model Params: {total:.2f}M (Trainable: {trainable:.2f}M)')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_vlm_model(
    vlm_config,
    from_weight=None,
    tokenizer_path='./model/llm/qwen2-7b',
    vision_model_path='./model/vision_model/internvit-300m',
    save_dir='./out',
    device='cuda',
    freeze_llm=False
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = TinyVLM(vlm_config)
    
    if from_weight is not None and from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            Logger(f'Loaded weights from {weight_path}')
    
    if freeze_llm:
        for name, param in model.named_parameters():
            if 'vision_proj' not in name:
                param.requires_grad = False
        
        if hasattr(model, 'llm') and hasattr(model.llm, 'model'):
            num_layers = len(model.llm.model.layers)
            for i in range(max(0, num_layers - 2), num_layers):
                for name, param in model.llm.model.layers[i].named_parameters():
                    param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable Params: {trainable_params:.3f}M')
    
    processor = model.processor
    return model.to(device), tokenizer, processor


def vlm_checkpoint(
    vlm_config,
    weight='tiny_vlm',
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir='./checkpoints',
    **kwargs
):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}.pth'
    resume_path = f'{save_dir}/{weight}_resume.pth'
    
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        
        clean_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('vision_encoder.') and not k.startswith('llm.')
        }
        
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
