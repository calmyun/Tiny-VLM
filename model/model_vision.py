import torch
import torch.nn as nn
from transformers import PretrainedConfig, AutoModel, AutoImageProcessor
from typing import Optional, Tuple, List
import os

class InternVisionConfig(PretrainedConfig):
    model_type = "intern_vit"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        qk_normalization: bool = True,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn


class InternVisionModel(nn.Module):
    def __init__(self, model_path: str = "./model/vision_model/internvit-300m"):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        if os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs
    
    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state
        return img_embedding
    
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)
