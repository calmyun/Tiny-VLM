import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Tuple, List, Union
from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(PretrainedConfig):
    model_type = "tiny-vlm"

    def __init__(
        self,
        llm_model_path: str = "./model/llm/qwen2-7b",
        vision_model_path: str = "./model/vision_model/internvit-300m",
        image_special_token: str = '<image>',
        num_image_tokens: int = 256,
        vision_hidden_size: int = 1024,
        llm_hidden_size: int = 3584,
        max_seq_len: int = 2048,
        **kwargs,
    ):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.image_special_token = image_special_token
        self.num_image_tokens = num_image_tokens
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)


class VisionProjector(nn.Module):
    def __init__(self, vision_hidden_size: int = 1024, llm_hidden_size: int = 3584, num_image_tokens: int = 256):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_image_tokens = num_image_tokens
        
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
    
    def forward(self, image_features):
        projected_features = self.projector(image_features)
        return projected_features


class TinyVLM(nn.Module):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None):
        super().__init__()
        if config is None:
            config = VLMConfig()
        self.config = config
        self.params = config
        
        self.llm = None
        self.tokenizer = None
        self.vision_encoder = None
        self.processor = None
        self.vision_proj = None
        
        self._init_model()

    def _find_model_path(self, base_path):
        if os.path.exists(base_path):
            if os.path.exists(os.path.join(base_path, 'config.json')):
                return base_path
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
                    return item_path
        return base_path

    def _init_model(self):
        llm_path = self._find_model_path(self.config.llm_model_path)
        vision_path = self._find_model_path(self.config.vision_model_path)
        
        if os.path.exists(llm_path) and os.path.exists(os.path.join(llm_path, 'config.json')):
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                trust_remote_code=True
            )
        
        if os.path.exists(vision_path) and os.path.exists(os.path.join(vision_path, 'config.json')):
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()
            
            self.vision_encoder = AutoModel.from_pretrained(
                vision_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            self.processor = AutoImageProcessor.from_pretrained(
                vision_path,
                trust_remote_code=True
            )
            
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
        
        self.vision_proj = VisionProjector(
            vision_hidden_size=self.config.vision_hidden_size,
            llm_hidden_size=self.config.llm_hidden_size,
            num_image_tokens=self.config.num_image_tokens
        )

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

    def _find_image_token_indices(self, input_ids, image_token_id):
        batch_size = input_ids.shape[0]
        indices = []
        for b in range(batch_size):
            positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                indices.append((b, positions[0].item()))
        return indices

    def _replace_image_tokens(self, hidden_states, input_ids, vision_features, image_token_id):
        batch_size = hidden_states.shape[0]
        image_indices = self._find_image_token_indices(input_ids, image_token_id)
        
        if len(image_indices) == 0:
            return hidden_states
        
        for b, start_idx in image_indices:
            num_tokens = min(vision_features.shape[1], hidden_states.shape[1] - start_idx)
            if num_tokens > 0:
                hidden_states[b, start_idx:start_idx + num_tokens, :] = vision_features[b, :num_tokens, :]
        
        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        
        start_pos = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].shape[1]
        
        hidden_states = self.llm.model.embed_tokens(input_ids)
        
        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 5:
                pixel_values = pixel_values.squeeze(1)
            
            vision_features = self.get_image_embeddings(pixel_values, self.vision_encoder)
            vision_features = self.vision_proj(vision_features)
            
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.config.image_special_token)
            if image_token_id == self.tokenizer.unk_token_id:
                image_token_id = self.tokenizer.encode(self.config.image_special_token, add_special_tokens=False)
                if isinstance(image_token_id, list) and len(image_token_id) > 0:
                    image_token_id = image_token_id[0]
                else:
                    image_token_id = -1
            
            if image_token_id >= 0:
                hidden_states = self._replace_image_tokens(
                    hidden_states, input_ids, vision_features, image_token_id
                )
        
        outputs = self.llm.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.llm.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        if pixel_values is not None:
            if len(pixel_values.shape) == 5:
                pixel_values = pixel_values.squeeze(1)
            
            vision_features = self.get_image_embeddings(pixel_values, self.vision_encoder)
            vision_features = self.vision_proj(vision_features)
            
            hidden_states = self.llm.model.embed_tokens(inputs)
            
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.config.image_special_token)
            if image_token_id == self.tokenizer.unk_token_id:
                image_token_id = self.tokenizer.encode(self.config.image_special_token, add_special_tokens=False)
                if isinstance(image_token_id, list) and len(image_token_id) > 0:
                    image_token_id = image_token_id[0]
                else:
                    image_token_id = -1
            
            if image_token_id >= 0:
                hidden_states = self._replace_image_tokens(
                    hidden_states, inputs, vision_features, image_token_id
                )
            
            return self.llm.generate(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )
        else:
            return self.llm.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = VLMConfig.from_pretrained(model_path)
        model = cls(config)
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model
