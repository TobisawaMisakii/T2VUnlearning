import torch
import torch.nn as nn
import os
import json

import math
import torch.nn.init as init

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.attention import Attention
from typing import Any, Dict, Optional, Tuple, Union

from .utils import AdapterEraser

def save_cogvideo_eraser_from_transformer(folder_path, transformer):
    
    difs_eraser_ckpt = {}
    #erasers = {}
    eraser_rank = None
    for name, module in transformer.named_modules():
        if isinstance(module, CogVideoXWithEraser):
            eraser_name = f'{name}.adapter'
            if eraser_rank is None:
                eraser_rank = module.adapter.down.weight.shape[0]
            difs_eraser_ckpt[eraser_name] = module.adapter.state_dict()

    # save eraser weights
    os.makedirs(folder_path, exist_ok=True)
    eraser_weight_path = os.path.join(folder_path, f"eraser_weights.pt")
    torch.save(difs_eraser_ckpt, eraser_weight_path)
    

    # save eraser config
    eraser_config = {
        'eraser_type': 'adapter',
        'eraser_rank': eraser_rank,
    }
    eraser_config_path = os.path.join(folder_path, "eraser_config.json")
    with open(eraser_config_path, 'w') as f:
        json.dump(eraser_config, f, indent=4)


def setup_cogvideo_adapter_eraser(model, eraser_rank, device, dtype):
    def replace_transformer_block(model):
        for name, module in model.named_modules():
            if isinstance(module, CogVideoXBlock):
                print("changing: ",name)
                original_attention = module.attn1
                modified_attention = CogVideoXWithEraser(original_attention, eraser_rank).to(dtype=dtype)
                modified_attention.to_empty(device=device)
                module.attn1 = modified_attention

    replace_transformer_block(model)
    erasers = {}
    for name, module in model.named_modules():
        if isinstance(module, CogVideoXWithEraser):
            eraser_name = f'{name}.adapter'
            print(eraser_name)
            erasers[eraser_name] = module.adapter
    return erasers

def inject_eraser_from_dict(transformer, erasers, eraser_rank):
    for name, module in transformer.named_modules():
        if isinstance(module, CogVideoXBlock):
            #print("changing: ",name)
            original_attention = module.attn1
            modified_attention = CogVideoXWithEraser(original_attention, eraser_rank)
            module.attn1 = modified_attention
            eraser_name = f'{name}.attn1.adapter'
            module.attn1.adapter.load_state_dict(erasers[eraser_name].state_dict())
            module.attn1.adapter.to(device = transformer.device, dtype = transformer.dtype)



def inject_eraser(transformer, eraser_ckpt, eraser_rank, eraser_type='adapter'):
    for name, module in transformer.named_modules():
        if isinstance(module, CogVideoXBlock):
            print("changing: ",name)
            original_attention = module.attn1
            modified_attention = CogVideoXWithEraser(original_attention, eraser_rank)
            module.attn1 = modified_attention
            eraser_name = f'{name}.attn1.{eraser_type}'
            module.attn1.adapter.load_state_dict(eraser_ckpt[eraser_name])
            module.attn1.adapter.to(device = transformer.device, dtype = transformer.dtype)
            #setattr(module, name, block_w_adapter)
        

class CogVideoXWithEraser(nn.Module):
    def __init__(
        self,
        attn,
        eraser_rank
    ):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_v.weight.shape[-1], eraser_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        
        hidden_states, encoder_hidden_states = self.attn(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            **cross_attention_kwargs,
        )

        if self.adapter.use_eraser:
            hidden_states = hidden_states + self.adapter(hidden_states)

        return hidden_states, encoder_hidden_states
    
