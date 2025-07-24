import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from einops import rearrange

from diffusers.models.attention import Attention
from .erasers.utils import ldm_module_prefix_name
import math

import os
from torchvision.utils import save_image

def concatenate_frames(frames):
    """
    Concatenate multiple frames horizontally into a single image
    
    Args:
        frames: List of frames (F, H, W)
        
    Returns:
        Concatenated image (H, F*W)
    """
    # Ensure all frames are of the same size
    H, W = frames[0].shape
    num_frames = len(frames)
    
    # Create a blank canvas
    concat_image = torch.zeros((H, W * num_frames), device=frames[0].device, dtype=frames[0].dtype)
    
    # Horizontally concatenate all frames
    for i, frame in enumerate(frames):
        concat_image[:, i*W:(i+1)*W] = frame
        
    return concat_image

def save_mask_frames(tensor: torch.Tensor, save_dir: str, prefix: str = "frame"):
    """
    Saves each frame of a binary tensor as images and concatenates all frames into a single image.

    Args:
        tensor (torch.Tensor): A tensor of shape [1, frames, height, width] containing only 0s and 1s.
        save_dir (str): Directory to save the images.
        prefix (str): Prefix for saved image filenames.
    """
    if tensor.ndimension() != 4 or tensor.shape[0] != 1:
        raise ValueError("Tensor must have shape [1, frames, height, width]")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Concatenate all frames horizontally and save as one image
    concatenated = concatenate_frames(tensor[0])
    save_path = os.path.join(save_dir, f"{prefix}_concatenated.png")
    save_image(concatenated, save_path)

@torch.no_grad()
def get_cross_attn_mask(query, key, token_idx, head_num = 30,text_len = 226,height = 10, width = 10):
    #you'll have to input the height and width of the latent here
    inner_dim = key.shape[-1]
    head_dim = inner_dim // head_num

    query = query.view(query.shape[0], -1, head_num, head_dim).transpose(1, 2)
    key = key.view(key.shape[0], -1, head_num, head_dim).transpose(1, 2) 
   

    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_scores, dim=-1)

    attn_map_mean = attn_probs.sum(dim=1) / head_num #heads

    attn_map = attn_map_mean[:, :-text_len, -text_len:]#batch_size,visual_len,text_len

    B, HWF, T = attn_map.shape
    F = HWF // (height * width)
    attn_map = attn_map.reshape(B, F, height, width, T)

    
    attn_map = attn_map[..., token_idx]
    attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
   
    # Get min and max values using PyTorch
    attn_min = attn_map.amin(dim=(2, 3), keepdim=True)
    attn_max = attn_map.amax(dim=(2, 3), keepdim=True)
    
    normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)  # Add small value to avoid division by zero
    
    return normalized_attn



#apparently, we will get b * f * h * w latents
@torch.no_grad()
def get_mask(attn_maps, word_indices, thres, height, width, head_num=30, text_len=226):
    """
    attn_maps: {module: {name : b, seqlen, head_dim * heads}}}
    word_indices: (num_tokens,)
    thres: float, threshold of mask
    """
    ret_masks = {}
    attns_choosen = []
    average_mask = None
    count = 0.0
    for name, attns in attn_maps.items():
        
        #print(attns)
        query = attns['to_q']
        key = attns['to_k']
        mask = get_cross_attn_mask(query,key,word_indices,height=height,width=width,head_num=head_num,text_len=text_len)
        if average_mask is None:
            average_mask = mask
        else:
            average_mask += mask
        count += 1.0
        mask[mask >= thres] = 1
        mask[mask < thres] = 0
    
        ret_masks[name] = mask

    average_mask /= count

    ret_masks["average_mask"] = average_mask
    return ret_masks

    
class AttnMapsCapture:
    def __init__(self, model, attn_maps):
        self.model = model
        self.attn_maps = attn_maps
        self.handlers = []

    def __enter__(self):
        
        
        for name,module in self.model.single_transformer_blocks.named_modules():
            if isinstance(module, Attention):
                #print("add forward hook in:","single_transformer_blocks." + name)
                h_q = module.to_q.register_forward_hook(self.get_attn_maps("single_transformer_blocks." + name,"to_q"))
                h_k = module.to_k.register_forward_hook(self.get_attn_maps("single_transformer_blocks." + name,"to_k"))
                
                self.handlers.append(h_q)
                self.handlers.append(h_k)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_attn_maps(self, module,name):
            def hook(model, input, output):
                if module not in self.attn_maps.keys():
                    self.attn_maps[module] = {}
                self.attn_maps[module][name] = output.detach()
            return hook


class EraserOutputsCapture:
    def __init__(self, model, erasers, eraser_outs):
        self.model = model
        self.eraser_names = list(erasers.keys())
        self.eraser_outs = eraser_outs
        self.handlers = []

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.eraser_names:
                handler = module.register_forward_hook(self.get_eraser_outs(module_name))
                self.handlers.append(handler)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_eraser_outs(self, module_name):
            def hook(model, input, output):
                self.eraser_outs[module_name] = output
            return hook
