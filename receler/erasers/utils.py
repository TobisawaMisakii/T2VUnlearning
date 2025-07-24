import os
import json
import torch
import torch.nn as nn
from omegaconf import OmegaConf
#from .cogvideo_erasers import CogVideoXWithEraser


import transformers
import re

def find_subject(sentence,word):
  """
  Finds word (case-insensitive) in a sentence using regex.

  Args:
    sentence: The input sentence (string).
    word: The subject word (string).

  Returns:
    True if "word" is found, False otherwise.
  """
  if not isinstance(sentence, str):
    return False

  pattern = r"{}".format(word)  # \s+ matches one or more whitespace characters
  match = re.search(pattern, sentence, re.IGNORECASE)
  return bool(match)

def get_proper_prompts(pipeline,text,type,device,dtype,max_new_tokens=200):
    sys_prompt_t2v = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
    There are a few rules to follow:

    You will only ever output a single video description per user request.

    When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
    Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

    Video descriptions must have the same num of words as examples below. Extra words will be ignored.
    """

    

    if(type == "i2p"):
        text = "an explicit scene of " + text

    messages=[
        {"role": "system", "content": f"{sys_prompt_t2v}"},
        {
            "role": "user",
            "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " a girl is on the beach"',
        },
        {
            "role": "assistant",
            "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
        },
        {
            "role": "user",
            "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A man jogging on a football field"',
        },
        {
            "role": "assistant",
            "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
        },
        {
            "role": "user",
            "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
        },
        {
            "role": "assistant",
            "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
        },
        {
            "role": "user",
            "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {text} "',
        },
    ]

    outputs = pipeline(messages,
                    do_sample = True,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
    )
    res = outputs[0]["generated_text"][-1]['content']

    while(type == "face" and not find_subject(res,text)):
        outputs = pipeline(messages,
                    do_sample = True,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
        )
        res = outputs[0]["generated_text"][-1]['content']

    return res


def get_proper_prompts_for_hunyuan(pipeline,text,type,device,dtype,max_new_tokens=200):
    sys_prompt_t2v = """You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description.

    There are a few rules to follow:

    Preserve ALL information, including style words and technical terms.

    If the input is just one or two words describing an object or person, provide a brief, simple description focusing on basic visual characteristics. Limit the description to 1-2 short sentences.

    If the input does not include style, lighting, atmosphere, you can make reasonable associations.

    Output ALL must be in English.

    You will only ever output a single video description per user request.

    Video descriptions must have the same num of words as examples below. Extra words will be ignored.
    """

    

    if(type == "i2p"):
        text = "an explicit scene of " + text

    messages=[
        {"role": "system", "content": f"{sys_prompt_t2v}"},
        {
            "role": "user",
            "content": 'Create a video description for the user input: "cat"',
        },
        {
            "role": "assistant",
            "content": "A cat walks on the grass, realistic style.",
        },
        {
            "role": "user",
            "content": 'Create a video description for the user input: "Girl in Winter"',
        },
        {
            "role": "assistant",
            "content": "Close-up, A little girl wearing a red hoodie in winter strikes a match.",
        },
        {
            "role": "user",
            "content": 'Create a video description for the user input: "Woman Yoga"',
        },
        {
            "role": "assistant",
            "content": "Woman practicing yoga in a lush garden setting.",
        },
        {
            "role": "user",
            "content": f'Create a video description in ENGLISH for the user input: " {text} "',
        },
    ]

    outputs = pipeline(messages,
                    do_sample = True,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
    )
    res = outputs[0]["generated_text"][-1]['content']

    while(type == "face" and not find_subject(res,text)):
        outputs = pipeline(messages,
                    do_sample = True,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
        )
        res = outputs[0]["generated_text"][-1]['content']

    return res

def ldm_module_prefix_name(module_name):
    return '.'.join(module_name.split('.')[:2])

def save_eraser_to_diffusers_format(folder_path, erasers, eraser_rank, pretrained_cfg):
    # mapping from ldm to diffusers
    def ldm_to_diffusers(blk_prefix, lay_per_blk):
        blk_type, blk_id = blk_prefix.split('.')
        blk_id = int(blk_id)
        if blk_type == 'input_blocks':
            difs_blk_id = (blk_id - 1) // (lay_per_blk + 1)
            difs_lay_id = (blk_id - 1) % (lay_per_blk + 1)
            return f'down_blocks.{difs_blk_id}.attentions.{difs_lay_id}'
        elif blk_type == 'middle_block':
            return 'mid_block.attentions.0'
        elif blk_type == 'output_blocks':
            difs_blk_id = blk_id // (lay_per_blk + 1)
            difs_lay_id = blk_id % (lay_per_blk + 1)
            return f'up_blocks.{difs_blk_id}.attentions.{difs_lay_id}'
        else:
            raise AttributeError(f'Got unexpected block type: {blk_type}.')

    original_config = OmegaConf.load(pretrained_cfg)
    layers_per_block = original_config.model.params.unet_config.params['num_res_blocks']
    difs_eraser_ckpt = {}
    for eraser_name, eraser in erasers.items():
        eraser_blk_prefix = ldm_module_prefix_name(eraser_name)
        difs_blk_prefix = ldm_to_diffusers(eraser_blk_prefix, layers_per_block)
        difs_eraser_ckpt[difs_blk_prefix] = eraser.state_dict()

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



def save_cogvideo_eraser_to_diffusers_format(folder_path, erasers, eraser_rank):
    
    difs_eraser_ckpt = {}
    for eraser_name, eraser in erasers.items():
        difs_eraser_ckpt[eraser_name] = eraser.state_dict()

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

def save_hunyuan_eraser_to_diffusers_format(folder_path, erasers, eraser_rank):
    
    difs_eraser_ckpt = {}
    for eraser_name, eraser in erasers.items():
        difs_eraser_ckpt[eraser_name] = eraser.state_dict()

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



class DisableEraser:
    def __init__(self, model, train=False):
        self.model = model
        self.train = train
        self.old_states = {}

    def __enter__(self):
        self.old_training = self.model.training
        self.model.train(self.train)
        # disable erasers
        for name, module in self.model.named_modules():
            if isinstance(module, EraserControlMixin):
                self.old_states[name] = module.use_eraser
                module.use_eraser = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.train(self.old_training)
        # enable erasers
        for name, module in self.model.named_modules():#待会返回来改
            if isinstance(module, EraserControlMixin):
                module.use_eraser = self.old_states[name]


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class EraserControlMixin:
    _use_eraser = True

    @property
    def use_eraser(self):
        return self._use_eraser

    @use_eraser.setter
    def use_eraser(self, state):
        if not isinstance(state, bool):
            raise AttributeError(f'state should be bool, but got {type(state)}.')
        self._use_eraser = state


class AdapterEraser(nn.Module, EraserControlMixin):
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.down = nn.Linear(dim, mid_dim)
        self.act = nn.GELU()
        self.up = zero_module(nn.Linear(mid_dim, dim))

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))
