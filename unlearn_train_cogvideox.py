import argparse
from typing import Literal
import wandb
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.utils import export_to_video, load_image, load_video
from receler.erasers.cogvideo_erasers import (
    inject_eraser, 
    CogVideoXWithEraser, 
    setup_cogvideo_adapter_eraser,
    )
from receler.erasers.utils import DisableEraser

import os
import json


def generate_video(
    prompt: str,
    model_path: str,
    eraser_path: str = None,
    eraser_rank: int = 128,
    output_path: str = "./output",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_frames: int = 49,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    generate_clean: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - eraser_path (str): The path of the eraser weights to be used.
    - eraser_rank (int): The rank of the eraser weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_frames (int): Number of generated frames.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if eraser_path:
        eraser_ckpt_path = os.path.join(eraser_path, f'eraser_weights.pt')
        eraser_config_path = os.path.join(eraser_path, f'eraser_config.json')
        with open(eraser_config_path) as f:
            eraser_config = json.load(f)
        # inject erasers into pretrained cogvideo
        inject_eraser(pipe.transformer, eraser_ckpt=torch.load(eraser_ckpt_path, map_location='cpu'), eraser_rank=eraser_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    pipe.to("cuda")

    pipe.enable_sequential_cpu_offload()

    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
    if generate_clean:
        with DisableEraser(pipe.transformer):
                video_before = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
    video_erased = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    
    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    if generate_clean:
        export_to_video(video_before, output_path + "_clean.mp4", fps=8)
    export_to_video(video_erased, output_path + "_erase.mp4", fps=8)


def args_parser():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--concept", type=str, required=True, 
        help="The sensitive content to be erased"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True, 
        help="The path to the text file containing the prompt, now assuming using nudity-ring-a-bell.csv, which contains unsafe and safe prompt pairs"
    )
    parser.add_argument(
        "--image_or_video_path",type=str,default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, required=True,
        help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--eraser_ckpt_path", type=str, default=None, 
        help="The path of the eraser weights/config to be saved"
    )
    parser.add_argument(
        "--eraser_rank", type=int, default=128, 
        help="The rank of the eraser weights"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, 
        help="Number of steps for the inference process"
    )
    parser.add_argument(
        "--num_frames", type=int, default=49, 
        help="Number of frames to generate per prompt, should be 8N + 1 where N <= 6"
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", 
        help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", 
        help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=20, 
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--eta", type=float, default=7.0, 
        help="The eta hyperparam for unlearning training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, 
        help="The seed for reproducibility"
    )
    return parser.parse_args()


def unlearn_train(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)  
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # pipe_origin = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    # pipe_origin.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()

    # pipe_origin.to("cuda")
    # pipe_origin.enable_sequential_cpu_offload()

    if not args.generate_type == "t2v":
        raise NotImplementedError(
            f"Generate type {args.generate_type} is not implemented for unlearning training."
        )

    eraser = setup_cogvideo_adapter_eraser(
        model=pipe.transformer,
        eraser_rank=args.eraser_rank,
        device="cuda",
        dtype=dtype,
    )
    # parameters = [param for module in eraser.values() for param in module.parameters()]
    # print(f"Number of parameters in eraser: {sum(p.numel() for p in parameters)}")
    # print(f"Number of modules in eraser: {len(eraser)}")
    # exit(0)

    adam_optimizer = optim.AdamW(
        params=[param for module in eraser.values() for param in module.parameters()],
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    intermediate_outputs_with_adapter_safe = {}
    intermediate_outputs_with_adapter_unsafe = {}
    intermediate_outputs_original_safe = {}
    intermediate_outputs_original_unsafe = {}
    intermediate_outputs_original_noprompt = {}

    hook_handles = []
    def register_hooks(model, target_module_names, output_dict):
        for name, module in model.named_modules():
            if name in target_module_names:
                print(f"Registering hook for {name}")
                def hook_fn(module, input, output, name=name):
                    output.retain_grad()  # Retain gradients for the output
                    output_dict[name] = output
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)

    target_module_names = [
        "transformer_blocks.29.ff",
    ]

    # print(pipe.transformer.transformer_blocks[0].attn1.adapter)
    # print(pipe.transformer.transformer_blocks[0])
    # print(pipe.transformer.transformer_blocks[29].ff)

    for epoch in range(args.num_epoch):
        for prompt, res_prompt in train_data_loader(args.prompt_path):
            # 存储中间结果
            intermediate_outputs_with_adapter_unsafe.clear()
            intermediate_outputs_original_unsafe.clear()

            register_hooks(pipe.transformer, target_module_names, intermediate_outputs_with_adapter_unsafe)
            # register_hooks(pipe_origin.transformer, target_module_names, intermediate_outputs_original_unsafe)

            # Forward pass —— with eraser —— unsafe prompt
            video = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(args.seed),
            ).frames[0]
            v_unsafe_adapter = intermediate_outputs_with_adapter_unsafe[target_module_names[0]][1]
            print(v_unsafe_adapter.requires_grad)
            print(v_unsafe_adapter.grad_fn)
            exit(0)
            
            # Forward pass —— without eraser —— unsafe prompt
            video_origin = pipe_origin(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(args.seed),
            ).frames[0]
            v_unsafe_origin = intermediate_outputs_original_unsafe[target_module_names[0]][1]

            # clear hook handles which are tracking unsafe prompt
            for handle in hook_handles:
                handle.remove()
            hook_handles.clear()

            # register new hooks for safe prompt
            intermediate_outputs_with_adapter_safe.clear()
            intermediate_outputs_original_safe.clear()

            register_hooks(pipe_origin.transformer, target_module_names, intermediate_outputs_original_safe)
            register_hooks(pipe.transformer, target_module_names, intermediate_outputs_with_adapter_safe)

            # Forward pass —— with eraser —— safe prompt
            video_safe = pipe(
                prompt=res_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(args.seed),
            ).frames[0]
            v_safe_adapter = intermediate_outputs_with_adapter_safe[target_module_names[0]][1]

            # Forward pass —— without eraser —— safe prompt
            video_origin_safe = pipe_origin(
                prompt=res_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(args.seed),
            ).frames[0]
            v_safe_origin = intermediate_outputs_original_safe[target_module_names[0]][1]

            # clear hook handles which are tracking safe prompt
            for handle in hook_handles:
                handle.remove()
            hook_handles.clear()

            # no prompt
            intermediate_outputs_original_noprompt.clear()
            register_hooks(pipe_origin.transformer, target_module_names, intermediate_outputs_original_noprompt)
            video_noprompt = pipe_origin(
                prompt="",
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(args.seed),
            ).frames[0]
            v_noprompt_origin = intermediate_outputs_original_noprompt[target_module_names[0]][1]

            # calculate loss
            loss_unlearn = 0.0
            loss_preserve = 0.0
            loss_localize = 0.0

            # loss unlearn
            # **adapter**模型， 在**unsafe prompt**下的第30个（最后一个）transformer block的 feedforward layer 输出 $v_{\theta'}(x_t, c, t)$
            v_neg = v_noprompt_origin - args.eta * (v_unsafe_origin - v_noprompt_origin)
            loss_unlearn = torch.mean((v_neg - v_unsafe_adapter) ** 2)

            # loss preserve
            # loss localize

            loss = loss_unlearn + loss_preserve + loss_localize

            # backward
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            wandb.log({
                "epoch": epoch,
                "prompt": prompt,
                "res_prompt": res_prompt,
                "loss_unlearn": loss_unlearn.item(),
                "loss_preserve": loss_preserve.item(),
                "loss_localize": loss_localize.item(),
                "loss_total": loss.item(),
            })


def train_data_loader(prompt_path, batch_size=1):
    import pandas as pd
    df = pd.read_csv(prompt_path)
    prompts = df['prompt'].tolist()
    res_prompts = df['normal prompt'].tolist()
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_res_prompts = res_prompts[i:i + batch_size]
        yield batch_prompts, batch_res_prompts


if __name__ == "__main__":
    args = args_parser()

    project_name = "cogvideox-unlearn"
    exp_name = "l_unlearn_exp"
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    wandb.init(
        project=project_name,  # 自定义项目名
        name=f"{exp_name}-{run_name}",      # 每次运行的名字，可换成时间戳、实验参数等
        config=vars(args),            # 保存参数信息
        mode="disabled",
    )

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    unlearn_train(args)
