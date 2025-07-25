import argparse
from typing import Literal
import wandb
from typing import Optional, Union, List
import inspect
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import (
    CogVideoXPipeline,
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

from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDPMScheduler, CogVideoXDDIMScheduler

import os
import json
import copy


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

    parser.add_argument(
        "--output_path", type=str, default="./output.mp4",
        help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--output_type",
        type=Literal["latent", "video"],
        default="video",
        choices=["latent", "video"],
        help="The output type of the generated video, can be 'latent' or 'video'",
    )
    parser.add_argument(
        "is_train",
        type=bool,
        default=False,
        help="Whether to train the model or do inference"
    )
    return parser.parse_args()


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CogVideoXPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


def inference(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = 'cuda'
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.enable_sequential_cpu_offload()
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.model_path, subfolder="scheduler", torch_dtype=dtype
    )

    vae =  pipe.vae
    text_encoder = pipe.text_encoder
    transformer = pipe.transformer

    # freeze transformer
    for param in transformer.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    vae.to("cuda")
    text_encoder.to("cuda")
    transformer.to("cuda")
    adapter_transformer.to("cuda")

    height = transformer.config.sample_height * pipe.vae_scale_factor_spatial
    width = transformer.config.sample_width * pipe.vae_scale_factor_spatial
    num_frames = transformer.config.sample_frames

    num_videos_per_prompt = 1
    do_classifier_free_guidance = False

    with torch.no_grad():
        # 1. Encode input prompt
        batch_size = 1
        prompt_embeds = pipe.encode_prompt(
            args.prompt,
            negative_prompt=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=226,
            device='cuda',
        )[0].detach()    # torch.Size([1, 226, 4096])

        # 2. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, args.num_inference_steps, device='cuda', timesteps=None
        ) # do not support custom timesteps
        pipe._num_timesteps = len(timesteps)
        # print(f"Using timesteps: {timesteps}")

        # 3. Prepare latents
        latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

        patch_size_t = pipe.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * pipe.vae_scale_factor_temporal

        latent_channels = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device='cuda',
            generator=None,
            latents=None,
        )# torch.Size([1, 13, 16, 60, 90])
        # print(f"Using latents shape: {latents.shape}")

        # Create rotary embeds if required
        image_rotary_emb = (
            pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if transformer.config.use_rotary_positional_embeddings
            else None
        )

        # Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):

                pipe._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                # no adapter, unsafe prompt
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]    # torch.Size([1, 13, 16, 60, 90])
                noise_pred = noise_pred.float().detach()


                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(scheduler, CogVideoXDPMScheduler):
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()

            pipe._current_timestep = None

    if not args.output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = pipe.decode_latents(latents)
            video = pipe.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    video = CogVideoXPipelineOutput(frames=video).frames[0]
    export_to_video(video, args.output_path + "_original.mp4", fps=8)


def unlearn_train(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.model_path, subfolder="scheduler", torch_dtype=dtype
    )

    vae =  pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    transformer = pipe.transformer

    # freeze transformer
    for param in transformer.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # 检查任意一个参数的requires_grad状态
    print(next(pipe.transformer.parameters()).requires_grad)

    adapter_transformer = copy.deepcopy(transformer)
    eraser = setup_cogvideo_adapter_eraser(
        model=adapter_transformer,
        eraser_rank=args.eraser_rank,
        device="cuda",
        dtype=dtype,
    )

    
    for block in transformer.transformer_blocks:
        if hasattr(block, "adapter"):
            for param in block.adapter.parameters():
                param.requires_grad = True

    # for name, param in transformer.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable: {name}, shape: {tuple(param.shape)}")
    # for name, param in adapter_transformer.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable in atptransformer: {name}, shape: {tuple(param.shape)}")

    for name, module in eraser.items():
        for pname, param in module.named_parameters():
            print(f"{name}.{pname} - mean: {param.data.mean()}, std: {param.data.std()}, requires_grad: {param.requires_grad}")
    # exit(0)

    adam_optimizer = optim.AdamW(
        params=[param for module in eraser.values() for param in module.parameters()],
        lr=1e-6,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    vae.to("cuda")
    text_encoder.to("cuda")
    transformer.to("cuda")
    adapter_transformer.to("cuda")

    height = transformer.config.sample_height * pipe.vae_scale_factor_spatial
    width = transformer.config.sample_width * pipe.vae_scale_factor_spatial
    num_frames = transformer.config.sample_frames
    # Using height: 480, width: 720, num_frames: 49
    # print(f"Using height: {height}, width: {width}, num_frames: {num_frames}")
    # exit(0)

    num_videos_per_prompt = 1
    do_classifier_free_guidance = False

    for epoch in range(args.num_epoch):
        # only compute once
        zero_embeds = pipe.encode_prompt(
                    "",
                    negative_prompt=None,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=226,
                    device='cuda',
                )[0].detach()    # torch.Size([1, 226, 4096])
        
        # for each prompt pairs do unlearning training
        for prompt, res_prompt in train_data_loader(args.prompt_path):
            batch_size = 1

            with torch.no_grad():
                # 1. Encode input prompt
                prompt_embeds = pipe.encode_prompt(
                    prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=226,
                    device='cuda',
                )[0].detach()    # torch.Size([1, 226, 4096])

                res_prompt_embeds = pipe.encode_prompt(
                    res_prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=226,
                    device='cuda',
                )[0].detach()    # torch.Size([1, 226, 4096])

                # 2. Prepare timesteps
                timesteps, num_inference_steps = retrieve_timesteps(
                    scheduler, args.num_inference_steps, device='cuda', timesteps=None
                ) # do not support custom timesteps
                pipe._num_timesteps = len(timesteps)
                # print(f"Using timesteps: {timesteps}")

                # 3. Prepare latents
                latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

                patch_size_t = pipe.transformer.config.patch_size_t
                additional_frames = 0
                if patch_size_t is not None and latent_frames % patch_size_t != 0:
                    additional_frames = patch_size_t - latent_frames % patch_size_t
                    num_frames += additional_frames * pipe.vae_scale_factor_temporal

                latent_channels = pipe.transformer.config.in_channels
                latents = pipe.prepare_latents(
                    batch_size * num_videos_per_prompt,
                    latent_channels,
                    num_frames,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device='cuda',
                    generator=None,
                    latents=None,
                )# torch.Size([1, 13, 16, 60, 90])
                # print(f"Using latents shape: {latents.shape}")

                # Create rotary embeds if required
                image_rotary_emb = (
                    pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
                    if transformer.config.use_rotary_positional_embeddings
                    else None
                )

            # Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

            with pipe.progress_bar(total=num_inference_steps) as progress_bar:
                # for DPM-solver++
                old_pred_original_sample = None
                for i, t in enumerate(timesteps):

                    pipe._current_timestep = t
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # predict noise model_output

                    with torch.no_grad():
                        # no adapter, unsafe prompt
                        noise_pred = transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]    # torch.Size([1, 13, 16, 60, 90])
                        noise_pred = noise_pred.float().detach()
                        v_unsafe_origin = noise_pred
                        # print("v_unsafe_origin shape:", v_unsafe_origin.shape)
                        # print("v_unsafe_origin grad:", v_unsafe_origin.requires_grad)

                        # no adapter, safe prompt
                        v_safe_origin = transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=res_prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        v_safe_origin = v_safe_origin.float().detach()
                        # print("v_safe_origin shape:", v_safe_origin.shape)
                        # print("v_safe_origin grad:", v_safe_origin.requires_grad)

                        # no adapter, no prompt
                        v_noprompt_origin = transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=zero_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        v_noprompt_origin = v_noprompt_origin.float().detach()
                        # print("v_noprompt_origin shape:", v_noprompt_origin.shape)
                        # print("v_noprompt_origin grad:", v_noprompt_origin.requires_grad)

                    # with adapter, unsafe prompt
                    v_unsafe_adapter = adapter_transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    v_unsafe_adapter = v_unsafe_adapter.float()
                    # print("v_unsafe_adapter shape:", v_unsafe_adapter.shape)
                    # print("v_unsafe_adapter grad:", v_unsafe_adapter.requires_grad)

                    # with adapter, safe prompt
                    v_safe_adapter = adapter_transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=res_prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    v_safe_adapter = v_safe_adapter.float()

                    # calculate loss
                    loss_unlearn = 0.0
                    loss_preserve = 0.0
                    loss_localize = 0.0

                    # loss unlearn
                    v_neg = v_noprompt_origin - args.eta * (v_unsafe_origin - v_noprompt_origin)
                    loss_unlearn = torch.mean((v_neg - v_unsafe_adapter) ** 2)
                    # print(loss_unlearn.requires_grad, loss_unlearn.grad_fn)

                    print("loss_unlearn:", loss_unlearn.item())
                    print("v_neg mean/std:", v_neg.mean().item(), v_neg.std().item())
                    print("v_unsafe_adapter mean/std:", v_unsafe_adapter.mean().item(), v_unsafe_adapter.std().item())


                    # loss preserve
                    loss_preserve = torch.mean((v_safe_origin - v_safe_adapter) ** 2)

                    # loss localize


                    loss = loss_unlearn + loss_preserve + loss_localize


                    # backward
                    adam_optimizer.zero_grad()
                    loss.backward()
                    adam_optimizer.step()

                    for name, module in eraser.items():
                        for pname, param in module.named_parameters():
                            print(f"{name}.{pname} - mean: {param.data.mean()}, std: {param.data.std()}, requires_grad: {param.requires_grad}")
                    exit(0)


                    wandb.log({
                        "epoch": epoch,
                        "prompt": prompt,
                        "res_prompt": res_prompt,
                        "loss_unlearn": loss_unlearn.item(),
                        "loss_preserve": loss_preserve.item(),
                        "loss_localize": loss_localize.item() if isinstance(loss_localize, torch.Tensor) else loss_localize,
                        "loss_total": loss.item(),
                    })
                    print(f"Epoch {epoch}, Step {i+1}/{len(timesteps)}, Loss: {loss.item()}")


                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(scheduler, CogVideoXDPMScheduler):
                        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                        progress_bar.update()

            pipe._current_timestep = None


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
    if args.is_train:
        unlearn_train(args)
    else:
        inference(args)
