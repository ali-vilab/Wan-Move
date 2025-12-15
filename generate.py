# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import random
import math

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.modules.trajectory import draw_tracks_on_video

EXAMPLE_PROMPT = {
    # for Wan-Move
    "wan-move-i2v": {
        "prompt":
            "A laptop is placed on a wooden table. The silver laptop is connected to a small grey external hard drive and transfers data through a white USB-C cable. The video is shot with a downward close-up lens.",
        "image":
            "examples/example.jpg",
        "track":
            "examples/example_tracks.npy",
        "track_visibility":
            "examples/example_visibility.npy"
    }
}
def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )

    # General arguments
    parser.add_argument(
        "--task",
        type=str,
        default="wan-move-i2v",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan-Move-14B-480P",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--vis_track",
        action="store_true",
        default=False,
        help="Whether to visualize trajectory (e.g. tracks on first frame and save).")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="The precision to use for the model.")

    # Specified for single video generation
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="The point trajectories to control motion.")
    parser.add_argument(
        "--track_visibility",
        type=str,
        default=None,
        help="The visibility of point trajectories.")

    # Specified for MoveBench evaluation
    parser.add_argument(
        "--eval_bench",
        action="store_true",
        help="Use the data from MoveBench.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="results",
        help="The path of the ouput video for MoveBench evaluation.")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Single-object trajectory or Multi-object trajectory.")
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Language of captions/prompts. Choose from {'zh', 'en'}. Default is 'zh'.")
    args = parser.parse_args()

    _validate_args(args)

    return args

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def _load_datalist(path):
    """Read datalist. Each line: '<video_rel_path>,<caption>', split by the first comma."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            line = ln.strip()
            if not line or line.startswith('#'):
                continue
            if ',' not in line:
                raise ValueError(f"Invalid datalist line (missing comma): {line}")
            video_rel, caption = line.split(',', 1)
            items.append((video_rel.strip(), caption.strip()))
    return items

def _distribute_block(n_total, rank, world_size):
    """Split block [0, n_total) to each rank for Multi-GPU infernece. Return [start, end)"""
    if world_size <= 1:
        return 0, n_total
    per = math.ceil(n_total / world_size)
    start = rank * per
    end = min(start + per, n_total)
    return start, end

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    if args.dtype == "fp32":
        cfg.param_dtype = torch.float32
    elif args.dtype == "fp16":
        cfg.param_dtype = torch.float16
    elif args.dtype == "bf16":
        cfg.param_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---------------------------
    # Change Wan_I2V to Wan-Move_I2V pipeline
    # --------------------------
    if "i2v" in args.task:
        logging.info("Creating WanMove pipeline.")
        wan_move = wan.WanMove(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        if rank == 0:
            os.makedirs(args.save_path, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        # ---- Process datalist ----
        logging.info("Generating motion-controllable video ...")

        if args.eval_bench:
            datalist = f"MoveBench/{args.language}/{args.mode}_track.txt"
            assert os.path.isfile(datalist)
            all_items = _load_datalist(datalist)
            n_total = len(all_items)
            steps = (n_total + world_size - 1) // world_size

            try:
                for s in range(steps):
                    global_idx = s * world_size + rank
                    do_save = global_idx < n_total
                    eff_idx = global_idx if do_save else (n_total - 1)
                    video_id, caption = all_items[eff_idx]    
                    video_id = os.path.splitext(video_id)[0]
                    image_path = os.path.join(f"MoveBench/{args.language}/first_frame", video_id+'.jpg')
                    track_path = os.path.join(f"MoveBench/{args.language}/track", args.mode, video_id + '_tracks.npy')
                    track_vis_path = os.path.join(f"MoveBench/{args.language}/track", args.mode, video_id + '_visibility.npy')
                    
                    img = Image.open(image_path).convert("RGB")
                    track = np.load(track_path)
                    track_visibility = np.load(track_vis_path)
                    input_prompt = caption

                    assert not args.use_prompt_extend
                    logging.info(f"[Rank {rank}] Generating idx={global_idx} base={video_id}")

                    video = wan_move.generate(
                        input_prompt,
                        img,
                        track,
                        track_visibility,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=args.base_seed + global_idx,
                        offload_model=args.offload_model,
                        eval_bench=args.eval_bench)

                    # Save videos
                    if do_save and video is not None:
                        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        formatted_prompt = input_prompt.replace(" ", "_").replace("/",
                                                                            "_")[:50]
                        suffix = '.mp4'
                        save_file = f"{args.save_path}/{video_id}" + suffix

                        logging.info(f"[Rank {rank}] Saving video to {save_file}")

                        if args.vis_track:
                            first_frame_repeat = torch.as_tensor(np.array(img)).permute(2,0,1).unsqueeze(0).unsqueeze(1).repeat(1, args.frame_num, 1, 1, 1)
                            track_video = draw_tracks_on_video(first_frame_repeat, torch.from_numpy(track), torch.from_numpy(track_visibility))
                            track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).permute(1,0,2,3).mul(2).sub(1).to(device)
                            cache_video(
                                tensor=torch.stack([track_video, video]),
                                save_file=save_file,
                                fps=cfg.sample_fps,
                                nrow=1,
                                normalize=True,
                                value_range=(-1, 1))

                        else:
                            cache_video(
                                tensor=video[None],
                                save_file=save_file,
                                fps=cfg.sample_fps,
                                nrow=1,
                                normalize=True,
                                value_range=(-1, 1))
                  

            except Exception as e:
                logging.exception(f"[Rank {rank}] eval_bench error: {e}")
            finally:
                torch.cuda.synchronize()
                dist.barrier()
                dist.destroy_process_group()
                logging.info(f"Rank {rank} finished its shard.")
                return

        # Inference single instance
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        if args.track is None:
            args.track = EXAMPLE_PROMPT[args.task]["track"]
        if args.track_visibility is None:
            args.track_visibility = EXAMPLE_PROMPT[args.task]["track_visibility"]

        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")
        logging.info(f"Input trajectory: {args.track}")
        logging.info(f"Input trajectory's visibility: {args.track_visibility}")

        img = Image.open(args.image).convert("RGB")
        track = np.load(args.track)
        track_visibility = np.load(args.track_visibility)

        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        video = wan_move.generate(
            args.prompt,
            img,
            track,
            track_visibility,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            eval_bench=args.eval_bench)

    else:
        raise ValueError(f"Unkown task type: {args.task}")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if args.vis_track:
            first_frame_repeat = torch.as_tensor(np.array(img)).permute(2,0,1).unsqueeze(0).unsqueeze(1).repeat(1, args.frame_num, 1, 1, 1)
            track_video = draw_tracks_on_video(first_frame_repeat, torch.from_numpy(track), torch.from_numpy(track_visibility))
            track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).permute(1,0,2,3).mul(2).sub(1).to(device)
            cache_video(
                tensor=torch.stack([track_video, video]),
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

        else:
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
