import argparse
import logging
import os
import sys
import random
import math
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import gradio as gr
from datetime import datetime

# Ensure we can import from wan
sys.path.append(os.getcwd())

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.modules.trajectory import draw_tracks_on_video

# --- Argument Parsing and Validation (Adapted from generate.py) ---

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

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
        default=True, # Default to True for Gradio to save VRAM
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
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="The precision to use for the model.")
    
    # Gradio specific
    parser.add_argument("--share", action="store_true", help="Share the Gradio app.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app.")

    args = parser.parse_args()
    _validate_args(args)
    return args

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

# --- Trajectory Utilities ---

# --- Trajectory Utilities ---

from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import imageio
import re

# Color map for trajectories (matching visualize.py)
COLOR_MAP = [
    (102, 153, 255), # Blue-ish
    (0, 255, 255),   # Cyan
    (255, 255, 0),   # Yellow
    (255, 102, 204), # Pink
    (0, 255, 0),     # Green
    (255, 0, 0),     # Red
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (255, 255, 255), # White
    (165, 42, 42)    # Brown
]

COLOR_EMOJIS = ["🔵", "💠", "🟡", "🌸", "🟢", "🔴", "🟣", "🟠", "⚪", "🟤"]

def interpolate_trajectory(points, num_frames=81, smooth=True, speeds=None):
    """
    points: List of (x, y)
    speeds: List of speed values at each point. Length must match points.
    """
    if not points:
        return np.zeros((num_frames, 1, 2)), np.zeros((num_frames, 1)), None, None, None, None
        
    points = np.array(points)
    if len(points) == 1:
        return np.tile(points[None, :, :], (num_frames, 1, 1)), np.ones((num_frames, 1)), None, None, None, None

    # 1. Parameterize points by cumulative distance (chord length)
    dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    
    if total_dist == 0:
         return np.tile(points[0][None, None, :], (num_frames, 1, 1)), np.ones((num_frames, 1)), None, None, None, None

    # Normalize distance s in [0, 1]
    s_points = cumulative_dist / total_dist
    
    # 2. Create spatial interpolation functions x(s), y(s)
    kind = 'cubic' if smooth and len(points) > 3 else 'linear'
    try:
        fx = interp1d(s_points, points[:, 0], kind=kind)
        fy = interp1d(s_points, points[:, 1], kind=kind)
    except Exception:
        fx = interp1d(s_points, points[:, 0], kind='linear')
        fy = interp1d(s_points, points[:, 1], kind='linear')

    # 3. Handle Speed / Timing
    if speeds is None or len(speeds) != len(points):
        speeds = [1.0] * len(points)
    
    speeds = np.array(speeds)
    speeds = np.maximum(speeds, 0.1)
    
    try:
        fv = interp1d(s_points, speeds, kind='linear') 
    except:
        fv = interp1d(s_points, speeds, kind='linear')
        
    s_fine = np.linspace(0, 1, 1000)
    v_fine = fv(s_fine)
    
    dtds = 1.0 / v_fine
    t_fine = cumulative_trapezoid(dtds, s_fine, initial=0)
    t_total = t_fine[-1]
    
    t_fine_norm = t_fine / t_total
    
    ft_inv = interp1d(t_fine_norm, s_fine, kind='linear', bounds_error=False, fill_value=(0, 1))
    
    t_uniform = np.linspace(0, 1, num_frames)
    s_samples = ft_inv(t_uniform)
    
    x_new = fx(s_samples)
    y_new = fy(s_samples)
    
    interpolated_points = np.stack([x_new, y_new], axis=1)
            
    tracks = np.array(interpolated_points)[:, None, :] # [F, 1, 2]
    visibility = np.ones((num_frames, 1)) # [F, 1]
    
    return tracks, visibility, t_uniform, v_fine, s_fine, t_fine_norm

def draw_trajectory_on_image(img, points, smooth=True, speeds=None, color_idx=0):
    """
    Draws ONLY the active trajectory on the input image for editing.
    """
    if img is None: return None
    img_pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    
    # Use specific color for the active trajectory
    traj_color = COLOR_MAP[color_idx % len(COLOR_MAP)]
    
    # Draw input points
    r = 5
    for i, p in enumerate(points):
        # Start green, end red, others use trajectory color
        pt_color = traj_color
        if i == 0: pt_color = (0, 255, 0)
        elif i == len(points) - 1: pt_color = (255, 0, 0)
        
        draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=pt_color, outline=(0,0,0))
        
    if len(points) > 1:
        vis_tracks, _, _, _, _, _ = interpolate_trajectory(points, num_frames=200, smooth=smooth, speeds=speeds)
        vis_points = vis_tracks[:, 0, :]
        
        for i in range(len(vis_points) - 1):
            p0 = tuple(vis_points[i])
            p1 = tuple(vis_points[i+1])
            draw.line([p0, p1], fill=traj_color, width=3)
            
    return np.array(img_pil)

# --- Speed Curve Editor Utilities ---

def get_display_range(speeds):
    if not speeds:
        return 0.0, 2.0
    s_min = min(speeds)
    s_max = max(speeds)
    if s_min == s_max:
        center = s_min
        span = 1.0
        return max(0, center - span), center + span
    span = s_max - s_min
    padding = max(0.5, span * 0.3)
    y_min = max(0, s_min - padding)
    y_max = s_max + padding
    return y_min, y_max

def draw_speed_curve_editor(speeds, width=800, height=600):
    y_min, y_max = get_display_range(speeds)
    
    if not speeds:
        img = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        font_size = 30
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = None
        text = "Add points to edit speed"
        draw.text((width//2 - 150, height//2), text, fill=(0,0,0), font=font)
        return np.array(img)
        
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    total_span = y_max - y_min
    if total_span <= 2.0: step = 0.5
    elif total_span <= 5.0: step = 1.0
    else: step = 2.0
    
    start_tick = (int(y_min / step) + 1) * step
    if start_tick < y_min: start_tick += step
    
    current_tick = start_tick
    while current_tick < y_max:
        ratio = (current_tick - y_min) / (y_max - y_min)
        y_pixel = height - (ratio * height)
        draw.line([(60, y_pixel), (width, y_pixel)], fill=(220, 220, 220), width=2)
        label = f"{current_tick:.1f}x"
        draw.text((5, y_pixel - 10), label, fill=(120, 120, 120))
        current_tick += step
        
    if y_min <= 1.0 <= y_max:
        ratio = (1.0 - y_min) / (y_max - y_min)
        y_pixel = height - (ratio * height)
        draw.line([(60, y_pixel), (width, y_pixel)], fill=(180, 180, 180), width=3)
        draw.text((5, y_pixel - 10), "1.0x", fill=(80, 80, 80))

    num_points = len(speeds)
    coords = []
    for i, s in enumerate(speeds):
        if num_points == 1:
            x = 60 + (width - 60) / 2
        else:
            x = 60 + i * ((width - 60) / (num_points - 1))
        s_clamped = max(y_min, min(y_max, s))
        ratio = (s_clamped - y_min) / (y_max - y_min)
        y = height - (ratio * height)
        coords.append((x, y))
        
    if num_points > 1:
        try:
            x_vals = [c[0] for c in coords]
            y_vals = [c[1] for c in coords]
            f = interp1d(x_vals, y_vals, kind='cubic')
            x_plot = np.linspace(60, width, 200)
            y_plot = f(x_plot)
            y_plot = np.clip(y_plot, 0, height)
            plot_points = list(zip(x_plot, y_plot))
            draw.line(plot_points, fill=(0, 0, 255), width=4)
        except:
            draw.line(coords, fill=(0, 0, 255), width=4)
            
    r = 12
    for cx, cy in coords:
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 0, 0), outline=(0,0,0))
        
    return np.array(img)

def process_speed_curve_click(evt: gr.SelectData, speeds):
    if not speeds:
        return speeds
    width = 800
    height = 600
    y_min, y_max = get_display_range(speeds)
    click_x, click_y = evt.index[0], evt.index[1]
    eff_width = width - 60
    eff_click_x = click_x - 60
    num_points = len(speeds)
    if num_points == 1:
        idx = 0
    else:
        spacing = eff_width / (num_points - 1)
        idx = int(round(eff_click_x / spacing))
        idx = max(0, min(num_points - 1, idx))
    ratio = (height - click_y) / height
    new_speed = y_min + ratio * (y_max - y_min)
    new_speed = max(0.1, new_speed)
    speeds[idx] = new_speed
    return speeds

def generate_preview_gif(img, trajectories, smooth):
    """
    trajectories: List of dicts {'points': [], 'speeds': [], 'color_idx': int}
    """
    if img is None: return None
    
    # Check if any trajectory has points
    has_points = any(len(t['points']) > 0 for t in trajectories)
    if not has_points: return None
    
    preview_frames = 20
    h, w = img.shape[:2]
    target_size = 320
    scale = 1.0
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_small = Image.fromarray(img).resize((new_w, new_h))
    else:
        img_small = Image.fromarray(img)
        
    base_frame = np.array(img_small)
    
    # Pre-calculate tracks for all trajectories
    all_tracks = []
    for traj in trajectories:
        if not traj['points']:
            all_tracks.append(None)
            continue
        tracks, _, _, _, _, _ = interpolate_trajectory(traj['points'], num_frames=preview_frames, smooth=smooth, speeds=traj['speeds'])
        tracks_scaled = tracks[:, 0, :] * scale
        all_tracks.append((tracks_scaled, traj['color_idx']))
    
    frames = []
    for i in range(preview_frames):
        frame = base_frame.copy()
        frame_pil = Image.fromarray(frame).convert("RGBA")
        
        # Create a transparent overlay for dots
        overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for track_data in all_tracks:
            if track_data is None: continue
            tracks_scaled, color_idx = track_data
            
            p = tracks_scaled[i]
            
            # Skip invalid points
            if not np.isfinite(p).all():
                continue
                
            r = 6
            base_color = COLOR_MAP[color_idx % len(COLOR_MAP)]
            # Add alpha channel (e.g., 180/255)
            color = base_color + (180,)
            
            draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=color, outline=None)
        
        # Composite overlay onto frame
        frame_composed = Image.alpha_composite(frame_pil, overlay)
        frames.append(np.array(frame_composed.convert("RGB")))
        
    temp_path = os.path.join("gradio_results", "temp_preview.gif")
    os.makedirs("gradio_results", exist_ok=True)
    imageio.mimsave(temp_path, frames, duration=0.1, loop=0)
    return temp_path

def update_ui(img, trajectories, current_idx, smooth):
    # Get current trajectory data
    current_traj = trajectories[current_idx]
    points = current_traj['points']
    speeds = current_traj['speeds']
    color_idx = current_traj['color_idx']
    
    # Update Input Image (Show ONLY active trajectory)
    img_with_curve = draw_trajectory_on_image(img, points, smooth, speeds, color_idx)
    
    # Update Preview GIF (Show ALL trajectories)
    preview_gif = generate_preview_gif(img, trajectories, smooth)
    
    # Update Speed Editor (Show active trajectory speeds)
    speed_editor = draw_speed_curve_editor(speeds)
        
    return img_with_curve, preview_gif, speed_editor

def process_image_click(img, evt: gr.SelectData, trajectories, current_idx, smooth):
    if img is None:
        return img, trajectories, None, None
    
    # Check if speeds have been modified (not all 1.0)
    current_speeds = trajectories[current_idx]['speeds']
    if any(abs(s - 1.0) > 1e-6 for s in current_speeds):
        gr.Warning("Cannot add points after adjusting speed. Please clear trajectory to restart.")
        # Return current state without changes
        img_curve, prev_gif, speed_editor = update_ui(img, trajectories, current_idx, smooth)
        return img_curve, trajectories, prev_gif, speed_editor
    
    x, y = evt.index[0], evt.index[1]
    
    # Update current trajectory
    trajectories[current_idx]['points'].append((x, y))
    trajectories[current_idx]['speeds'].append(1.0)
    
    img_curve, prev_gif, speed_editor = update_ui(img, trajectories, current_idx, smooth)
    
    return img_curve, trajectories, prev_gif, speed_editor

def on_speed_editor_click(img, evt: gr.SelectData, trajectories, current_idx, smooth):
    speeds = trajectories[current_idx]['speeds']
    speeds = process_speed_curve_click(evt, speeds)
    trajectories[current_idx]['speeds'] = speeds
    
    img_curve, prev_gif, speed_editor = update_ui(img, trajectories, current_idx, smooth)
    return img_curve, trajectories, prev_gif, speed_editor

def clear_trajectory(original_img, trajectories, current_idx):
    # Clear ONLY current trajectory
    trajectories[current_idx]['points'] = []
    trajectories[current_idx]['speeds'] = []
    
    empty_editor = draw_speed_curve_editor([])
    
    # Re-render UI
    # Note: Input image will be clean (if only one traj) or show nothing for this traj
    # Preview might still show other trajectories
    img_curve, prev_gif, speed_editor = update_ui(original_img, trajectories, current_idx, True) # Assume smooth=True for clear
    
    return img_curve, trajectories, prev_gif, speed_editor

def delete_trajectory(img, trajectories, current_idx, smooth):
    if len(trajectories) <= 1:
        # If only one, just clear it
        return clear_trajectory(img, trajectories, 0)
    
    # Remove current
    trajectories.pop(current_idx)
    
    # Adjust index
    new_idx = max(0, current_idx - 1)
    
    # Update choices
    choices = [f"Trajectory {i+1} {COLOR_EMOJIS[t['color_idx']%len(COLOR_EMOJIS)]}" for i, t in enumerate(trajectories)]
    choices.append("Create New...")
    
    new_val = choices[new_idx]
    
    img_curve, prev_gif, speed_editor = update_ui(img, trajectories, new_idx, smooth)
    
    return trajectories, new_idx, gr.Dropdown(choices=choices, value=new_val), img_curve, prev_gif, speed_editor

def load_tracks_to_editor(img, track_file, smooth):
    if img is None:
        return None, None, gr.Dropdown(), None, None, None, "Please upload an image first."
    if track_file is None:
        return None, None, gr.Dropdown(), None, None, None, "Please upload a tracks.npy file."
        
    try:
        tracks = np.load(track_file) # [F, N, 2] or [1, F, N, 2]
    except Exception as e:
        return None, None, gr.Dropdown(), None, None, None, f"Error loading file: {e}"
        
    if tracks.ndim == 4:
        tracks = tracks[0] # [F, N, 2]
        
    if tracks.ndim != 3:
        return None, None, gr.Dropdown(), None, None, None, f"Invalid shape: {tracks.shape}"
        
    num_frames, num_tracks, _ = tracks.shape
    
    new_trajectories = []
    
    # Downsample to ~5 keypoints
    indices = np.linspace(0, num_frames-1, 5, dtype=int)
    
    for i in range(num_tracks):
        points = []
        speeds = []
        
        track_points = tracks[:, i, :] # [F, 2]
        
        for idx in indices:
            p = track_points[idx]
            points.append((float(p[0]), float(p[1])))
            speeds.append(1.0) # Reset speed to 1.0 as we can't easily infer it
            
        new_trajectories.append({
            'points': points,
            'speeds': speeds,
            'color_idx': i % len(COLOR_MAP)
        })
        
    if len(new_trajectories) > len(COLOR_MAP):
        gr.Warning(f"Loaded {len(new_trajectories)} trajectories. Colors will repeat after {len(COLOR_MAP)}.")
        
    # Update UI
    choices = [f"Trajectory {i+1} {COLOR_EMOJIS[t['color_idx']%len(COLOR_EMOJIS)]}" for i, t in enumerate(new_trajectories)]
    choices.append("Create New...")
    
    current_idx = 0
    img_curve, prev_gif, speed_editor = update_ui(img, new_trajectories, current_idx, smooth)
    
    return new_trajectories, current_idx, gr.Dropdown(choices=choices, value=choices[0]), img_curve, prev_gif, speed_editor, "Tracks loaded successfully!"

def export_data(img, trajectories, save_dir, smooth):
    if not trajectories or img is None:
        return "Please upload an image."
    
    os.makedirs(save_dir, exist_ok=True)
    img_pil = Image.fromarray(img).convert("RGB")
    img_path = os.path.join(save_dir, "input_image.jpg")
    img_pil.save(img_path)
    
    all_tracks = []
    all_vis = []
    
    for traj in trajectories:
        if not traj['points']: continue
        tracks, visibility, _, _, _, _ = interpolate_trajectory(traj['points'], num_frames=81, smooth=smooth, speeds=traj['speeds'])
        all_tracks.append(tracks) # [F, 1, 2]
        all_vis.append(visibility) # [F, 1]
        
    if not all_tracks:
        return "No valid trajectories to export."
        
    # Stack: [F, N, 2]
    tracks_combined = np.concatenate(all_tracks, axis=1)
    vis_combined = np.concatenate(all_vis, axis=1)
    
    # Expand to [1, F, N, 2]
    tracks_save = tracks_combined[None, ...]
    vis_save = vis_combined[None, ...]
    
    np.save(os.path.join(save_dir, "tracks.npy"), tracks_save)
    np.save(os.path.join(save_dir, "visibility.npy"), vis_save)
    
    return f"Saved to {save_dir}"

def get_gallery_images(gallery_dir):
    if not os.path.exists(gallery_dir):
        return []
    files = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.mp4') and not f.endswith('_vis.mp4') and "temp" not in f]
    return sorted(files, key=os.path.getmtime, reverse=True)

# --- Global Model Holder ---
wan_move_model = None
model_args = None
model_cfg = None
device = None

def init_model(args):
    global wan_move_model, model_args, model_cfg, device
    model_args = args
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    cfg = WAN_CONFIGS[args.task]
    if args.dtype == "fp32":
        cfg.param_dtype = torch.float32
    elif args.dtype == "fp16":
        cfg.param_dtype = torch.float16
    elif args.dtype == "bf16":
        cfg.param_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    
    model_cfg = cfg
    
    logging.info(f"Initializing WanMove pipeline with dtype {args.dtype}...")
    wan_move_model = wan.WanMove(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    logging.info("Model initialized.")

def generate_video(img, trajectories, prompt, save_dir, track_file, vis_file, smooth):
    global wan_move_model, model_args, model_cfg, device
    
    if img is None:
        return None, None, "Please upload an input image."
    
    # Lazy loading
    if wan_move_model is None:
        logging.info("Lazy loading model...")
        if model_args is None:
             return None, None, "Model arguments not initialized."
        init_model(model_args)

    os.makedirs(save_dir, exist_ok=True)
    img_pil = Image.fromarray(img).convert("RGB")
    img_path = os.path.join(save_dir, "input_image.jpg")
    img_pil.save(img_path)

    # Determine source of tracks
    if track_file is not None and vis_file is not None:
        logging.info("Using uploaded trajectory files.")
        try:
            tracks = np.load(track_file)
            visibility = np.load(vis_file)
        except Exception as e:
            return None, None, f"Error loading files: {e}"
        
        np.save(os.path.join(save_dir, "tracks.npy"), tracks)
        np.save(os.path.join(save_dir, "visibility.npy"), visibility)
        
        if tracks.ndim == 3: tracks_input = tracks[None, ...]
        elif tracks.ndim == 4: tracks_input = tracks
        else: return None, None, f"Invalid tracks shape: {tracks.shape}"
            
        if visibility.ndim == 2: visibility_input = visibility[None, ...]
        elif visibility.ndim == 3: visibility_input = visibility
        else: return None, None, f"Invalid visibility shape: {visibility.shape}"

    elif trajectories:
        logging.info("Using drawn trajectories.")
        export_msg = export_data(img, trajectories, save_dir, smooth)
        if "Please" in export_msg or "No valid" in export_msg:
            return None, None, export_msg
            
        # Re-interpolate for generation
        all_tracks = []
        all_vis = []
        for traj in trajectories:
            if not traj['points']: continue
            t, v, _, _, _, _ = interpolate_trajectory(traj['points'], num_frames=model_args.frame_num, smooth=smooth, speeds=traj['speeds'])
            all_tracks.append(t)
            all_vis.append(v)
            
        tracks_combined = np.concatenate(all_tracks, axis=1) # [F, N, 2]
        vis_combined = np.concatenate(all_vis, axis=1)
        
        tracks_input = tracks_combined[None, ...]
        visibility_input = vis_combined[None, ...]
        
    else:
        return None, None, "Please draw a trajectory OR upload track/visibility files."
    
    logging.info(f"Generating video for prompt: {prompt}")
    
    seed = model_args.base_seed if model_args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    w, h = img_pil.size
    aspect_ratio = w / h
    
    supported_sizes = SUPPORTED_SIZES.get(model_args.task, [])
    if not supported_sizes:
        target_size_name = model_args.size
    else:
        best_diff = float('inf')
        target_size_name = supported_sizes[0]
        for size_name in supported_sizes:
            sw, sh = SIZE_CONFIGS[size_name]
            s_ar = sw / sh
            diff = abs(math.log(aspect_ratio) - math.log(s_ar))
            if diff < best_diff:
                best_diff = diff
                target_size_name = size_name
                
    logging.info(f"Input Image: {w}x{h} (AR={aspect_ratio:.2f}). Selected target size: {target_size_name}")
    max_area = MAX_AREA_CONFIGS[target_size_name]
    
    current_shift = model_args.sample_shift
    if "i2v" in model_args.task and target_size_name in ["832*480", "480*832"]:
         current_shift = 3.0

    try:
        video = wan_move_model.generate(
            prompt,
            img_pil,
            tracks_input,
            visibility_input,
            max_area=max_area,
            frame_num=model_args.frame_num,
            shift=current_shift,
            sample_solver=model_args.sample_solver,
            sampling_steps=model_args.sample_steps,
            guide_scale=model_args.sample_guide_scale,
            seed=seed,
            offload_model=model_args.offload_model,
            eval_bench=False
        )
        
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        save_filename = f"{formatted_time}_{formatted_prompt}.mp4"
        vis_filename = f"{formatted_time}_{formatted_prompt}_vis.mp4"
        save_path = os.path.join(save_dir, save_filename)
        vis_path = os.path.join(save_dir, vis_filename)
        
        cache_video(
            tensor=video[None],
            save_file=save_path,
            fps=model_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
            
        video_vis_input = ((video.unsqueeze(0).permute(0, 2, 1, 3, 4).float() + 1.0) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
        track_video = draw_tracks_on_video(video_vis_input, torch.from_numpy(tracks_input), torch.from_numpy(visibility_input))
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).permute(1,0,2,3).mul(2).sub(1).to(device)
        
        if track_video.shape[-2:] != video.shape[-2:]:
             track_video = torch.nn.functional.interpolate(track_video, size=video.shape[-2:], mode='bilinear', align_corners=False)

        cache_video(
            tensor=track_video[None],
            save_file=vis_path,
            fps=model_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
                
        return save_path, vis_path, f"Generation successful! Saved to {save_path}"
        
    except Exception as e:
        logging.exception("Generation failed")
        return None, None, f"Generation failed: {str(e)}"

if __name__ == "__main__":
    args = _parse_args()
    model_args = args # Store args for lazy loading
    
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <div align="center">
            <h1>Wan-Move Local Demo</h1>
            
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap;">
                <a href="https://arxiv.org/abs/2512.08765"><img src="https://img.shields.io/badge/ArXiv-Paper-brown" alt="Paper"></a>
                <a href="https://github.com/ali-vilab/Wan-Move"><img src="https://img.shields.io/badge/GitHub-Code-blue" alt="Code"></a>
                <a href="https://huggingface.co/Ruihang/Wan-Move-14B-480P"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow" alt="Model"></a>
                <a href="https://www.youtube.com/watch?v=_5Cy7Z2NQJQ"><img src="https://img.shields.io/badge/YouTube-Video-red" alt="YouTube"></a>
                <a href="https://wan-move.github.io/"><img src="https://img.shields.io/badge/Demo-Page-bron" alt="Demo"></a>
            </div>
            </div>
            """
        )
        
        with gr.Tabs():
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="1. Input Image (Click to add points)", type="numpy")
                    
                    with gr.Column(scale=1):
                        preview_output = gr.Image(label="2. Preview (GIF)", interactive=False)
                        
                    with gr.Column(scale=1):
                        speed_editor_output = gr.Image(label="3. Speed Control (Adjust AFTER adding points)", type="numpy", interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        traj_dropdown = gr.Dropdown(label="Select Trajectory", choices=["Trajectory 1 🔵", "Create New..."], value="Trajectory 1 🔵", interactive=True)
                    with gr.Column(scale=1):
                        smooth_chk = gr.Checkbox(label="Smooth Curve (Spline)", value=True)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            prompt_input = gr.Textbox(label="Prompt", value="A video of...")
                            save_dir_input = gr.Textbox(label="Save Directory", value="gradio_results")
                            
                            with gr.Accordion("Upload Trajectory", open=False):
                                gr.Markdown("**Note:** Uploaded files are used directly for generation (High Precision). 'Load Tracks to Editor' approximates them with 5 points (Low Precision).")
                                with gr.Row():
                                    track_file_input = gr.File(label="Upload tracks.npy", type="filepath")
                                    vis_file_input = gr.File(label="Upload visibility.npy", type="filepath")
                                load_tracks_btn = gr.Button("Load Tracks to Editor")
                        
                        with gr.Row():
                            clear_btn = gr.Button("Clear Trajectory")
                            delete_btn = gr.Button("Delete Trajectory", variant="stop")
                            export_btn = gr.Button("Export Data")
                            gen_btn = gr.Button("Generate Video", variant="primary")
                            
                        status_output = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Generated Video")
                        with gr.Accordion("Trajectory Visualization Video", open=False):
                            vis_video_output = gr.Video(label="Visualized Trajectory")
                
                # State
                # List of dicts: {'points': [], 'speeds': [], 'color_idx': int}
                trajectories_state = gr.State([{'points': [], 'speeds': [], 'color_idx': 0}])
                current_traj_index = gr.State(0)
                original_img_state = gr.State(None)
                
                # Events
                def on_upload(img):
                    empty_editor = draw_speed_curve_editor([])
                    # Reset to single empty trajectory
                    init_traj = [{'points': [], 'speeds': [], 'color_idx': 0}]
                    return img, init_traj, 0, img, None, empty_editor, gr.Dropdown(choices=["Trajectory 1 🔵", "Create New..."], value="Trajectory 1 🔵")
                
                img_input.upload(on_upload, img_input, [img_input, trajectories_state, current_traj_index, original_img_state, preview_output, speed_editor_output, traj_dropdown])
                
                def on_track_file_upload(file):
                    if file:
                        gr.Info("Tip: Uploaded tracks are High Precision. Loading them to the editor will simplify them to 5 points (Low Precision).")
                
                track_file_input.upload(on_track_file_upload, track_file_input, None)
                
                # Dropdown Change
                def on_traj_change(img, trajectories, dropdown_val, smooth):
                    # Parse value to get index or create new
                    if dropdown_val == "Create New...":
                        new_idx = len(trajectories)
                        
                        # Warning if exceeding color map
                        if new_idx >= len(COLOR_MAP):
                            gr.Warning(f"Trajectory count ({new_idx+1}) exceeds distinct colors ({len(COLOR_MAP)}). Colors will repeat.")
                            
                        new_color_idx = new_idx % len(COLOR_MAP)
                        trajectories.append({'points': [], 'speeds': [], 'color_idx': new_color_idx})
                        
                        # Update choices
                        choices = [f"Trajectory {i+1} {COLOR_EMOJIS[t['color_idx']%len(COLOR_EMOJIS)]}" for i, t in enumerate(trajectories)]
                        choices.append("Create New...")
                        
                        new_val = choices[new_idx]
                        
                        img_curve, prev_gif, speed_editor = update_ui(img, trajectories, new_idx, smooth)
                        return trajectories, new_idx, gr.Dropdown(choices=choices, value=new_val), img_curve, prev_gif, speed_editor
                    else:
                        # Extract index from string "Trajectory N ..."
                        try:
                            match = re.search(r'Trajectory (\d+)', dropdown_val)
                            if match:
                                idx = int(match.group(1)) - 1
                            else:
                                idx = 0
                        except:
                            idx = 0
                        
                        img_curve, prev_gif, speed_editor = update_ui(img, trajectories, idx, smooth)
                        return trajectories, idx, gr.Dropdown(), img_curve, prev_gif, speed_editor

                traj_dropdown.change(on_traj_change, 
                                     [original_img_state, trajectories_state, traj_dropdown, smooth_chk],
                                     [trajectories_state, current_traj_index, traj_dropdown, img_input, preview_output, speed_editor_output])

                # Load Tracks Button
                load_tracks_btn.click(load_tracks_to_editor,
                                      [original_img_state, track_file_input, smooth_chk],
                                      [trajectories_state, current_traj_index, traj_dropdown, img_input, preview_output, speed_editor_output, status_output])

                # Click Handler for Image (Add Points)
                img_input.select(process_image_click, 
                                 [original_img_state, trajectories_state, current_traj_index, smooth_chk], 
                                 [img_input, trajectories_state, preview_output, speed_editor_output])
                
                # Click Handler for Speed Editor (Adjust Speed)
                speed_editor_output.select(on_speed_editor_click,
                                           [original_img_state, trajectories_state, current_traj_index, smooth_chk],
                                           [img_input, trajectories_state, preview_output, speed_editor_output])

                # Smooth Checkbox
                def on_smooth_change(img, trajectories, current_idx, smooth):
                    img_curve, prev_gif, speed_editor = update_ui(img, trajectories, current_idx, smooth)
                    return img_curve, prev_gif, speed_editor
                
                smooth_chk.change(on_smooth_change, [original_img_state, trajectories_state, current_traj_index, smooth_chk], [img_input, preview_output, speed_editor_output])

                # Clear
                clear_btn.click(clear_trajectory, 
                                [original_img_state, trajectories_state, current_traj_index], 
                                [img_input, trajectories_state, preview_output, speed_editor_output])
                
                # Delete
                def on_delete(img, trajectories, current_idx, smooth):
                    # If only one, clear it but return same dropdown
                    if len(trajectories) <= 1:
                        img_curve, trajectories, prev_gif, speed_editor = clear_trajectory(img, trajectories, 0)
                        return trajectories, 0, gr.Dropdown(), img_curve, prev_gif, speed_editor
                    else:
                        return delete_trajectory(img, trajectories, current_idx, smooth)

                delete_btn.click(on_delete,
                                 [original_img_state, trajectories_state, current_traj_index, smooth_chk],
                                 [trajectories_state, current_traj_index, traj_dropdown, img_input, preview_output, speed_editor_output])
                
                export_btn.click(export_data, [original_img_state, trajectories_state, save_dir_input, smooth_chk], status_output)
                
                gen_btn.click(generate_video, 
                              [original_img_state, trajectories_state, prompt_input, save_dir_input, track_file_input, vis_file_input, smooth_chk], 
                              [video_output, vis_video_output, status_output])
                
            with gr.Tab("Gallery"):
                refresh_btn = gr.Button("Refresh Gallery")
                gallery = gr.Gallery(label="Generated Videos", columns=4)
                
                refresh_btn.click(get_gallery_images, save_dir_input, gallery)

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

# --- Speed Curve Editor Utilities ---

def get_display_range(speeds):
    """
    Calculates the display range (y_min, y_max) for the speed editor.
    """
    if not speeds:
        return 0.0, 2.0
        
    s_min = min(speeds)
    s_max = max(speeds)
    
    if s_min == s_max:
        # Default range centered around the value
        # If 1.0, range 0-2.
        # If 3.0, range 1-5?
        center = s_min
        span = 1.0
        return max(0, center - span), center + span
        
    # Add padding
    span = s_max - s_min
    padding = max(0.5, span * 0.3) # At least 0.5 padding
    
    y_min = max(0, s_min - padding)
    y_max = s_max + padding
    
    return y_min, y_max

def draw_speed_curve_editor(speeds, width=800, height=600):
    """
    Draws an interactive speed curve editor with dynamic range.
    Resolution increased for better visual quality.
    """
    y_min, y_max = get_display_range(speeds)
    
    if not speeds:
        # Return empty placeholder
        img = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Scale font size based on resolution
        font_size = 30
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = None
            
        text = "Add points to edit speed"
        # Simple centering
        draw.text((width//2 - 150, height//2), text, fill=(0,0,0), font=font)
        return np.array(img)
        
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw Grid Lines and Labels
    total_span = y_max - y_min
    
    if total_span <= 2.0: step = 0.5
    elif total_span <= 5.0: step = 1.0
    else: step = 2.0
    
    start_tick = (int(y_min / step) + 1) * step
    if start_tick < y_min: start_tick += step
    
    current_tick = start_tick
    while current_tick < y_max:
        ratio = (current_tick - y_min) / (y_max - y_min)
        y_pixel = height - (ratio * height)
        
        draw.line([(60, y_pixel), (width, y_pixel)], fill=(220, 220, 220), width=2)
        
        label = f"{current_tick:.1f}x"
        draw.text((5, y_pixel - 10), label, fill=(120, 120, 120))
        
        current_tick += step
        
    # Highlight 1.0x if visible
    if y_min <= 1.0 <= y_max:
        ratio = (1.0 - y_min) / (y_max - y_min)
        y_pixel = height - (ratio * height)
        draw.line([(60, y_pixel), (width, y_pixel)], fill=(180, 180, 180), width=3)
        draw.text((5, y_pixel - 10), "1.0x", fill=(80, 80, 80))

    # Plot points
    num_points = len(speeds)
    coords = []
    for i, s in enumerate(speeds):
        if num_points == 1:
            x = 60 + (width - 60) / 2
        else:
            x = 60 + i * ((width - 60) / (num_points - 1))
            
        s_clamped = max(y_min, min(y_max, s))
        ratio = (s_clamped - y_min) / (y_max - y_min)
        y = height - (ratio * height)
        coords.append((x, y))
        
    # Draw smooth curve
    if num_points > 1:
        try:
            x_vals = [c[0] for c in coords]
            y_vals = [c[1] for c in coords]
            f = interp1d(x_vals, y_vals, kind='cubic')
            
            x_plot = np.linspace(60, width, 200)
            y_plot = f(x_plot)
            y_plot = np.clip(y_plot, 0, height)
            
            plot_points = list(zip(x_plot, y_plot))
            draw.line(plot_points, fill=(0, 0, 255), width=4)
        except:
            draw.line(coords, fill=(0, 0, 255), width=4)
            
    # Draw handles
    r = 12
    for cx, cy in coords:
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 0, 0), outline=(0,0,0))
        
    return np.array(img)

def process_speed_curve_click(evt: gr.SelectData, speeds):
    """
    Handle click on speed curve editor with dynamic range.
    """
    if not speeds:
        return speeds
        
    width = 800
    height = 600 # Updated height
    y_min, y_max = get_display_range(speeds)
    
    click_x, click_y = evt.index[0], evt.index[1]
    
    eff_width = width - 60
    eff_click_x = click_x - 60
    
    num_points = len(speeds)
    if num_points == 1:
        idx = 0
    else:
        spacing = eff_width / (num_points - 1)
        idx = int(round(eff_click_x / spacing))
        idx = max(0, min(num_points - 1, idx))
        
    ratio = (height - click_y) / height
    new_speed = y_min + ratio * (y_max - y_min)
    new_speed = max(0.1, new_speed)
    
    speeds[idx] = new_speed
    
    return speeds

def generate_preview_gif(img, points, smooth, speeds):
    if img is None or not points: return None
    
    # Generate low-res, low-fps GIF for preview
    preview_frames = 20
    tracks, _, _, _, _, _ = interpolate_trajectory(points, num_frames=preview_frames, smooth=smooth, speeds=speeds)
    
    h, w = img.shape[:2]
    
    # Resize for faster preview generation
    target_size = 320
    scale = 1.0
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_small = Image.fromarray(img).resize((new_w, new_h))
    else:
        img_small = Image.fromarray(img)
        
    base_frame = np.array(img_small)
    tracks_scaled = tracks[:, 0, :] * scale
    
    frames = []
    for i in range(preview_frames):
        frame = base_frame.copy()
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        
        # Draw current position
        p = tracks_scaled[i]
        r = 6 
        draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=(255, 255, 0), outline=(0,0,0))
        
        frames.append(np.array(frame_pil))
        
    # Save to temp file
    temp_path = os.path.join("gradio_results", "temp_preview.gif")
    os.makedirs("gradio_results", exist_ok=True)
    
    # Use imageio to write GIF
    # duration is seconds per frame. fps=10 -> 0.1s
    imageio.mimsave(temp_path, frames, duration=0.1, loop=0)
    
    return temp_path

def update_ui(img, points, smooth, speeds):
    # Consolidate interpolation calls if needed, but for now just optimize the heavy video gen
    
    # Update Input Image (Static Curve)
    img_with_curve = draw_trajectory_on_image(img, points, smooth, speeds)
    
    # Update Preview GIF
    preview_gif = generate_preview_gif(img, points, smooth, speeds)
    
    # Update Speed Editor Image
    speed_editor = draw_speed_curve_editor(speeds)
        
    return img_with_curve, preview_gif, speed_editor

def process_image_click(img, evt: gr.SelectData, points, smooth, speeds):
    if img is None:
        return img, points, speeds, None, None
    
    x, y = evt.index[0], evt.index[1]
    points.append((x, y))
    
    # Add default speed for new point
    if speeds is None: speeds = []
    speeds.append(1.0)
    
    img_curve, prev_gif, speed_editor = update_ui(img, points, smooth, speeds)
    
    return img_curve, points, speeds, prev_gif, speed_editor

def on_speed_editor_click(img, evt: gr.SelectData, points, smooth, speeds):
    speeds = process_speed_curve_click(evt, speeds)
    img_curve, prev_gif, speed_editor = update_ui(img, points, smooth, speeds)
    return img_curve, speeds, prev_gif, speed_editor

def clear_trajectory(original_img):
    empty_editor = draw_speed_curve_editor([])
    return original_img, [], [], None, empty_editor

def export_data(img, points_state, save_dir, smooth, speeds):
    if not points_state or img is None:
        return "Please upload an image and draw a trajectory."
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Image
    img_pil = Image.fromarray(img).convert("RGB")
    img_path = os.path.join(save_dir, "input_image.jpg")
    img_pil.save(img_path)
    
    # Interpolate and Save Tracks
    tracks, visibility, _, _, _, _ = interpolate_trajectory(points_state, num_frames=81, smooth=smooth, speeds=speeds)
    # Expand to [1, F, N, 2] and [1, F, N]
    tracks_save = tracks[None, ...]
    visibility_save = visibility[None, ...]
    
    np.save(os.path.join(save_dir, "tracks.npy"), tracks_save)
    np.save(os.path.join(save_dir, "visibility.npy"), visibility_save)
    
    return f"Saved to {save_dir}"

def get_gallery_images(gallery_dir):
    if not os.path.exists(gallery_dir):
        return []
    files = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.mp4') and not f.endswith('_vis.mp4') and "temp" not in f]
    return sorted(files, key=os.path.getmtime, reverse=True)

# --- Global Model Holder ---
wan_move_model = None
model_args = None
model_cfg = None
device = None

def init_model(args):
    global wan_move_model, model_args, model_cfg, device
    model_args = args
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    cfg = WAN_CONFIGS[args.task]
    if args.dtype == "fp32":
        cfg.param_dtype = torch.float32
    elif args.dtype == "fp16":
        cfg.param_dtype = torch.float16
    elif args.dtype == "bf16":
        cfg.param_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    
    model_cfg = cfg
    
    logging.info(f"Initializing WanMove pipeline with dtype {args.dtype}...")
    wan_move_model = wan.WanMove(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    logging.info("Model initialized.")

def generate_video(img, points_state, prompt, save_dir, track_file, vis_file, smooth, speeds):
    global wan_move_model, model_args, model_cfg, device
    
    if img is None:
        return None, None, "Please upload an input image."
    
    if wan_move_model is None:
        return None, None, "Model not initialized."

    os.makedirs(save_dir, exist_ok=True)
    img_pil = Image.fromarray(img).convert("RGB")
    img_path = os.path.join(save_dir, "input_image.jpg")
    img_pil.save(img_path)

    # Determine source of tracks
    if track_file is not None and vis_file is not None:
        logging.info("Using uploaded trajectory files.")
        try:
            tracks = np.load(track_file)
            visibility = np.load(vis_file)
        except Exception as e:
            return None, None, f"Error loading files: {e}"
        
        # Save copies for record
        np.save(os.path.join(save_dir, "tracks.npy"), tracks)
        np.save(os.path.join(save_dir, "visibility.npy"), visibility)
        
        # Handle dimensions
        if tracks.ndim == 3: # [F, N, 2]
            tracks_input = tracks[None, ...]
        elif tracks.ndim == 4:
            tracks_input = tracks
        else:
            return None, None, f"Invalid tracks shape: {tracks.shape}. Expected [F, N, 2] or [1, F, N, 2]."
            
        if visibility.ndim == 2: # [F, N]
            visibility_input = visibility[None, ...]
        elif visibility.ndim == 3:
            visibility_input = visibility
        else:
            return None, None, f"Invalid visibility shape: {visibility.shape}. Expected [F, N] or [1, F, N]."

    elif points_state:
        logging.info("Using drawn trajectory.")
        # Export data first (for record keeping)
        export_msg = export_data(img, points_state, save_dir, smooth, speeds)
        if "Please" in export_msg:
            return None, None, export_msg
            
        tracks, visibility, _, _, _, _ = interpolate_trajectory(points_state, num_frames=model_args.frame_num, smooth=smooth, speeds=speeds)
        tracks_input = tracks[None, ...]
        visibility_input = visibility[None, ...]
        
    else:
        return None, None, "Please draw a trajectory OR upload track/visibility files."
    
    logging.info(f"Generating video for prompt: {prompt}")
    
    seed = model_args.base_seed if model_args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    # Dynamic Size Selection
    w, h = img_pil.size
    aspect_ratio = w / h
    
    supported_sizes = SUPPORTED_SIZES.get(model_args.task, [])
    if not supported_sizes:
        target_size_name = model_args.size
    else:
        best_diff = float('inf')
        target_size_name = supported_sizes[0]
        
        for size_name in supported_sizes:
            sw, sh = SIZE_CONFIGS[size_name]
            s_ar = sw / sh
            diff = abs(math.log(aspect_ratio) - math.log(s_ar))
            if diff < best_diff:
                best_diff = diff
                target_size_name = size_name
                
    logging.info(f"Input Image: {w}x{h} (AR={aspect_ratio:.2f}). Selected target size: {target_size_name}")
    
    # Determine max_area
    max_area = MAX_AREA_CONFIGS[target_size_name]
    
    current_shift = model_args.sample_shift
    if "i2v" in model_args.task and target_size_name in ["832*480", "480*832"]:
         current_shift = 3.0

    try:
        video = wan_move_model.generate(
            prompt,
            img_pil,
            tracks_input,
            visibility_input,
            max_area=max_area,
            frame_num=model_args.frame_num,
            shift=current_shift,
            sample_solver=model_args.sample_solver,
            sampling_steps=model_args.sample_steps,
            guide_scale=model_args.sample_guide_scale,
            seed=seed,
            offload_model=model_args.offload_model,
            eval_bench=False # Not eval bench mode
        )
        
        # Save video
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        save_filename = f"{formatted_time}_{formatted_prompt}.mp4"
        vis_filename = f"{formatted_time}_{formatted_prompt}_vis.mp4"
        save_path = os.path.join(save_dir, save_filename)
        vis_path = os.path.join(save_dir, vis_filename)
        
        # Save main video
        cache_video(
            tensor=video[None],
            save_file=save_path,
            fps=model_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
            
        # Generate and save visualization video
        # Use the generated video as background
        # video is [C, F, H, W] in [-1, 1]
        # Convert to [1, F, C, H, W] in [0, 255] uint8
        video_vis_input = ((video.unsqueeze(0).permute(0, 2, 1, 3, 4).float() + 1.0) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
        
        track_video = draw_tracks_on_video(video_vis_input, torch.from_numpy(tracks_input), torch.from_numpy(visibility_input))
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).permute(1,0,2,3).mul(2).sub(1).to(device)
        
        # Resize track_video to match video shape if needed (should match now since we use video as input)
        if track_video.shape[-2:] != video.shape[-2:]:
             track_video = torch.nn.functional.interpolate(track_video, size=video.shape[-2:], mode='bilinear', align_corners=False)

        cache_video(
            tensor=track_video[None], # Add batch dim
            save_file=vis_path,
            fps=model_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
                
        return save_path, vis_path, f"Generation successful! Saved to {save_path}"
        
    except Exception as e:
        logging.exception("Generation failed")
        return None, None, f"Generation failed: {str(e)}"
        return None, None, f"Generation failed: {str(e)}"
