# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------  Wan-Move 14B ------------------------#

wan_move_14B = EasyDict(__name__='Config: Wan-Move 14B')
wan_move_14B.update(wan_shared_cfg)
wan_move_14B.sample_neg_prompt = "镜头晃动，" + wan_move_14B.sample_neg_prompt

wan_move_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
wan_move_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
wan_move_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
wan_move_14B.clip_dtype = torch.float16
wan_move_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
wan_move_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
wan_move_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
wan_move_14B.vae_stride = (4, 8, 8)

# transformer
wan_move_14B.patch_size = (1, 2, 2)
wan_move_14B.dim = 5120
wan_move_14B.ffn_dim = 13824
wan_move_14B.freq_dim = 256
wan_move_14B.num_heads = 40
wan_move_14B.num_layers = 40
wan_move_14B.window_size = (-1, -1)
wan_move_14B.qk_norm = True
wan_move_14B.cross_attn_norm = True
wan_move_14B.eps = 1e-6
