import os
import sys
import torch
import numpy as np
import copy
import time
import queue
import cv2
import pickle
import glob
from PIL import Image

# 将liveact_core加入路径以解析内部import
current_dir = os.path.dirname(os.path.abspath(__file__))
liveact_core_path = os.path.join(current_dir, 'liveact_core')
if liveact_core_path not in sys.path:
    sys.path.append(liveact_core_path)

from model_liveact.model_memory import WanModel
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from torchvision import transforms
from util_liveact import center_rescale_crop_keep_ratio, get_embedding, get_msk, get_audio_emb
import torchaudio.transforms as T
import torchaudio

from avatars.base_avatar import BaseAvatar
from utils.logger import logger
from utils.image import read_imgs
from utils.device import initialize_device
from registry import register
from fp8_gemm import FP8GemmOptions, enable_fp8_gemm

device = initialize_device()

def load_model(opt):
    logger.info("Loading LiveAct Model...")
    world_size = 1
    
    ckpt_dir = opt.liveact_ckpt
    wan_i2v_model = WanModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    wan_i2v_model = wan_i2v_model.to(dtype=torch.bfloat16)
    
    width, height = [int(_) for _ in opt.liveact_size.split('*')]
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))
    
    for n in range(40):
        wan_i2v_model.blocks[n].self_attn.init_kvidx(frame_len, world_size)
        
    enable_fp8_gemm(wan_i2v_model, options=FP8GemmOptions())
    # 为了防止OOM，强制部分卸载或放到特定设备
    if getattr(opt, 'block_offload', True):
        for name, child in wan_i2v_model.named_children():
            if name != 'blocks':
                child.to(device)
        wan_i2v_model.enable_block_offload(onload_device=device)
    else:
        wan_i2v_model = wan_i2v_model.to(device)
        
    wan_i2v_model.eval()
    
    vae = LightVAE(vae_path=os.path.join(ckpt_dir, 'Wan2.1_VAE.pth'), dtype=torch.bfloat16, device=device, use_lightvae=False, parallel=False)
    
    clip = CLIPModel(
        checkpoint_path=os.path.join(ckpt_dir, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
        tokenizer_path=os.path.join(ckpt_dir, 'xlm-roberta-large'), dtype=torch.bfloat16, device=device)
    clip.model = clip.model.to(device, dtype=torch.bfloat16)
    
    t5_cpu = getattr(opt, 't5_cpu', True)
    text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device='cpu' if t5_cpu else device,
                                  checkpoint_path=os.path.join(ckpt_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
                                  tokenizer_path=os.path.join(ckpt_dir, 'google/umt5-xxl'))
                                  
    audio_encoder = Wav2Vec2Model.from_pretrained(
        opt.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device, dtype=torch.bfloat16).eval()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_dir, local_files_only=True)
    audio_encoder.feature_extractor._freeze_parameters()
    wan_i2v_model.freqs = wan_i2v_model.freqs.to(device)
    
    for _model in [wan_i2v_model, clip.model, audio_encoder, vae.model]:
        for name, param in _model.named_parameters():
            param.requires_grad = False
            
    vae.model.eval()
    
    return {
        'wan': wan_i2v_model,
        'vae': vae,
        'clip': clip,
        'text_encoder': text_encoder,
        'audio_encoder': audio_encoder,
        'wav2vec_feature_extractor': wav2vec_feature_extractor
    }

def load_avatar(opt, avatar_id):
    logger.info(f"Loading LiveAct Avatar: {avatar_id}")
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if len(input_img_list) == 0:
        cond_image_path = None
        frame_list_cycle = []
    else:
        cond_image_path = input_img_list[0]
        frame_list_cycle = read_imgs(input_img_list)
        
    return frame_list_cycle, cond_image_path

def warm_up(batch_size, model):
    logger.info("LiveAct model warmed up.")
    pass

@register("avatar", "liveact")
class LiveActAvatar(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.models = model
        self.frame_list_cycle, self.cond_image_path = avatar
        
        width, height = [int(_) for _ in opt.liveact_size.split('*')]
        self.width = width
        self.height = height
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (height, width))),
            transforms.ToTensor(),
            transforms.Resize((height, width)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # 初始化条件信息
        self._init_cond()
        
        self.audio_buffer = []
        self.tgt_fps = 24
        self.blksz_lst = [6, 8]
        self.vae_stride = (4, 8, 8)
        self.frame_num = (sum(self.blksz_lst) - 1) * 4 + 1
        
        # 记录推理状态
        self.iter_idx = 0
        self.pre_latent = None

    def _init_cond(self):
        prompt = "A talking head video."
        t5_cpu = getattr(self.opt, 't5_cpu', True)
        self.context = [self.models['text_encoder'](texts=prompt, device='cpu' if t5_cpu else device)[0].to(device, dtype=torch.bfloat16)]
        
        image = Image.open(self.cond_image_path).convert("RGB")
        cond_image = self.transform(image).unsqueeze(1).unsqueeze(0).to(device, torch.bfloat16)
        
        clip = self.models['clip']
        clip.model.to(device)
        self.clip_context = clip.visual(cond_image)
        clip.model.cpu()
        
        self.ref_target_masks = torch.ones(3, self.height // 8, self.width // 8).to(device, torch.bfloat16)
        
        frame_num = (sum([6, 8]) - 1) * 4 + 1
        msk = get_msk(frame_num, cond_image, (4, 8, 8), device)
        
        vae = self.models['vae']
        wan_i2v_model = self.models['wan']
        
        video_frames = torch.zeros(
            1, cond_image.shape[1], frame_num - cond_image.shape[2], self.height, self.width
        ).to(cond_image.device, cond_image.dtype)
        padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
        y = vae.encode(padding_frames_pixels_values.to(vae.device)).to(wan_i2v_model.device).unsqueeze(0)
        self.y = torch.concat([msk, y], dim=1)
        
        # Init KV cache
        patch_size = (1, 2, 2)
        frame_len = (self.height // (patch_size[1] * 8)) * (self.width // (patch_size[2] * 8))
        kv_cache_tokens = frame_len * sum([6, 8])
        kv_cache_device = 'cpu' if getattr(self.opt, 'offload_cache', False) else device
        kv_cache_dtype = torch.float8_e4m3fn if getattr(self.opt, 'fp8_kv_cache', True) else torch.bfloat16
        kv_scale_shape = (1, kv_cache_tokens, 40, 1)
        self.timesteps = [torch.tensor([_]).to(device, dtype=torch.float32) for _ in [1000.0, 937.5, 833.33333333, 0.0]]
        
        self.kv_cache = {
            i: {
                layer_id: {
                    'k': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'v': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if getattr(self.opt, 'fp8_kv_cache', True) else None,
                    'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if getattr(self.opt, 'fp8_kv_cache', True) else None,
                    'mean_memory': getattr(self.opt, 'mean_memory', False),
                    'offload_cache': getattr(self.opt, 'offload_cache', False),
                    'fp8_kv_cache': getattr(self.opt, 'fp8_kv_cache', True),
                }
                for layer_id in range(40)
            } for i in range(len(self.timesteps) - 1)
        }

    def inference(self, quit_event):
        logger.info('LiveAct start inference thread')
        
        audio_chunk_size = 16000 # 1s chunk buffer as example for liveact
        
        while not quit_event.is_set():
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            audio_frames = []
            for _ in range(self.batch_size * 2):
                try:
                    audioframe = self.asr.output_queue.get(timeout=0.1)
                    audio_frames.append(audioframe)
                except queue.Empty:
                    break
                    
            if len(audio_frames) == 0:
                continue

            pcm_data = np.concatenate([f.data for f in audio_frames])
            self.audio_buffer.extend(pcm_data.tolist())
            
            # 由于LiveAct比较重，我们用一个固定的音频块大小来做 streaming 演示
            while len(self.audio_buffer) >= audio_chunk_size:
                audio_array = np.array(self.audio_buffer[:audio_chunk_size])
                self.audio_buffer = self.audio_buffer[audio_chunk_size:]
                
                # Resample if needed
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
                
                # Get audio embedding
                audio_embedding = get_embedding(audio_tensor[0], self.models['wav2vec_feature_extractor'], self.models['audio_encoder'], device=device)
                audio_embs = get_audio_emb(audio_embedding, 0, self.frame_num, device)
                
                y_cut = self.y[:, :, :self.frame_num // 4 + 1, ...]
                
                f = self.iter_idx if self.iter_idx <= 1 else 1
                latent = torch.randn(16, self.blksz_lst[f], self.height // 8, self.width // 8,
                                     dtype=torch.bfloat16, device=device)
                                     
                wan_i2v_model = self.models['wan']
                vae = self.models['vae']
                patch_size = (1, 2, 2)
                frame_len = (self.height // (patch_size[1] * 8)) * (self.width // (patch_size[2] * 8))
                
                for i in range(len(self.timesteps) - 1):
                    timestep = self.timesteps[i]
                    arg_c = {'context': self.context, 'clip_fea': self.clip_context, 'ref_target_masks': self.ref_target_masks,
                             'audio': audio_embs, 'y': y_cut[:, :, sum(self.blksz_lst[:f]):sum(self.blksz_lst[:f + 1])],
                             'start_idx': sum(self.blksz_lst[:f]) * frame_len, 'end_idx': sum(self.blksz_lst[:f + 1]) * frame_len,
                             'update_cache': self.iter_idx > 1}
                    noise_pred = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=self.kv_cache[i],
                                               skip_audio=False, **arg_c)[0]

                    dt = self.timesteps[i] - self.timesteps[i + 1]
                    dt = dt / 1000
                    latent = latent + (-noise_pred) * dt[0]
                    
                if f == 0:
                    _latent = latent
                    _videos = vae.decode(_latent.squeeze(0))
                else:
                    _latent = torch.concat([self.pre_latent[:, -3:], latent], dim=1)
                    _videos = vae.decode(_latent.squeeze(0))[:, :, 9:]
                    
                self.pre_latent = latent
                self.iter_idx += 1
                
                video = ((_videos.permute(1, 2, 3, 0) + 1.0) / 2 * 255).cpu().numpy().astype(np.uint8)
                
                for i in range(video.shape[0]):
                    frame = video[i]
                    # BGR convert if needed
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    idx = i % len(self.frame_list_cycle) if len(self.frame_list_cycle) > 0 else 0
                    self.res_frame_queue.put((frame, audio_frames[:2], idx))
                    
        logger.info('LiveAct inference thread stop')
        
    def paste_back_frame(self, pred_frame, idx:int):
        return pred_frame