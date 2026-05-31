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

# 将flashhead_core加入路径以解析内部import
current_dir = os.path.dirname(os.path.abspath(__file__))
flashhead_core_path = os.path.join(current_dir, 'flashhead_core')
if flashhead_core_path not in sys.path:
    sys.path.append(flashhead_core_path)

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
from avatars.base_avatar import BaseAvatar
from avatars.audio_features.mel import MelASR
from utils.logger import logger
from utils.image import read_imgs
from utils.device import initialize_device
from registry import register
from collections import deque

device = initialize_device()

def load_model(opt):
    logger.info("Loading FlashHead Model...")
    world_size = 1
    model_type = getattr(opt, 'flashhead_model_type', 'lite')
    pipeline = get_pipeline(world_size=world_size, ckpt_dir=opt.flashhead_ckpt, wav2vec_dir=opt.wav2vec_dir, model_type=model_type)
    return pipeline

def load_avatar(opt, avatar_id):
    logger.info(f"Loading FlashHead Avatar: {avatar_id}")
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if len(input_img_list) == 0:
        logger.warning(f"No condition images found in {full_imgs_path}")
        cond_image_path = None
        frame_list_cycle = []
    else:
        cond_image_path = input_img_list[0]
        frame_list_cycle = read_imgs(input_img_list)
        
    return frame_list_cycle, cond_image_path

def warm_up(batch_size, model):
    logger.info("FlashHead model warmed up.")
    pass

@register("avatar", "flashhead")
class FlashHeadAvatar(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.pipeline = model
        self.frame_list_cycle, self.cond_image_path = avatar
        
        # 初始化 ASR
        self.asr = MelASR(opt, self)
        self.asr.warm_up()
        
        # 提取推理参数
        self.infer_params = get_infer_params()
        self.sample_rate = self.infer_params['sample_rate']
        self.tgt_fps = self.infer_params['tgt_fps']
        self.frame_num = self.infer_params['frame_num']
        self.motion_frames_num = self.infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        # 初始化条件图
        get_base_data(self.pipeline, cond_image_path_or_dir=self.cond_image_path, base_seed=42, use_face_crop=False)
        
        # 音频缓冲区配置（参考 generate_video.py 的 stream 模式）
        self.audio_chunk_size = self.slice_len * self.sample_rate // self.tgt_fps
        
        # 维护一个连续的audio queue，用于计算audio_embedding
        cached_audio_duration = self.infer_params['cached_audio_duration']
        self.cached_audio_length_sum = self.sample_rate * cached_audio_duration
        self.audio_end_idx = cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        
        self.audio_dq = deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        self.is_first_chunk = True
        
        # 预分配静音帧（当没有音频输入时显示静态图）
        self.silent_frames = []
        
        logger.info(f"FlashHead Avatar initialized: slice_len={self.slice_len}, audio_chunk_size={self.audio_chunk_size}")

    def inference(self, quit_event):
        logger.info('FlashHead start inference thread')
        
        while not quit_event.is_set():
            try:
                # 从 ASR 获取音频特征
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            # 收集音频帧 (PCM 数据)
            audio_frames = []
            for _ in range(self.batch_size * 2):
                try:
                    audioframe = self.asr.output_queue.get(timeout=0.1)
                    audio_frames.append(audioframe)
                except queue.Empty:
                    break
                    
            if len(audio_frames) == 0:
                continue

            # 将 pcm 组装起来
            pcm_data = np.concatenate([f.data for f in audio_frames])
            
            # 直接处理这段音频（参考 generate_video.py 的 stream 模式）
            # 将音频加入 deque
            self.audio_dq.extend(pcm_data.tolist())
            audio_array = np.array(self.audio_dq)
            
            # 检查是否有足够的音频进行推理
            # 需要至少 frame_num 帧对应的音频长度
            min_audio_length = self.frame_num * self.sample_rate // self.tgt_fps
            if len(audio_array) < min_audio_length:
                continue
            
            try:
                # 音频编码
                audio_embedding = get_audio_embedding(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
                
                # 生成视频
                video = run_pipeline(self.pipeline, audio_embedding)
                
                # 处理 motion frames
                if not self.is_first_chunk:
                    video = video[self.motion_frames_num:]
                else:
                    self.is_first_chunk = False
                
                # video 是 (T, H, W, C) 的 numpy 数组，值范围 [0, 255]
                # 转换为 uint8 并放入队列
                video_np = video.cpu().numpy().astype(np.uint8)
                
                # 将生成的帧放入 res_frame_queue
                # 每帧对应一个音频块
                audio_per_frame = len(audio_frames) // video_np.shape[0] if video_np.shape[0] > 0 else 1
                
                for i in range(video_np.shape[0]):
                    frame = video_np[i]
                    # 获取对应的音频帧
                    start_idx = i * audio_per_frame
                    end_idx = min((i + 1) * audio_per_frame, len(audio_frames))
                    frame_audio = audio_frames[start_idx:end_idx] if start_idx < len(audio_frames) else audio_frames[:1]
                    
                    idx = i % len(self.frame_list_cycle) if len(self.frame_list_cycle) > 0 else 0
                    self.res_frame_queue.put((frame, frame_audio, idx))
                    
            except Exception as e:
                logger.error(f"FlashHead inference error: {e}")
                import traceback
                traceback.print_exc()
                continue
                    
        logger.info('FlashHead inference thread stop')
        
    def paste_back_frame(self, pred_frame, idx:int):
        # FlashHead 生成的是完整图像，不需要 paste_back
        return pred_frame
