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
        logger.info("[FlashHead] Initializing MelASR...")
        try:
            self.asr = MelASR(opt, self)
            self.asr.warm_up()
            logger.info("[FlashHead] MelASR initialized and warmed up successfully.")
        except Exception as e:
            logger.error(f"[FlashHead] MelASR initialization failed: {e}")
            raise
        
        # 提取推理参数
        self.infer_params = get_infer_params()
        self.sample_rate = self.infer_params['sample_rate']
        self.tgt_fps = self.infer_params['tgt_fps']
        self.frame_num = self.infer_params['frame_num']
        self.motion_frames_num = self.infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        logger.info(f"[FlashHead] Infer params: sample_rate={self.sample_rate}, tgt_fps={self.tgt_fps}, "
                    f"frame_num={self.frame_num}, motion_frames_num={self.motion_frames_num}, slice_len={self.slice_len}")
        
        # 初始化条件图
        logger.info(f"[FlashHead] Initializing base data with cond_image_path={self.cond_image_path}")
        get_base_data(self.pipeline, cond_image_path_or_dir=self.cond_image_path, base_seed=42, use_face_crop=False)
        logger.info("[FlashHead] Base data initialized.")
        
        # 音频缓冲区配置（参考 generate_video.py 的 stream 模式）
        self.audio_chunk_size = self.slice_len * self.sample_rate // self.tgt_fps
        
        # 维护一个连续的audio queue，用于计算audio_embedding
        cached_audio_duration = self.infer_params['cached_audio_duration']
        self.cached_audio_length_sum = self.sample_rate * cached_audio_duration
        self.audio_end_idx = cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        
        self.audio_dq = deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        self.is_first_chunk = True
        self.chunk_idx = 0
        
        # 预分配静音帧（当没有音频输入时显示静态图）
        self.silent_frames = []
        
        logger.info(f"[FlashHead] Avatar initialized: slice_len={self.slice_len}, "
                    f"audio_chunk_size={self.audio_chunk_size}, cached_audio_length_sum={self.cached_audio_length_sum}, "
                    f"audio_start_idx={self.audio_start_idx}, audio_end_idx={self.audio_end_idx}")

    def inference(self, quit_event):
        logger.info('[FlashHead] Start inference thread')
        
        infer_count = 0
        infer_time_total = 0.0
        
        while not quit_event.is_set():
            try:
                # 从 ASR 获取音频特征（用于触发推理节奏）
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 收集音频帧 (PCM 数据)
            audio_frames = []
            is_all_silence = True
            
            try:
                for _ in range(self.batch_size * 2):
                    try:
                        audioframe = self.asr.output_queue.get(timeout=0.1)
                        audio_frames.append(audioframe)
                        if audioframe.type == 0:
                            is_all_silence = False
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"[FlashHead] Error collecting audio frames: {e}")
                continue
                    
            if len(audio_frames) == 0:
                continue
            
            # 全静音时显示静态图片（参考 base_avatar.py 逻辑）
            if is_all_silence:
                logger.debug(f"[FlashHead] Silence detected, outputting static frames. audio_frames={len(audio_frames)}")
                for i in range(self.batch_size):
                    idx = i % len(self.frame_list_cycle) if len(self.frame_list_cycle) > 0 else 0
                    frame_audio = audio_frames[i*2:i*2+2] if i*2+2 <= len(audio_frames) else audio_frames[:1]
                    self.res_frame_queue.put((None, frame_audio, idx))
                continue
            
            try:
                # 将 pcm 组装起来
                pcm_data = np.concatenate([f.data for f in audio_frames])
                logger.debug(f"[FlashHead] Collected audio frames: {len(audio_frames)}, pcm_data shape: {pcm_data.shape}")
                
                # 将音频加入 deque（参考 generate_video.py 的 stream 模式）
                self.audio_dq.extend(pcm_data.tolist())
                audio_array = np.array(self.audio_dq)
                
                logger.debug(f"[FlashHead] Audio deque length: {len(self.audio_dq)}, audio_array shape: {audio_array.shape}")
                
                # 检查是否有足够的音频进行推理
                # 需要至少 frame_num 帧对应的音频长度
                min_audio_length = self.frame_num * self.sample_rate // self.tgt_fps
                if len(audio_array) < min_audio_length:
                    logger.debug(f"[FlashHead] Audio too short for inference: {len(audio_array)} < {min_audio_length}")
                    continue
                
                # 音频编码
                logger.debug(f"[FlashHead] Getting audio embedding: audio_start_idx={self.audio_start_idx}, audio_end_idx={self.audio_end_idx}")
                audio_embedding = get_audio_embedding(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
                
                # 生成视频
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                video = run_pipeline(self.pipeline, audio_embedding)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                infer_time = end_time - start_time
                infer_time_total += infer_time
                infer_count += 1
                
                # 处理 motion frames（参考 generate_video.py: chunk_idx != 0 时去掉 motion_frames_num）
                if not self.is_first_chunk:
                    video = video[self.motion_frames_num:]
                else:
                    self.is_first_chunk = False
                    logger.info(f"[FlashHead] First chunk processed, motion_frames_num={self.motion_frames_num} kept.")
                
                # video 转换为 numpy 数组 (T, H, W, C)，值范围 [0, 255]
                if isinstance(video, torch.Tensor):
                    video_np = video.cpu().numpy()
                else:
                    video_np = np.array(video)
                
                # 确保值范围在 [0, 255] 并转换为 uint8
                if video_np.max() <= 1.0:
                    video_np = (video_np * 255).clip(0, 255)
                video_np = video_np.astype(np.uint8)
                
                logger.info(f"[FlashHead] Generated video chunk-{self.chunk_idx}: shape={video_np.shape}, "
                            f"dtype={video_np.dtype}, range=[{video_np.min()}, {video_np.max()}], "
                            f"infer_time={infer_time:.3f}s")
                
                # 将生成的帧放入 res_frame_queue
                # 每帧对应正确的音频块
                num_video_frames = video_np.shape[0]
                audio_per_frame = max(1, len(audio_frames) // num_video_frames) if num_video_frames > 0 else 1
                
                for i in range(num_video_frames):
                    frame = video_np[i]
                    # 获取对应的音频帧
                    start_idx = i * audio_per_frame
                    end_idx = min((i + 1) * audio_per_frame, len(audio_frames))
                    frame_audio = audio_frames[start_idx:end_idx] if start_idx < len(audio_frames) else audio_frames[:1]
                    
                    idx = i % len(self.frame_list_cycle) if len(self.frame_list_cycle) > 0 else 0
                    self.res_frame_queue.put((frame, frame_audio, idx))
                
                self.chunk_idx += 1
                
                # 定期报告平均推理速度
                if infer_count >= 100:
                    avg_infer_time = infer_time_total / infer_count
                    logger.info(f"[FlashHead] Average inference time (last {infer_count} chunks): {avg_infer_time:.3f}s")
                    infer_count = 0
                    infer_time_total = 0.0
                    
            except Exception as e:
                logger.error(f"[FlashHead] Inference error: {e}")
                import traceback
                traceback.print_exc()
                continue
                    
        logger.info('[FlashHead] Inference thread stop')
        
    def paste_back_frame(self, pred_frame, idx:int):
        # FlashHead 生成的是完整图像，不需要 paste_back
        return pred_frame
