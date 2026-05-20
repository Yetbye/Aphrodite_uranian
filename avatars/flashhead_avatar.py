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
from utils.logger import logger
from utils.image import read_imgs
from utils.device import initialize_device
from registry import register

device = initialize_device()

def load_model(opt):
    logger.info("Loading FlashHead Model...")
    world_size = 1
    # 强制将一些环境变量或参数传给 pipeline
    # 注意: flashhead 区分 pro / lite, 默认可以用 lite 以保证实时性
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
        
        # 提取推理参数
        self.infer_params = get_infer_params()
        self.sample_rate = self.infer_params['sample_rate']
        self.tgt_fps = self.infer_params['tgt_fps']
        self.frame_num = self.infer_params['frame_num']
        self.motion_frames_num = self.infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        # 初始化条件图
        get_base_data(self.pipeline, cond_image_path_or_dir=self.cond_image_path, base_seed=42, use_face_crop=False)
        
        # 为了兼容流式，FlashHead需要攒够一定长度的音频
        # 一次生成 slice_len 帧视频对应的音频长度
        self.audio_buffer = []
        self.audio_chunk_size = self.slice_len * self.sample_rate // self.tgt_fps
        
        # 维护一个连续的audio queue，用于计算audio_embedding
        cached_audio_duration = self.infer_params['cached_audio_duration']
        self.cached_audio_length_sum = self.sample_rate * cached_audio_duration
        self.audio_end_idx = cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        
        from collections import deque
        self.audio_dq = deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        self.is_first_chunk = True

    def inference(self, quit_event):
        logger.info('FlashHead start inference thread')
        while not quit_event.is_set():
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            # BaseAvatar中的asr默认输出的audio_frames是20ms块
            # 我们需要把它们收集起来转成numpy array
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
            self.audio_buffer.extend(pcm_data.tolist())
            
            # 当缓冲够了一个 block 的要求时
            while len(self.audio_buffer) >= self.audio_chunk_size:
                human_speech_array = np.array(self.audio_buffer[:self.audio_chunk_size])
                self.audio_buffer = self.audio_buffer[self.audio_chunk_size:]
                
                # Streaming encode
                self.audio_dq.extend(human_speech_array.tolist())
                audio_array = np.array(self.audio_dq)
                
                audio_embedding = get_audio_embedding(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
                
                video = run_pipeline(self.pipeline, audio_embedding)
                
                if not self.is_first_chunk:
                    video = video[self.motion_frames_num:]
                else:
                    self.is_first_chunk = False
                    
                video = video.cpu().numpy().astype(np.uint8)
                
                # 将生成的帧放入res_frame_queue
                # FlashHead生成的是一连串帧，我们需要把它一帧一帧地压入BaseAvatar要求的队列格式中
                for i in range(video.shape[0]):
                    frame = video[i]
                    # 我们需要配套给它一个假的audio_frames以便后续处理
                    # 为了简化，直接传递None或第一个audioframe的元数据
                    idx = i % len(self.frame_list_cycle) if len(self.frame_list_cycle) > 0 else 0
                    
                    # 取对应长度的原始音频给回推流
                    # 这里是简化的处理，实际可能需要更精确的音视频时间戳对齐
                    self.res_frame_queue.put((frame, audio_frames[:2], idx))
                    
        logger.info('FlashHead inference thread stop')
        
    def paste_back_frame(self, pred_frame, idx:int):
        # FlashHead 生成的是整个半身或头部，如果是全画幅可能不需要 paste_back，这里直接返回
        # 如果需要做人脸替换，需要引入额外的 crop & paste 逻辑
        return pred_frame
