#!/usr/bin/env python3
"""
FlashHead 模型独立测试脚本

测试内容:
1. Pipeline 初始化测试: 测试 get_pipeline() 是否能正确加载模型
2. 音频编码测试: 测试 get_audio_embedding() 是否能正确编码音频
3. 视频生成测试: 测试 run_pipeline() 是否能正确生成视频
4. 完整流程测试: 测试从音频到视频的完整流程

使用随机数据作为输入，不需要真实音频文件或条件图片。
"""

import os
import sys
import time
import argparse
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import librosa
from PIL import Image
from loguru import logger

# ---------------------------------------------------------------------------
# 路径设置: 将 flashhead_core 加入 Python 路径
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASHHEAD_CORE_PATH = os.path.join(CURRENT_DIR, "avatars", "flashhead_core")
if FLASHHEAD_CORE_PATH not in sys.path:
    sys.path.insert(0, FLASHHEAD_CORE_PATH)
    logger.info(f"[Path] Added to sys.path: {FLASHHEAD_CORE_PATH}")

# ---------------------------------------------------------------------------
# 导入 FlashHead 推理模块
# ---------------------------------------------------------------------------
try:
    from flash_head.inference import (
        get_pipeline,
        get_base_data,
        get_infer_params,
        get_audio_embedding,
        run_pipeline,
    )
    logger.info("[Import] flash_head.inference imported successfully.")
except Exception as e:
    logger.error(f"[Import] Failed to import flash_head.inference: {e}")
    traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
# 配置类
# ---------------------------------------------------------------------------
@dataclass
class TestConfig:
    """测试配置"""
    ckpt_dir: str = "./models/SoulX-FlashHead-1_3B"
    wav2vec_dir: str = "./models/wav2vec2-base-960h"
    model_type: str = "lite"          # "lite", "pro", "pretrained"
    cond_image_path: str = "./data/avatars/wav2lip256_avatar1/full_imgs/0.jpg"
    audio_path: str = "./test_data/Taylor Swift、Shawn Mendes - Lover (Remix).wav"
    base_seed: int = 42
    use_face_crop: bool = False
    world_size: int = 1

    # 音频参数 (来自 infer_params.yaml)
    sample_rate: int = 16000
    tgt_fps: int = 25
    frame_num: int = 33
    motion_frames_latent_num: int = 2
    cached_audio_duration: int = 8

    # 测试控制
    skip_pipeline_init: bool = False
    skip_audio_encode: bool = False
    skip_video_generate: bool = False
    skip_full_pipeline: bool = False


def get_test_config_from_args() -> TestConfig:
    """从命令行参数解析测试配置"""
    parser = argparse.ArgumentParser(description="FlashHead Model Independent Test Script")
    parser.add_argument("--ckpt_dir", type=str, default="./models/SoulX-FlashHead-1_3B",
                        help="FlashHead checkpoint directory")
    parser.add_argument("--wav2vec_dir", type=str, default="./models/wav2vec2-base-960h",
                        help="Wav2vec checkpoint directory")
    parser.add_argument("--model_type", type=str, default="lite", choices=["lite", "pro", "pretrained"],
                        help="Model type: lite / pro / pretrained")
    parser.add_argument("--cond_image_path", type=str,
                        default="./data/avatars/wav2lip256_avatar1/full_imgs/0.jpg",
                        help="Condition image path for prepare_params")
    parser.add_argument("--audio_path", type=str,
                        default="./test_data/Taylor Swift、Shawn Mendes - Lover (Remix).wav",
                        help="Audio file path for testing")
    parser.add_argument("--base_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_face_crop", action="store_true", help="Enable face crop")
    parser.add_argument("--skip_pipeline_init", action="store_true", help="Skip pipeline init test")
    parser.add_argument("--skip_audio_encode", action="store_true", help="Skip audio encode test")
    parser.add_argument("--skip_video_generate", action="store_true", help="Skip video generate test")
    parser.add_argument("--skip_full_pipeline", action="store_true", help="Skip full pipeline test")
    args = parser.parse_args()

    cfg = TestConfig(
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        model_type=args.model_type,
        cond_image_path=args.cond_image_path,
        audio_path=args.audio_path,
        base_seed=args.base_seed,
        use_face_crop=args.use_face_crop,
        skip_pipeline_init=args.skip_pipeline_init,
        skip_audio_encode=args.skip_audio_encode,
        skip_video_generate=args.skip_video_generate,
        skip_full_pipeline=args.skip_full_pipeline,
    )
    return cfg


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def print_tensor_info(name: str, tensor: torch.Tensor, level: str = "INFO") -> None:
    """打印张量的详细信息"""
    msg = (f"[Tensor] {name}: "
           f"shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
           f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
           f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
    if level == "INFO":
        logger.info(msg)
    else:
        logger.debug(msg)


def print_section(title: str) -> None:
    """打印带分隔线的章节标题"""
    line = "=" * 70
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


def print_result(test_name: str, passed: bool, elapsed: float, details: str = "") -> None:
    """打印测试结果"""
    status = "PASSED" if passed else "FAILED"
    icon = "✓" if passed else "✗"
    logger.info(f"[{icon}] {test_name}: {status} ({elapsed:.3f}s) {details}")


def load_audio_file(
    audio_path: str,
    sample_rate: int = 16000,
    duration_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    加载真实音频文件

    Args:
        audio_path: 音频文件路径
        sample_rate: 目标采样率 (Hz)
        duration_seconds: 加载时长 (秒), None 表示加载全部

    Returns:
        np.ndarray: 音频波形数组, shape=(num_samples,), dtype=float32
    """
    if not os.path.exists(audio_path):
        logger.warning(f"[Audio] File not found: {audio_path}, falling back to random audio")
        return generate_random_audio(sample_rate, duration_seconds or 8.0)
    
    try:
        # 使用 librosa 加载音频
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration_seconds)
        
        logger.info(f"[Audio] Loaded from file: {audio_path}")
        logger.info(f"[Audio] Original sample rate: {sr}, shape: {audio.shape}, "
                    f"duration={len(audio)/sr:.2f}s, range=[{audio.min():.4f}, {audio.max():.4f}]")
        
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"[Audio] Failed to load {audio_path}: {e}")
        logger.info("[Audio] Falling back to random audio")
        return generate_random_audio(sample_rate, duration_seconds or 8.0)


def generate_random_audio(
    sample_rate: int = 16000,
    duration_seconds: float = 8.0,
    seed: int = 42,
) -> np.ndarray:
    """
    生成随机音频数据 (模拟真实音频波形)

    Args:
        sample_rate: 采样率 (Hz)
        duration_seconds: 音频时长 (秒)
        seed: 随机种子

    Returns:
        np.ndarray: 音频波形数组, shape=(num_samples,), dtype=float32
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sample_rate * duration_seconds)

    # 组合多种频率成分, 模拟语音-like 信号
    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)

    # 基频 (类似人声)
    fundamental = 150.0  # Hz
    audio = np.sin(2 * np.pi * fundamental * t).astype(np.float32)

    # 添加谐波
    for harmonic in [2, 3, 4]:
        amplitude = 0.3 / harmonic
        audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t).astype(np.float32)

    # 添加低频调制 (模拟音调变化)
    modulator = 0.2 * np.sin(2 * np.pi * 3.0 * t).astype(np.float32)
    audio = audio * (1.0 + modulator)

    # 添加高斯噪声
    noise = rng.normal(0, 0.05, num_samples).astype(np.float32)
    audio = audio + noise

    # 归一化到 [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    logger.info(f"[Audio] Generated random audio: shape={audio.shape}, "
                f"duration={duration_seconds:.2f}s, sample_rate={sample_rate}Hz, "
                f"range=[{audio.min():.4f}, {audio.max():.4f}]")
    return audio


def load_image_file(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    加载真实图片文件

    Args:
        image_path: 图片文件路径
        target_size: 目标尺寸 (width, height)

    Returns:
        np.ndarray: RGB 图片数组, shape=(H, W, 3), dtype=uint8
    """
    if not os.path.exists(image_path):
        logger.warning(f"[Image] File not found: {image_path}, falling back to random image")
        return generate_random_cond_image(target_size[1], target_size[0])
    
    try:
        # 使用 PIL 加载图片
        image_pil = Image.open(image_path).convert("RGB")
        
        # 调整尺寸
        if image_pil.size != target_size:
            image_pil = image_pil.resize(target_size, Image.LANCZOS)
            logger.info(f"[Image] Resized from {image_pil.size} to {target_size}")
        
        image_np = np.array(image_pil)
        
        logger.info(f"[Image] Loaded from file: {image_path}")
        logger.info(f"[Image] Shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        return image_np
    except Exception as e:
        logger.error(f"[Image] Failed to load {image_path}: {e}")
        logger.info("[Image] Falling back to random image")
        return generate_random_cond_image(target_size[1], target_size[0])


def generate_random_cond_image(
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    生成随机条件图片 (RGB)

    Args:
        height: 图片高度
        width: 图片宽度
        seed: 随机种子

    Returns:
        np.ndarray: RGB 图片数组, shape=(H, W, 3), dtype=uint8
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    logger.info(f"[Image] Generated random condition image: shape={image.shape}, dtype={image.dtype}")
    return image


def save_random_image(path: str, height: int = 512, width: int = 512, seed: int = 42) -> str:
    """
    生成并保存一张随机图片, 返回保存路径

    Args:
        path: 保存路径
        height: 图片高度
        width: 图片宽度
        seed: 随机种子

    Returns:
        str: 保存后的图片路径
    """
    image_np = generate_random_cond_image(height, width, seed)
    image_pil = Image.fromarray(image_np, mode="RGB")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image_pil.save(path)
    logger.info(f"[Image] Saved random condition image to: {path}")
    return path


# ---------------------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------------------
def test_pipeline_init(cfg: TestConfig) -> Tuple[bool, Optional[object], float]:
    """
    测试 Pipeline 初始化

    Args:
        cfg: 测试配置

    Returns:
        (passed, pipeline_instance, elapsed_time)
    """
    print_section("TEST 1: Pipeline Initialization")
    logger.info(f"[Config] ckpt_dir={cfg.ckpt_dir}")
    logger.info(f"[Config] wav2vec_dir={cfg.wav2vec_dir}")
    logger.info(f"[Config] model_type={cfg.model_type}")
    logger.info(f"[Config] world_size={cfg.world_size}")

    # 检查路径存在性 (仅做日志提示, 不强制要求存在, 因为测试可能用 mock)
    if not os.path.exists(cfg.ckpt_dir):
        logger.warning(f"[Check] ckpt_dir does not exist: {cfg.ckpt_dir}")
    if not os.path.exists(cfg.wav2vec_dir):
        logger.warning(f"[Check] wav2vec_dir does not exist: {cfg.wav2vec_dir}")

    pipeline = None
    start_time = time.perf_counter()
    try:
        pipeline = get_pipeline(
            world_size=cfg.world_size,
            ckpt_dir=cfg.ckpt_dir,
            model_type=cfg.model_type,
            wav2vec_dir=cfg.wav2vec_dir,
        )
        elapsed = time.perf_counter() - start_time

        # 验证 pipeline 对象
        assert pipeline is not None, "Pipeline is None"
        assert hasattr(pipeline, "model"), "Pipeline missing 'model' attribute"
        assert hasattr(pipeline, "vae"), "Pipeline missing 'vae' attribute"
        assert hasattr(pipeline, "audio_encoder"), "Pipeline missing 'audio_encoder' attribute"
        assert hasattr(pipeline, "device"), "Pipeline missing 'device' attribute"
        assert hasattr(pipeline, "config"), "Pipeline missing 'config' attribute"

        logger.info(f"[Pipeline] Device: {pipeline.device}")
        logger.info(f"[Pipeline] Model type: {pipeline.model_type}")
        logger.info(f"[Pipeline] Use LTX: {pipeline.use_ltx}")
        logger.info(f"[Pipeline] Config: {pipeline.config}")

        # 验证模型参数
        model_params = sum(p.numel() for p in pipeline.model.parameters())
        logger.info(f"[Pipeline] Model parameters: {model_params:,}")

        print_result("Pipeline Initialization", True, elapsed)
        return True, pipeline, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"[Pipeline Init] Error: {e}")
        traceback.print_exc()
        print_result("Pipeline Initialization", False, elapsed, str(e))
        return False, None, elapsed


def test_audio_encoding(
    cfg: TestConfig,
    pipeline: object,
) -> Tuple[bool, Optional[torch.Tensor], float]:
    """
    测试音频编码

    Args:
        cfg: 测试配置
        pipeline: 已初始化的 pipeline 实例

    Returns:
        (passed, audio_embedding, elapsed_time)
    """
    print_section("TEST 2: Audio Encoding")

    if pipeline is None:
        logger.error("[Audio Encode] Pipeline is None, skipping test.")
        return False, None, 0.0

    # 加载真实音频文件（如果不存在则回退到随机音频）
    audio_array = load_audio_file(
        audio_path=cfg.audio_path,
        sample_rate=cfg.sample_rate,
        duration_seconds=cfg.cached_audio_duration,
    )

    start_time = time.perf_counter()
    try:
        audio_embedding = get_audio_embedding(pipeline, audio_array)
        elapsed = time.perf_counter() - start_time

        # 验证输出
        assert audio_embedding is not None, "Audio embedding is None"
        assert isinstance(audio_embedding, torch.Tensor), f"Expected Tensor, got {type(audio_embedding)}"
        
        print_tensor_info("audio_embedding", audio_embedding)
        
        # 记录实际维度，支持 4D 或 5D
        logger.info(f"[Audio Embed] Tensor dimension: {audio_embedding.dim()}D, shape: {list(audio_embedding.shape)}")
        
        # 验证形状（支持 4D [1,T,5,D] 或 5D [1,T,5,1,D]）
        if audio_embedding.dim() == 4:
            logger.info(f"[Audio Embed] Shape details: batch={audio_embedding.shape[0]}, "
                        f"frames={audio_embedding.shape[1]}, "
                        f"context_len={audio_embedding.shape[2]}, "
                        f"hidden_dim={audio_embedding.shape[3]}")
        elif audio_embedding.dim() == 5:
            logger.info(f"[Audio Embed] Shape details: batch={audio_embedding.shape[0]}, "
                        f"frames={audio_embedding.shape[1]}, "
                        f"context_len={audio_embedding.shape[2]}, "
                        f"extra_dim={audio_embedding.shape[3]}, "
                        f"hidden_dim={audio_embedding.shape[4]}")
        else:
            logger.warning(f"[Audio Embed] Unexpected dimension: {audio_embedding.dim()}D")

        print_result("Audio Encoding", True, elapsed)
        return True, audio_embedding, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"[Audio Encode] Error: {e}")
        traceback.print_exc()
        print_result("Audio Encoding", False, elapsed, str(e))
        return False, None, elapsed


def test_video_generation(
    cfg: TestConfig,
    pipeline: object,
    audio_embedding: torch.Tensor,
) -> Tuple[bool, Optional[torch.Tensor], float]:
    """
    测试视频生成

    Args:
        cfg: 测试配置
        pipeline: 已初始化的 pipeline 实例
        audio_embedding: 音频嵌入张量

    Returns:
        (passed, video_frames, elapsed_time)
    """
    print_section("TEST 3: Video Generation")

    if pipeline is None:
        logger.error("[Video Gen] Pipeline is None, skipping test.")
        return False, None, 0.0
    if audio_embedding is None:
        logger.error("[Video Gen] Audio embedding is None, skipping test.")
        return False, None, 0.0

    # 检查是否需要调用 prepare_params
    if not hasattr(pipeline, 'frame_num') or pipeline.frame_num is None:
        logger.info("[Video Gen] Pipeline not prepared, calling get_base_data...")
        from flash_head.inference import get_base_data, get_infer_params
        
        # 加载条件图片
        cond_image_path = cfg.cond_image_path
        if not os.path.exists(cond_image_path):
            default_image = "./test_data/汪东城.jpg"
            if os.path.exists(default_image):
                cond_image_path = default_image
            else:
                cond_image_path = os.path.join(CURRENT_DIR, "test_data", "random_cond_image.jpg")
                save_random_image(cond_image_path, height=512, width=512, seed=cfg.base_seed)
        
        get_base_data(
            pipeline,
            cond_image_path_or_dir=cond_image_path,
            base_seed=cfg.base_seed,
            use_face_crop=cfg.use_face_crop,
        )
        logger.info(f"[Video Gen] Pipeline prepared: frame_num={pipeline.frame_num}, motion_frames_num={pipeline.motion_frames_num}")

    start_time = time.perf_counter()
    try:
        video_frames = run_pipeline(pipeline, audio_embedding)
        elapsed = time.perf_counter() - start_time

        # 验证输出
        assert video_frames is not None, "Video frames is None"
        assert isinstance(video_frames, torch.Tensor), f"Expected Tensor, got {type(video_frames)}"
        assert video_frames.dim() == 4, f"Expected 4D tensor, got {video_frames.dim()}D"

        print_tensor_info("video_frames", video_frames)

        # 验证形状: [T, H, W, C]
        logger.info(f"[Video] Shape details: T={video_frames.shape[0]}, "
                    f"H={video_frames.shape[1]}, W={video_frames.shape[2]}, C={video_frames.shape[3]}")

        # 验证数值范围 (应在 [0, 255] 之间)
        min_val = video_frames.min().item()
        max_val = video_frames.max().item()
        assert 0 <= min_val <= 255, f"Min value out of range: {min_val}"
        assert 0 <= max_val <= 255, f"Max value out of range: {max_val}"
        logger.info(f"[Video] Value range: [{min_val:.2f}, {max_val:.2f}]")

        print_result("Video Generation", True, elapsed)
        return True, video_frames, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"[Video Gen] Error: {e}")
        traceback.print_exc()
        print_result("Video Generation", False, elapsed, str(e))
        return False, None, elapsed


def test_full_pipeline(cfg: TestConfig) -> Tuple[bool, float]:
    """
    测试完整流程: 从音频到视频

    Args:
        cfg: 测试配置

    Returns:
        (passed, elapsed_time)
    """
    print_section("TEST 4: Full Pipeline (Audio -> Video)")

    total_start = time.perf_counter()
    pipeline = None
    audio_embedding = None
    video_frames = None

    try:
        # Step 1: 初始化 Pipeline
        logger.info("[Full Pipeline] Step 1/4: Initializing pipeline...")
        step_start = time.perf_counter()
        pipeline = get_pipeline(
            world_size=cfg.world_size,
            ckpt_dir=cfg.ckpt_dir,
            model_type=cfg.model_type,
            wav2vec_dir=cfg.wav2vec_dir,
        )
        step_elapsed = time.perf_counter() - step_start
        logger.info(f"[Full Pipeline] Step 1 done in {step_elapsed:.3f}s")

        # Step 2: 准备基础数据 (条件图片)
        logger.info("[Full Pipeline] Step 2/4: Preparing base data...")
        step_start = time.perf_counter()

        # 加载条件图片（真实图片或随机图片）
        cond_image_path = cfg.cond_image_path
        if not os.path.exists(cond_image_path):
            logger.warning(f"[Full Pipeline] Condition image not found: {cond_image_path}")
            # 尝试使用默认测试图片
            default_image = "./test_data/汪东城.jpg"
            if os.path.exists(default_image):
                cond_image_path = default_image
                logger.info(f"[Full Pipeline] Using default test image: {cond_image_path}")
            else:
                cond_image_path = os.path.join(CURRENT_DIR, "test_data", "random_cond_image.jpg")
                save_random_image(cond_image_path, height=512, width=512, seed=cfg.base_seed)

        get_base_data(
            pipeline,
            cond_image_path_or_dir=cond_image_path,
            base_seed=cfg.base_seed,
            use_face_crop=cfg.use_face_crop,
        )
        step_elapsed = time.perf_counter() - step_start
        logger.info(f"[Full Pipeline] Step 2 done in {step_elapsed:.3f}s")

        # Step 3: 音频编码
        logger.info("[Full Pipeline] Step 3/4: Encoding audio...")
        step_start = time.perf_counter()
        audio_array = load_audio_file(
            audio_path=cfg.audio_path,
            sample_rate=cfg.sample_rate,
            duration_seconds=cfg.cached_audio_duration,
        )
        audio_embedding = get_audio_embedding(pipeline, audio_array)
        step_elapsed = time.perf_counter() - step_start
        logger.info(f"[Full Pipeline] Step 3 done in {step_elapsed:.3f}s")
        print_tensor_info("audio_embedding (full)", audio_embedding)

        # Step 4: 视频生成
        logger.info("[Full Pipeline] Step 4/4: Generating video...")
        step_start = time.perf_counter()
        video_frames = run_pipeline(pipeline, audio_embedding)
        step_elapsed = time.perf_counter() - step_start
        logger.info(f"[Full Pipeline] Step 4 done in {step_elapsed:.3f}s")
        print_tensor_info("video_frames (full)", video_frames)

        total_elapsed = time.perf_counter() - total_start

        # 验证结果
        assert video_frames is not None, "Full pipeline: video_frames is None"
        assert video_frames.dim() == 4, f"Full pipeline: expected 4D video, got {video_frames.dim()}D"
        logger.info(f"[Full Pipeline] Generated video: shape={list(video_frames.shape)}, "
                    f"dtype={video_frames.dtype}")

        print_result("Full Pipeline", True, total_elapsed)
        return True, total_elapsed

    except Exception as e:
        total_elapsed = time.perf_counter() - total_start
        logger.error(f"[Full Pipeline] Error: {e}")
        traceback.print_exc()
        print_result("Full Pipeline", False, total_elapsed, str(e))
        return False, total_elapsed


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    """主入口"""
    print_section("FlashHead Model Independent Test")
    logger.info(f"[System] Python: {sys.version}")
    logger.info(f"[System] PyTorch: {torch.__version__}")
    logger.info(f"[System] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"[System] CUDA version: {torch.version.cuda}")
        logger.info(f"[System] GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"[System] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    cfg = get_test_config_from_args()
    logger.info(f"[Config] {cfg}")

    results = {}
    pipeline = None
    audio_embedding = None
    video_frames = None

    # -----------------------------------------------------------------------
    # Test 1: Pipeline 初始化
    # -----------------------------------------------------------------------
    if not cfg.skip_pipeline_init:
        passed, pipeline, elapsed = test_pipeline_init(cfg)
        results["pipeline_init"] = {"passed": passed, "elapsed": elapsed}
    else:
        logger.info("[Skip] Pipeline initialization test skipped.")
        results["pipeline_init"] = {"passed": None, "elapsed": 0.0}

    # -----------------------------------------------------------------------
    # Test 2: 音频编码
    # -----------------------------------------------------------------------
    if not cfg.skip_audio_encode:
        passed, audio_embedding, elapsed = test_audio_encoding(cfg, pipeline)
        results["audio_encode"] = {"passed": passed, "elapsed": elapsed}
    else:
        logger.info("[Skip] Audio encoding test skipped.")
        results["audio_encode"] = {"passed": None, "elapsed": 0.0}

    # -----------------------------------------------------------------------
    # Test 3: 视频生成
    # -----------------------------------------------------------------------
    if not cfg.skip_video_generate:
        passed, video_frames, elapsed = test_video_generation(cfg, pipeline, audio_embedding)
        results["video_generate"] = {"passed": passed, "elapsed": elapsed}
    else:
        logger.info("[Skip] Video generation test skipped.")
        results["video_generate"] = {"passed": None, "elapsed": 0.0}

    # -----------------------------------------------------------------------
    # Test 4: 完整流程
    # -----------------------------------------------------------------------
    if not cfg.skip_full_pipeline:
        passed, elapsed = test_full_pipeline(cfg)
        results["full_pipeline"] = {"passed": passed, "elapsed": elapsed}
    else:
        logger.info("[Skip] Full pipeline test skipped.")
        results["full_pipeline"] = {"passed": None, "elapsed": 0.0}

    # -----------------------------------------------------------------------
    # 测试报告
    # -----------------------------------------------------------------------
    print_section("Test Summary Report")

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    total_time = 0.0

    for test_name, result in results.items():
        status = result["passed"]
        elapsed = result["elapsed"]
        total_time += elapsed

        if status is True:
            passed_tests += 1
            total_tests += 1
            logger.info(f"  [{test_name}] PASSED ({elapsed:.3f}s)")
        elif status is False:
            failed_tests += 1
            total_tests += 1
            logger.info(f"  [{test_name}] FAILED ({elapsed:.3f}s)")
        else:
            skipped_tests += 1
            logger.info(f"  [{test_name}] SKIPPED")

    logger.info("-" * 50)
    logger.info(f"Total tests:  {total_tests}")
    logger.info(f"Passed:       {passed_tests}")
    logger.info(f"Failed:       {failed_tests}")
    logger.info(f"Skipped:      {skipped_tests}")
    logger.info(f"Total time:   {total_time:.3f}s")
    logger.info("-" * 50)

    if failed_tests > 0:
        logger.error("Some tests FAILED. Please check the logs above.")
        sys.exit(1)
    elif passed_tests > 0:
        logger.info("All executed tests PASSED!")
        sys.exit(0)
    else:
        logger.warning("All tests were skipped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
