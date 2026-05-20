# ALIGNMENT: LiveTalking SoulX Extension

## 1. 项目和任务特性规范
**项目名称**: LiveTalking SoulX Extension
**任务目标**: 将 `LiveTalking` 项目进行扩展，使其支持接入并使用 `SoulX-FlashHead` 和 `SoulX-LiveAct` 这两个最新的实时数字人/人体动画模型，实现前后端分离的运行模式（Linux服务器跑模型推理，Windows本地通过浏览器/客户端交互）。

## 2. 原始需求
- 将 `d:\Research\VENUS\LiveTalking` 拓展至可以使用这两个模型 `d:\Research\VENUS\SoulX-FlashHead` 和 `d:\Research\VENUS\SoulX-LiveAct`。
- 保持类似 `SoulX-FlashHead-WEB` 的前后端分离架构，即模型在学校Linux系统上运行，用户在自己的Windows电脑上使用。

## 3. 需求理解与项目分析
### 3.1 现有架构分析
- **LiveTalking**: 
  - 是一个基于 WebRTC/RTMP 的实时流式数字人系统，采用高度模块化的架构（通过 `registry.py` 注册机制管理插件）。
  - 核心工作流：前端接收文本/音频 -> 触发大模型(LLM) -> 文本转语音(TTS) -> 音频特征提取(ASR/Whisper/Hubert) -> Avatar 推理模型生成视频帧 -> 通过 WebRTC 等协议推送给前端。
  - 目前的 Avatar 插件存放在 `LiveTalking/avatars/` 目录下（如 `musetalk_avatar.py`, `wav2lip_avatar.py`）。
- **SoulX-FlashHead**:
  - 基于 LTX-Video VAE 和 FlashHead 模型构建的高效实时 Talking Head 生成系统。
  - 推理核心在 `flash_head.inference.run_pipeline`，支持分块（chunk）级别的流式音频输入（`audio_encode_mode='stream'`）。
- **SoulX-LiveAct**:
  - 基于 Wan2.1 和 VACE 架构，支持小时级的高保真人像动画生成。
  - 推理核心在 `generate.py` 中，采用 AR diffusion 和 ConvKV Memory 机制实现流式长视频生成。

### 3.2 边界确认 (任务范围)
- 需要在 `LiveTalking` 中新增两个 Avatar 插件：`flash_head_avatar.py` 和 `liveact_avatar.py`。
- 修改 `LiveTalking` 的配置解析（`config.py`）和入口（`app.py`），使启动时可以通过 `--model flashhead` 或 `--model liveact` 挂载对应的模型。
- 由于 `SoulX` 模型依赖非常特定的库（如 `flash_attn`, `sageattention`, `vllm`, `lightx2v` 等），需要在执行时确保环境正确配置，或者编写明确的运行指南。

## 4. 疑问澄清与智能决策
在分析过程中，我发现了几个需要确认的关键决策点：

1. **代码合并策略**:
   - `SoulX-FlashHead` 和 `SoulX-LiveAct` 有自己庞大的代码库和特定的模型权重路径。是将这两个库的 Python 模块作为 package 直接引入 `LiveTalking`（例如通过设置 `sys.path` 或将其复制到 `LiveTalking` 目录），还是在各自的目录下运行修改版的 `LiveTalking` 后端？
   - *推荐决策*：保持 `LiveTalking` 代码库整洁，将 `SoulX` 的项目路径加入 `PYTHONPATH`，在 `LiveTalking` 新增的 avatar 插件中直接 `import flash_head` 或 `import wan`，这样便于维护。

2. **环境依赖冲突问题**:
   - `LiveTalking` 原始环境与 `SoulX-FlashHead`、`SoulX-LiveAct` 的依赖差异较大（特别是 Torch 版本、CUDA 版本、FlashAttention 等加速算子）。
   - *推荐决策*：建议为 `FlashHead` 和 `LiveAct` 分别创建独立的 Conda 环境运行后端的 `app.py`，因为它们底层的加速算子（如 SageAttention, FP8 GEMM）环境配置非常苛刻，无法轻易合并为一个单一的虚拟环境。

3. **流式推理适配**:
   - `LiveTalking` 的 `BaseAvatar` 假设音频按照 20ms 的 chunk 传入，并返回单帧或小批量的帧。
   - `SoulX` 模型是基于特定 `blksz_lst` (比如 28 帧或 24 帧) 的扩散模型生成，因此在插件中需要实现音频特征的缓冲池（Buffer），等攒够一个 block 所需的音频长度后，再调用模型进行一次前向传播，随后将生成的视频帧按顺序分发给 `LiveTalking` 的 `res_frame_queue`。
