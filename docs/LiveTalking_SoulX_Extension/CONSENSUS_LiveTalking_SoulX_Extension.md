# CONSENSUS: LiveTalking SoulX Extension

## 1. 明确的需求描述和验收标准
**需求描述**:
在现有的 `LiveTalking` 项目中集成 `SoulX-FlashHead` 和 `SoulX-LiveAct` 两个高保真数字人模型，使得用户可以在启动 `LiveTalking` 时通过命令行参数指定使用这两个模型之一。

**验收标准**:
1. **代码结构**: 采用“复制代码融合”的策略，将 `SoulX-FlashHead` 和 `SoulX-LiveAct` 的核心模型代码复制并集成到 `LiveTalking/avatars/` 相应的子目录下。
2. **插件化接入**: 实现 `avatars/flashhead_avatar.py` 和 `avatars/liveact_avatar.py`，继承 `BaseAvatar`，并在 `registry.py` 和 `app.py` 中注册这两个新模型。
3. **流式推理适配**: 在新编写的 Avatar 插件中，实现将 LiveTalking 传来的微小音频块（Chunk）缓冲为扩散模型所需的 Block，然后调用各自的 pipeline 生成视频帧，再逐帧推入输出队列（`res_frame_queue`）。
4. **环境管理与执行约束**: 不在用户的本地 Windows 电脑上执行环境安装和依赖配置操作。代码将假定在最终的 Linux 服务器部署时，由用户根据模型分别激活其所需的特定 Conda 环境（包含 `flash_attn`, `sageattention` 等）。

## 2. 技术实现方案
- **模型代码迁移**:
  - 将 `SoulX-FlashHead/flash_head` 拷贝至 `LiveTalking/avatars/flashhead_core`。
  - 将 `SoulX-LiveAct` 相关的核心文件（`wan`, `model_liveact`, `src`, `fp8_gemm.py`, `util_liveact.py` 等）拷贝至 `LiveTalking/avatars/liveact_core`。
- **LiveTalking 核心修改**:
  - **`config.py`**: 在命令行解析中为 `--model` 增加 `flashhead` 和 `liveact` 选项，并添加两者所需的特有配置项（如 `--flashhead_ckpt`, `--liveact_ckpt` 等）。
  - **`app.py`**: 在 `_avatar_modules` 字典中注册这两个新模块。
  - **`avatars/flashhead_avatar.py`**: 实现 `load_model`, `load_avatar`, `warm_up`, 以及类 `FlashHeadAvatar(BaseAvatar)` 的流式 `inference_batch` 和 `render` 逻辑适配。
  - **`avatars/liveact_avatar.py`**: 实现 `load_model`, `load_avatar`, `warm_up`, 以及类 `LiveActAvatar(BaseAvatar)` 的流式 AR diffusion 推理逻辑适配。

## 3. 任务边界限制
- 本任务只涉及核心代码层面的融合和接口适配（Python 级别），不涉及底层模型结构的重新训练或权重修改。
- 本地 Windows 仅用于代码编写和逻辑校验，不进行全流程的真实 GPU 推理运行（因为缺少双 4090/5090 以及特定 Linux 依赖）。

## 4. 所有不确定性已解决
- **集成方式**: 确认使用代码直接复制合并的方式。
- **运行环境**: 确认后端实际运行将在目标 Linux 环境下独立配置进行，本任务仅提供兼容的代码实现，不做任何系统环境破坏或修改。
