# TODO: LiveTalking SoulX Extension 待办事宜与操作指引

## 1. 待办事项 (需在 Linux 服务器部署时人工执行)

- [ ] **准备 Conda 运行环境**:
  - 对于 **FlashHead**：需要创建 `flashhead` 环境，安装 `torch==2.7.1`、`flash_attn`（建议 >=2.8.0）、`sageattention` 等。
  - 对于 **LiveAct**：需要创建 `liveact` 环境，安装包含 `vllm`、修改版 `SageAttention` 以及 `LightX2V` 等。
- [ ] **准备模型权重**:
  - 下载 FlashHead 权重（`SoulX-FlashHead-1_3B`）和 Wav2vec2 权重（`wav2vec2-base-960h`）至服务器 `LiveTalking/models/`。
  - 下载 LiveAct 权重和中文 Wav2vec2（`chinese-wav2vec2-base`）至服务器 `LiveTalking/models/`。
- [ ] **环境验证与测试**:
  - 在启动 `LiveTalking` 后端时检查是否存在由于 `PYTHONPATH` 或依赖缺失导致的 `ModuleNotFoundError`，如有，需通过 `pip install` 补齐对应依赖。

## 2. 操作指引

1. **运行 FlashHead 模型后端**:
```bash
conda activate flashhead
python app.py --transport webrtc --model flashhead --flashhead_ckpt ./models/SoulX-FlashHead-1_3B --wav2vec_dir ./models/wav2vec2-base-960h --avatar_id <YOUR_AVATAR_FOLDER>
```

2. **运行 LiveAct 模型后端**:
```bash
conda activate liveact
python app.py --transport webrtc --model liveact --liveact_ckpt ./models/SoulX-LiveAct --wav2vec_dir ./models/chinese-wav2vec2-base --avatar_id <YOUR_AVATAR_FOLDER>
```

3. **前端 Windows 交互**:
通过浏览器访问 `http://<服务器IP>:8010/dashboard.html`，即可开始使用数字人进行交互。
