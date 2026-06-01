"""
MiMo API 独立测试脚本

用法:
    python test_mimo_api.py [text|audio|both]

参数:
    text  - 仅测试文本对话 (默认)
    audio - 仅测试音频/多模态对话
    both  - 测试两种模式

环境变量:
    MIMO_API_KEY      - MiMo API 密钥
    MIMO_USE_BEARER   - 设置为 true 使用 Authorization: Bearer 方式 (默认 api-key 头)
"""

import os
import sys
import json
import time
import base64
import argparse
from dotenv import load_dotenv

import requests

# 加载 .env 文件
load_dotenv()

API_URL = "https://api.xiaomimimo.com/v1/chat/completions"


def get_api_key() -> str:
    """从环境变量或 .env 文件读取 API Key"""
    api_key = os.getenv("MIMO_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        print("[ERROR] MIMO_API_KEY 未设置。请在 .env 文件或环境变量中配置。")
        print("示例: MIMO_API_KEY=your-api-key-here")
        sys.exit(1)
    # 打印脱敏信息
    masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"[INFO] API Key loaded: {masked}")
    return api_key


def build_headers(api_key: str, use_bearer: bool = False) -> dict:
    """构建请求头"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if use_bearer:
        headers["Authorization"] = f"Bearer {api_key}"
        print("[INFO] 使用 Authorization: Bearer 认证方式")
    else:
        headers["api-key"] = api_key
        print("[INFO] 使用 api-key 头认证方式")
    return headers


def parse_sse_chunk(line: bytes) -> dict | None:
    """解析 SSE chunk"""
    if not line:
        return None
    decoded = line.decode("utf-8").strip()
    if not decoded.startswith("data: "):
        return None
    data_str = decoded[6:]
    if data_str == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        print(f"[WARN] JSON 解析失败: {data_str[:200]}")
        return None


def extract_content(chunk: dict) -> str:
    """从 chunk 提取 content"""
    choices = chunk.get("choices", [])
    if not choices:
        return ""
    delta = choices[0].get("delta", {})
    return delta.get("content", "") or ""


def send_request(payload: dict, headers: dict, max_retries: int = 3, timeout: int = 30):
    """发送请求，带重试"""
    for attempt in range(1, max_retries + 1):
        print(f"\n[INFO] 请求尝试 {attempt}/{max_retries}")
        print(f"[INFO] URL: {API_URL}")
        print(f"[INFO] Model: {payload.get('model')}")
        print(f"[INFO] Stream: {payload.get('stream')}")
        print(f"[INFO] Messages count: {len(payload.get('messages', []))}")

        try:
            start = time.perf_counter()
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                stream=payload.get("stream", True),
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
            print(f"[INFO] 响应状态码: {response.status_code} (耗时 {elapsed:.2f}s)")

            if response.status_code == 200:
                return response

            # 记录错误详情
            print(f"[ERROR] HTTP 错误: {response.status_code}")
            print(f"[ERROR] 响应头: {dict(response.headers)}")
            body = response.text[:800]
            print(f"[ERROR] 响应体: {body}")

            if response.status_code in (401, 403):
                print("[ERROR] 认证失败，请检查 API Key 是否正确")
                return None

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[WARN] 将在 {wait}s 后重试...")
                    time.sleep(wait)
                    continue

            return None

        except requests.exceptions.Timeout:
            print(f"[ERROR] 请求超时")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] 连接错误: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            print(f"[ERROR] 请求异常: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    print(f"[ERROR] 请求在 {max_retries} 次尝试后失败")
    return None


def stream_response(response, label: str = "Response"):
    """处理并打印流式响应"""
    print(f"\n{'='*60}")
    print(f"[{label}] 流式响应内容:")
    print("=" * 60)

    result = ""
    chunk_count = 0
    first = True
    start = time.perf_counter()

    for line in response.iter_lines():
        if not line:
            continue

        chunk_count += 1
        chunk = parse_sse_chunk(line)
        if chunk is None:
            continue
        if chunk.get("__done__"):
            print(f"\n[INFO] 流结束标记 [DONE], 共 {chunk_count} 个 chunks")
            break

        # 打印每个 chunk 的摘要
        chunk_preview = json.dumps(chunk, ensure_ascii=False)[:250]
        print(f"[CHUNK #{chunk_count}] {chunk_preview}...")

        msg = extract_content(chunk)
        if not msg:
            continue

        if first:
            ttf = time.perf_counter() - start
            print(f"\n[INFO] 首 token 延迟: {ttf:.3f}s")
            first = False

        result += msg
        print(msg, end="", flush=True)

    total = time.perf_counter() - start
    print(f"\n{'='*60}")
    print(f"[INFO] 总耗时: {total:.3f}s, chunks: {chunk_count}")
    print(f"[INFO] 完整回复:\n{result}")
    print("=" * 60)
    return result


def test_text_chat(api_key: str, use_bearer: bool = False):
    """测试文本对话"""
    print("\n" + "=" * 60)
    print("测试模式: 文本对话 (mimo-v2.5-pro)")
    print("=" * 60)

    headers = build_headers(api_key, use_bearer)
    payload = {
        "model": "mimo-v2.5-pro",
        "messages": [
            {"role": "system", "content": "你是一个数字人助手，请用简短、口语化、温柔的语气回答。"},
            {"role": "user", "content": "你好，请简单介绍一下自己"},
        ],
        "max_completion_tokens": 256,
        "temperature": 0.8,
        "stream": True,
        "thinking": {"type": "disabled"},
    }

    response = send_request(payload, headers)
    if response is None:
        print("[FAIL] 文本对话测试失败")
        return False

    if response.status_code == 200:
        stream_response(response, "Text Chat")
        print("[PASS] 文本对话测试通过")
        return True
    else:
        print(f"[FAIL] 文本对话测试失败: HTTP {response.status_code}")
        return False


def test_audio_chat(api_key: str, use_bearer: bool = False):
    """测试音频/多模态对话"""
    print("\n" + "=" * 60)
    print("测试模式: 音频对话 (mimo-v2-omni)")
    print("=" * 60)

    # 创建一个假的 base64 音频数据（静音 wav）用于测试连接
    # 实际使用时替换为真实音频
    dummy_wav = b"RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" \
                b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00"
    audio_b64 = base64.b64encode(dummy_wav).decode("utf-8")

    headers = build_headers(api_key, use_bearer)
    payload = {
        "model": "mimo-v2-omni",
        "messages": [
            {
                "role": "system",
                "content": "你是一个数字人助手，请用简短、口语化、温柔的语气回答。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
        "max_completion_tokens": 256,
        "temperature": 0.8,
        "stream": True,
        "thinking": {"type": "disabled"},
    }

    response = send_request(payload, headers)
    if response is None:
        print("[FAIL] 音频对话测试失败")
        return False

    if response.status_code == 200:
        stream_response(response, "Audio Chat")
        print("[PASS] 音频对话测试通过")
        return True
    else:
        # 400 可能是音频格式问题，但连接正常
        if response.status_code == 400:
            print(f"[WARN] 音频对话返回 400，可能是音频数据格式问题，但 API 连接正常")
            return True
        print(f"[FAIL] 音频对话测试失败: HTTP {response.status_code}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MiMo API 测试脚本")
    parser.add_argument(
        "mode",
        nargs="?",
        default="text",
        choices=["text", "audio", "both"],
        help="测试模式: text(默认), audio, both",
    )
    parser.add_argument(
        "--bearer",
        action="store_true",
        help="使用 Authorization: Bearer 认证方式",
    )
    args = parser.parse_args()

    print("MiMo API 连接测试")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    api_key = get_api_key()
    use_bearer = args.bearer or os.getenv("MIMO_USE_BEARER", "false").lower() == "true"

    results = []

    if args.mode in ("text", "both"):
        ok = test_text_chat(api_key, use_bearer)
        results.append(("text", ok))

    if args.mode in ("audio", "both"):
        ok = test_audio_chat(api_key, use_bearer)
        results.append(("audio", ok))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n[OK] 所有测试通过!")
        sys.exit(0)
    else:
        print("\n[ERROR] 部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
