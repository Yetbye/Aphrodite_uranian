import time
import os
import json
import requests
from typing import TYPE_CHECKING
from dotenv import load_dotenv

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# 加载 .env 文件
load_dotenv()

# 尝试导入 openai 作为备用 LLM
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    logger.warning("openai package not installed, fallback LLM (Qwen) will not be available")


def _get_mimo_api_key() -> str:
    """从环境变量或 .env 文件读取 MiMo API Key"""
    api_key = os.getenv("MIMO_API_KEY", "")
    # 去除可能的引号
    api_key = api_key.strip().strip('"').strip("'")
    return api_key


def _fallback_to_qwen(message, avatar_session: "BaseAvatar", datainfo: dict, label: str = "Fallback") -> bool:
    """
    当 MiMo API 失败时，回退到通义千问 (Qwen)
    返回 True 表示成功，False 表示失败
    """
    try:
        if not _HAS_OPENAI:
            logger.error(f"{label}: openai package not installed, cannot fallback to Qwen")
            return False
            
        dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip().strip('"').strip("'")
        if not dashscope_key:
            logger.error(f"{label}: DASHSCOPE_API_KEY not configured, cannot fallback to Qwen")
            return False
        
        logger.info(f"{label}: Falling back to Qwen (DashScope)")
        start = time.perf_counter()
        
        client = OpenAI(
            api_key=dashscope_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': '你是一个数字人助手，请用简短、口语化、温柔的语气回答。'},
                {'role': 'user', 'content': message}
            ],
            stream=True,
            stream_options={"include_usage": True}
        )
        
        result = ""
        first = True
        for chunk in completion:
            if len(chunk.choices) > 0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"{label} Qwen Time to first chunk: {end-start:.3f}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；":
                        result = result + msg[lastpos:i+1]
                        lastpos = i + 1
                        if len(result) > 10:
                            logger.info(f"{label} Qwen output: {result}")
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]
        
        end = time.perf_counter()
        logger.info(f"{label} Qwen Time to last chunk: {end-start:.3f}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)
        
        return True
        
    except Exception as e:
        logger.exception(f"{label} Qwen fallback error:")
        return False


def _build_headers(api_key: str, use_bearer: bool = False) -> dict:
    """
    构建请求头，支持两种方式：
    - use_bearer=False: api-key 头
    - use_bearer=True: Authorization: Bearer
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if use_bearer:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["api-key"] = api_key
    return headers


def _build_payload(model: str, messages: list, stream: bool = True, **kwargs) -> dict:
    """构建请求体"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    # 可选参数
    if "max_completion_tokens" in kwargs:
        payload["max_completion_tokens"] = kwargs["max_completion_tokens"]
    if "temperature" in kwargs:
        payload["temperature"] = kwargs["temperature"]
    if "thinking" in kwargs:
        payload["thinking"] = kwargs["thinking"]
    return payload


def _parse_sse_chunk(line: bytes) -> dict | None:
    """解析 SSE 流式响应的单个 chunk"""
    if not line:
        return None
    decoded_line = line.decode("utf-8").strip()
    if not decoded_line.startswith("data: "):
        return None
    data_str = decoded_line[6:]
    if data_str == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        logger.warning(f"MiMo SSE JSON parse error: {data_str[:200]}")
        return None


def _extract_content(chunk: dict) -> str:
    """从 chunk 中提取 content"""
    if not chunk or "choices" not in chunk:
        return ""
    choices = chunk.get("choices", [])
    if not choices:
        return ""
    delta = choices[0].get("delta", {})
    return delta.get("content", "") or ""


def _send_mimo_request(
    payload: dict,
    headers: dict,
    max_retries: int = 3,
    timeout: int = 30,
) -> requests.Response | None:
    """
    发送 MiMo API 请求，带重试机制
    """
    url = "https://api.xiaomimimo.com/v1/chat/completions"
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"MiMo API request attempt {attempt}/{max_retries}, "
                f"model={payload.get('model')}, stream={payload.get('stream')}, "
                f"messages_count={len(payload.get('messages', []))}"
            )
            logger.debug(f"MiMo API request headers: {json.dumps({k: v[:10] + '...' if k in ('api-key', 'Authorization') and isinstance(v, str) and len(v) > 20 else v for k, v in headers.items()})}")
            logger.debug(f"MiMo API request payload: {json.dumps(payload, ensure_ascii=False)}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=payload.get("stream", True),
                timeout=timeout,
            )
            logger.info(f"MiMo API response status: {response.status_code}")

            if response.status_code == 200:
                return response

            # 非 200 状态码，记录详细错误
            error_text = response.text[:500]
            logger.error(
                f"MiMo API HTTP error (attempt {attempt}/{max_retries}): "
                f"status={response.status_code}, "
                f"headers={dict(response.headers)}, "
                f"body={error_text}"
            )

            # 如果是 401/403，直接失败不重试
            if response.status_code in (401, 403):
                logger.error("MiMo API auth error, please check your API key.")
                return response

            # 如果是 429 (rate limit) 或 5xx，可以重试
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit or server error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            # 其他 4xx 错误，直接返回
            return response

        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"MiMo API timeout (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning(f"MiMo API connection error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            last_exception = e
            logger.warning(f"MiMo API request exception (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    logger.error(f"MiMo API failed after {max_retries} attempts. Last error: {last_exception}")
    return None


def _handle_streaming_response(
    response: requests.Response,
    avatar_session: "BaseAvatar",
    datainfo: dict,
    start_time: float,
    label: str = "MiMo LLM",
) -> None:
    """处理流式响应，按标点分句输出"""
    result = ""
    first = True
    chunk_count = 0

    try:
        for line in response.iter_lines():
            if not line:
                continue

            chunk_count += 1
            chunk = _parse_sse_chunk(line)
            if chunk is None:
                continue
            if chunk.get("__done__"):
                logger.info(f"{label} stream finished, total chunks={chunk_count}")
                break

            logger.debug(f"{label} chunk #{chunk_count}: {json.dumps(chunk, ensure_ascii=False)[:300]}")

            msg = _extract_content(chunk)
            if not msg:
                continue

            if first:
                end = time.perf_counter()
                logger.info(f"{label} Time to first chunk: {end - start_time:.3f}s")
                first = False

            # 按标点分句输出
            lastpos = 0
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；":
                    result = result + msg[lastpos : i + 1]
                    lastpos = i + 1
                    if len(result) > 10:
                        logger.info(f"{label} output: {result}")
                        avatar_session.put_msg_txt(result, datainfo)
                        result = ""
            result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"{label} Time to last chunk: {end - start_time:.3f}s, total chunks={chunk_count}")

        if result:
            logger.info(f"{label} final output: {result}")
            avatar_session.put_msg_txt(result, datainfo)

    except Exception as e:
        logger.exception(f"{label} stream processing error:")
        if result:
            avatar_session.put_msg_txt(result, datainfo)


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    """文本对话接口"""
    try:
        start = time.perf_counter()

        mimo_api_key = _get_mimo_api_key()
        if not mimo_api_key:
            logger.warning("MIMO_API_KEY not found in environment or .env file.")
            avatar_session.put_msg_txt("请配置 MIMO_API_KEY", datainfo)
            return

        # 优先使用 api-key 方式，如失败可切换为 Bearer
        use_bearer = os.getenv("MIMO_USE_BEARER", "false").lower() == "true"
        headers = _build_headers(mimo_api_key, use_bearer=use_bearer)

        model_name = "mimo-v2.5-pro"
        messages = [
            {"role": "system", "content": "你是一个数字人助手，请用简短、口语化、温柔的语气回答。"},
            {"role": "user", "content": message},
        ]
        payload = _build_payload(
            model=model_name,
            messages=messages,
            stream=True,
            max_completion_tokens=1024,
            temperature=0.8,
            thinking={"type": "disabled"},
        )

        end = time.perf_counter()
        logger.info(f"MiMo LLM Time init: {end - start:.3f}s, msg: {message}")

        response = _send_mimo_request(payload, headers, max_retries=3, timeout=30)

        if response is None:
            logger.error("MiMo LLM request failed after retries.")
            avatar_session.put_msg_txt("服务暂时不可用，请稍后再试。", datainfo)
            return

        if response.status_code != 200:
            error_msg = f"API 错误 (HTTP {response.status_code})"
            try:
                err_body = response.json()
                if "error" in err_body:
                    error_msg = err_body["error"].get("message", error_msg)
            except Exception:
                pass
            logger.error(f"MiMo LLM API error: {error_msg}")
            
            # 尝试回退到 Qwen
            logger.info("MiMo LLM failed, trying fallback to Qwen...")
            if _fallback_to_qwen(message, avatar_session, datainfo, label="MiMo LLM"):
                logger.info("MiMo LLM fallback to Qwen succeeded")
                return
            
            # 回退也失败，返回错误信息
            avatar_session.put_msg_txt(f"抱歉，{error_msg}，请稍后再试。", datainfo)
            return

        _handle_streaming_response(response, avatar_session, datainfo, start, label="MiMo LLM")

    except Exception as e:
        logger.exception("MiMo LLM exception:")
        # 尝试回退到 Qwen
        logger.info("MiMo LLM exception, trying fallback to Qwen...")
        try:
            if _fallback_to_qwen(message, avatar_session, datainfo, label="MiMo LLM"):
                logger.info("MiMo LLM fallback to Qwen succeeded")
                return
        except Exception:
            pass
        try:
            avatar_session.put_msg_txt("抱歉，发生了内部错误，请稍后再试。", datainfo)
        except Exception:
            pass
        return


def llm_audio_response(
    audio_b64,
    avatar_session: "BaseAvatar",
    datainfo: dict = {},
    images_b64: list = [],
):
    """音频/多模态对话接口"""
    try:
        start = time.perf_counter()

        mimo_api_key = _get_mimo_api_key()
        if not mimo_api_key:
            logger.warning("MIMO_API_KEY not found in environment or .env file.")
            avatar_session.put_msg_txt("请配置 MIMO_API_KEY", datainfo)
            return

        use_bearer = os.getenv("MIMO_USE_BEARER", "false").lower() == "true"
        headers = _build_headers(mimo_api_key, use_bearer=use_bearer)

        model_name = "mimo-v2-omni"

        content_array = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav",
                },
            }
        ]

        if images_b64:
            for img in images_b64:
                content_array.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                )

        messages = [
            {
                "role": "system",
                "content": "你是一个数字人助手，请用简短、口语化、温柔的语气回答。你能看到视频画面，如果用户给你发了画面，请结合画面和声音进行回复。",
            },
            {"role": "user", "content": content_array},
        ]
        payload = _build_payload(
            model=model_name,
            messages=messages,
            stream=True,
            max_completion_tokens=1024,
            temperature=0.8,
            thinking={"type": "disabled"},
        )

        end = time.perf_counter()
        logger.info(f"MiMo Audio LLM Time init: {end - start:.3f}s")

        response = _send_mimo_request(payload, headers, max_retries=3, timeout=30)

        if response is None:
            logger.error("MiMo Audio LLM request failed after retries.")
            avatar_session.put_msg_txt("服务暂时不可用，请稍后再试。", datainfo)
            return

        if response.status_code != 200:
            error_msg = f"API 错误 (HTTP {response.status_code})"
            try:
                err_body = response.json()
                if "error" in err_body:
                    error_msg = err_body["error"].get("message", error_msg)
            except Exception:
                pass
            logger.error(f"MiMo Audio LLM API error: {error_msg}")
            
            # 音频对话无法直接回退到 Qwen（因为 Qwen 不支持音频输入）
            # 但我们可以尝试用文本描述来回退
            logger.info("MiMo Audio LLM failed, trying fallback to Qwen with text description...")
            fallback_msg = f"[用户发送了语音消息]"
            if _fallback_to_qwen(fallback_msg, avatar_session, datainfo, label="MiMo Audio LLM"):
                logger.info("MiMo Audio LLM fallback to Qwen succeeded")
                return
            
            avatar_session.put_msg_txt(f"抱歉，{error_msg}，请稍后再试。", datainfo)
            return

        _handle_streaming_response(response, avatar_session, datainfo, start, label="MiMo Audio LLM")

    except Exception as e:
        logger.exception("MiMo Audio LLM exception:")
        # 尝试回退到 Qwen
        logger.info("MiMo Audio LLM exception, trying fallback to Qwen...")
        try:
            fallback_msg = f"[用户发送了语音消息]"
            if _fallback_to_qwen(fallback_msg, avatar_session, datainfo, label="MiMo Audio LLM"):
                logger.info("MiMo Audio LLM fallback to Qwen succeeded")
                return
        except Exception:
            pass
        try:
            avatar_session.put_msg_txt("抱歉，发生了内部错误，请稍后再试。", datainfo)
        except Exception:
            pass
        return
