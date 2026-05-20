import time
import os
import json
import requests
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        
        mimo_api_key = os.getenv("MIMO_API_KEY", "")
        if not mimo_api_key:
            logger.warning("MIMO_API_KEY not found in environment, falling back to dummy response.")
            avatar_session.put_msg_txt("请配置 MIMO_API_KEY", datainfo)
            return

        headers = {
            "api-key": mimo_api_key,
            "Content-Type": "application/json"
        }
        
        # Determine if we should use end-to-end multi-modal (mimo-v2-omni) or just text (mimo-v2.5-pro)
        model_name = "mimo-v2.5-pro"
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个数字人助手，请用简短、口语化、温柔的语气回答。"},
                {"role": "user", "content": message}
            ],
            "max_completion_tokens": 1024,
            "temperature": 0.8,
            "stream": True,
            "thinking": {"type": "disabled"}
        }

        end = time.perf_counter()
        logger.info(f"MiMo LLM Time init: {end-start}s, msg: {message}")
        
        response = requests.post(
            "https://api.xiaomimimo.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        )
        
        result = ""
        first = True
        
        if response.status_code != 200:
            logger.error(f"MiMo API error: {response.status_code} {response.text}")
            return
            
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            msg = delta.get("content", "")
                            
                            if first and msg:
                                end = time.perf_counter()
                                logger.info(f"MiMo LLM Time to first chunk: {end-start}s")
                                first = False
                                
                            if not msg:
                                continue
                                
                            lastpos = 0
                            for i, char in enumerate(msg):
                                if char in ",.!;:，。！？：；":
                                    result = result + msg[lastpos:i+1]
                                    lastpos = i + 1
                                    if len(result) > 10:
                                        logger.info(result)
                                        avatar_session.put_msg_txt(result, datainfo)
                                        result = ""
                            result = result + msg[lastpos:]
                    except json.JSONDecodeError:
                        pass
                        
        end = time.perf_counter()
        logger.info(f"MiMo LLM Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)
        
    except Exception as e:
        logger.exception('MiMo LLM exception:')
        return

def llm_audio_response(audio_b64, avatar_session: 'BaseAvatar', datainfo: dict = {}, images_b64: list = []):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        
        mimo_api_key = os.getenv("MIMO_API_KEY", "")
        if not mimo_api_key:
            logger.warning("MIMO_API_KEY not found in environment")
            return

        headers = {
            "api-key": mimo_api_key,
            "Content-Type": "application/json"
        }
        
        # Use omni model for audio processing
        model_name = "mimo-v2-omni"
        
        content_array = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav"
                }
            }
        ]
        
        if images_b64:
            for img in images_b64:
                content_array.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                })
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个数字人助手，请用简短、口语化、温柔的语气回答。你能看到视频画面，如果用户给你发了画面，请结合画面和声音进行回复。"},
                {
                    "role": "user", 
                    "content": content_array
                }
            ],
            "max_completion_tokens": 1024,
            "temperature": 0.8,
            "stream": True,
            "thinking": {"type": "disabled"}
        }

        end = time.perf_counter()
        logger.info(f"MiMo Audio LLM Time init: {end-start}s")
        
        response = requests.post(
            "https://api.xiaomimimo.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        )
        
        result = ""
        first = True
        
        if response.status_code != 200:
            logger.error(f"MiMo Audio API error: {response.status_code} {response.text}")
            return
            
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            msg = delta.get("content", "")
                            
                            if first and msg:
                                end = time.perf_counter()
                                logger.info(f"MiMo Audio LLM Time to first chunk: {end-start}s")
                                first = False
                                
                            if not msg:
                                continue
                                
                            lastpos = 0
                            for i, char in enumerate(msg):
                                if char in ",.!;:，。！？：；":
                                    result = result + msg[lastpos:i+1]
                                    lastpos = i + 1
                                    if len(result) > 10:
                                        logger.info(result)
                                        avatar_session.put_msg_txt(result, datainfo)
                                        result = ""
                            result = result + msg[lastpos:]
                    except json.JSONDecodeError:
                        pass
                        
        end = time.perf_counter()
        logger.info(f"MiMo Audio LLM Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)
        
    except Exception as e:
        logger.exception('MiMo Audio LLM exception:')
        return
