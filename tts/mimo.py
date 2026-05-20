import time
import os
import requests
import json
import base64
import numpy as np
import resampy
import soundfile as sf
from io import BytesIO

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "mimo")
class MimoTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        voicename = textevent.get('tts', {}).get('voice', self.opt.REF_FILE)
        if not voicename or voicename == "zh-CN-YunxiaNeural": # Fallback to mimo default if default edge was passed
            voicename = "mimo_default"
            
        mimo_api_key = os.getenv("MIMO_API_KEY", "")
        if not mimo_api_key:
            logger.error("MIMO_API_KEY not found in environment")
            return
            
        t = time.time()
        
        headers = {
            "api-key": mimo_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mimo-v2.5-tts",
            "messages": [
                {
                    "role": "assistant",
                    "content": text
                }
            ],
            "audio": {
                "format": "wav",
                "voice": voicename
            }
        }

        try:
            response = requests.post(
                "https://api.xiaomimimo.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"MiMo TTS API error: {response.status_code} {response.text}")
                return
                
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                logger.error("MiMo TTS API returned no choices")
                return
                
            message = choices[0].get("message", {})
            audio_data_b64 = message.get("audio", {}).get("data", "")
            
            if not audio_data_b64:
                logger.error("MiMo TTS API returned no audio data")
                return
                
            audio_bytes = base64.b64decode(audio_data_b64)
            self.input_stream.write(audio_bytes)
            
            logger.info(f'-------mimo tts time:{time.time()-t:.4f}s')
            
            if self.input_stream.getbuffer().nbytes <= 0:
                logger.error('mimo tts err: stream empty')
                return
                
            self.input_stream.seek(0)
            stream = self.__create_bytes_stream(self.input_stream)
            streamlen = stream.shape[0]
            idx = 0
            
            while streamlen >= self.chunk and self.state == State.RUNNING:
                eventpoint = {}
                streamlen -= self.chunk
                if idx == 0:
                    eventpoint = {'status': 'start', 'text': text}
                elif streamlen < self.chunk:
                    eventpoint = {'status': 'end', 'text': text}
                eventpoint.update(**textevent)
                
                self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                idx += self.chunk
                
            self.input_stream.seek(0)
            self.input_stream.truncate()
            
        except Exception as e:
            logger.exception('mimo tts exception:')

    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
