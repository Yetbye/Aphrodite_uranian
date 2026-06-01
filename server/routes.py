###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import numpy as np
import asyncio
import time
from aiohttp import web

from utils.logger import logger


# ─── 访问日志中间件 ────────────────────────────────────────────────────────

@web.middleware
async def access_log_middleware(request, handler):
    """记录所有 HTTP 请求的访问日志"""
    start_time = time.time()
    client_ip = request.remote or "unknown"
    method = request.method
    path = request.path
    
    logger.info(f"[HTTP] {client_ip} {method} {path} - Request started")
    
    try:
        response = await handler(request)
        elapsed = (time.time() - start_time) * 1000
        status = response.status
        logger.info(f"[HTTP] {client_ip} {method} {path} - Response {status} in {elapsed:.2f}ms")
        return response
    except web.HTTPException as ex:
        elapsed = (time.time() - start_time) * 1000
        logger.warning(f"[HTTP] {client_ip} {method} {path} - HTTP {ex.status} in {elapsed:.2f}ms: {ex.reason}")
        raise
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"[HTTP] {client_ip} {method} {path} - ERROR in {elapsed:.2f}ms: {e}")
        raise


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


def _sanitize_params(params: dict) -> dict:
    """脱敏处理：移除敏感字段，用于日志记录"""
    if not isinstance(params, dict):
        return params
    sensitive_keys = {'audio_data', 'file', 'refaudio', 'password', 'token', 'secret'}
    sanitized = {}
    for k, v in params.items():
        if k in sensitive_keys:
            sanitized[k] = '<masked>'
        else:
            sanitized[k] = v
    return sanitized


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    client = request.remote
    logger.info(f"[human] Request from {client}")
    try:
        params: dict = await request.json()
        logger.debug(f"[human] Params from {client}: {_sanitize_params(params)}")

        sessionid: str = params.get('sessionid', '')
        logger.debug(f"[human] sessionid={sessionid}")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[human] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")

        if params.get('interrupt'):
            logger.info(f"[human] Interrupt requested for sessionid={sessionid}")
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')
            logger.debug(f"[human] TTS params for sessionid={sessionid}: {params.get('tts')}")

        msg_type = params.get('type', '')
        text = params.get('text', '')
        if msg_type == 'echo':
            logger.info(f"[human] Echo mode, sessionid={sessionid}, text_len={len(text)}")
            avatar_session.put_msg_txt(text, datainfo)
        elif msg_type == 'chat':
            logger.info(f"[human] Chat mode, sessionid={sessionid}, text_len={len(text)}")
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, text, avatar_session, datainfo
                )
            else:
                logger.warning(f"[human] llm_response not available for chat mode, sessionid={sessionid}")

        logger.info(f"[human] Response ok to {client}, sessionid={sessionid}, type={msg_type}")
        return json_ok()
    except Exception as e:
        logger.error(f"[human] Exception from {client}: {e}")
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    client = request.remote
    logger.info(f"[interrupt_talk] Request from {client}")
    try:
        params = await request.json()
        logger.debug(f"[interrupt_talk] Params from {client}: {_sanitize_params(params)}")
        sessionid = params.get('sessionid', '')
        logger.debug(f"[interrupt_talk] sessionid={sessionid}")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[interrupt_talk] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")
        avatar_session.flush_talk()
        logger.info(f"[interrupt_talk] Talk interrupted, sessionid={sessionid}, client={client}")
        return json_ok()
    except Exception as e:
        logger.error(f"[interrupt_talk] Exception from {client}: {e}")
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    client = request.remote
    logger.info(f"[humanaudio] Request from {client}")
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        file_size = len(filebytes)
        logger.debug(f"[humanaudio] sessionid={sessionid}, file_size={file_size} bytes, client={client}")

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[humanaudio] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        logger.info(f"[humanaudio] Audio file processed, sessionid={sessionid}, size={file_size} bytes, client={client}")
        return json_ok()
    except Exception as e:
        logger.error(f"[humanaudio] Exception from {client}: {e}")
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    client = request.remote
    logger.info(f"[set_audiotype] Request from {client}")
    try:
        params = await request.json()
        logger.debug(f"[set_audiotype] Params from {client}: {_sanitize_params(params)}")
        sessionid = params.get('sessionid', '')
        audiotype = params.get('audiotype', '')
        logger.debug(f"[set_audiotype] sessionid={sessionid}, audiotype={audiotype}")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[set_audiotype] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")
        avatar_session.set_custom_state(audiotype)
        logger.info(f"[set_audiotype] Custom state set, sessionid={sessionid}, audiotype={audiotype}, client={client}")
        return json_ok()
    except Exception as e:
        logger.error(f"[set_audiotype] Exception from {client}: {e}")
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    client = request.remote
    logger.info(f"[record] Request from {client}")
    try:
        params = await request.json()
        logger.debug(f"[record] Params from {client}: {_sanitize_params(params)}")
        sessionid = params.get('sessionid', '')
        record_type = params.get('type', '')
        logger.debug(f"[record] sessionid={sessionid}, type={record_type}")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[record] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")
        if record_type == 'start_record':
            avatar_session.start_recording()
            logger.info(f"[record] Recording started, sessionid={sessionid}, client={client}")
        elif record_type == 'end_record':
            avatar_session.stop_recording()
            logger.info(f"[record] Recording stopped, sessionid={sessionid}, client={client}")
        else:
            logger.warning(f"[record] Unknown record type: {record_type}, sessionid={sessionid}, client={client}")
        return json_ok()
    except Exception as e:
        logger.error(f"[record] Exception from {client}: {e}")
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    client = request.remote
    logger.info(f"[is_speaking] Request from {client}")
    try:
        params = await request.json()
        logger.debug(f"[is_speaking] Params from {client}: {_sanitize_params(params)}")
        sessionid = params.get('sessionid', '')
        logger.debug(f"[is_speaking] sessionid={sessionid}")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[is_speaking] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")
        speaking = avatar_session.is_speaking()
        logger.info(f"[is_speaking] sessionid={sessionid}, is_speaking={speaking}, client={client}")
        return json_ok(data=speaking)
    except Exception as e:
        logger.error(f"[is_speaking] Exception from {client}: {e}")
        logger.exception('is_speaking exception:')
        return json_error(str(e))


async def chat_audio(request):
    """音频输入，端到端大模型处理"""
    client = request.remote
    logger.info(f"[chat_audio] Request from {client}")
    try:
        params = await request.json()
        logger.debug(f"[chat_audio] Params from {client}: {_sanitize_params(params)}")
        sessionid = params.get('sessionid', '')
        logger.debug(f"[chat_audio] sessionid={sessionid}")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.error(f"[chat_audio] Session not found: sessionid={sessionid}, client={client}")
            return json_error("session not found")

        avatar_session.flush_talk()
        datainfo = {}
        if params.get('tts'):
            datainfo['tts'] = params.get('tts')
            logger.debug(f"[chat_audio] TTS params for sessionid={sessionid}: {params.get('tts')}")

        images_data = params.get('images_data', [])
        logger.debug(f"[chat_audio] sessionid={sessionid}, images_count={len(images_data)}")

        llm_audio_response = request.app.get("llm_audio_response")
        if llm_audio_response:
            asyncio.get_event_loop().run_in_executor(
                None, llm_audio_response, params['audio_data'], avatar_session, datainfo, images_data
            )
            logger.info(f"[chat_audio] LLM audio response triggered, sessionid={sessionid}, client={client}")
        else:
            logger.warning(f"[chat_audio] llm_audio_response not available, sessionid={sessionid}, client={client}")
        return json_ok()
    except Exception as e:
        logger.error(f"[chat_audio] Exception from {client}: {e}")
        logger.exception('chat_audio route exception:')
        return json_error(str(e))


# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    logger.info("[setup_routes] Registering API routes...")
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/chat_audio", chat_audio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    
    # 使用绝对路径配置静态文件服务
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.join(current_dir, 'web')
    
    if os.path.exists(web_dir):
        logger.info(f"[setup_routes] Serving static files from: {web_dir}")
        app.router.add_static('/', path=web_dir, name='static')
        
        # 添加根路径重定向到 webrtcapi.html
        async def index_redirect(request):
            return web.HTTPFound('/webrtcapi.html')
        app.router.add_get('/', index_redirect)
        
        logger.info("[setup_routes] Static files configured successfully")
    else:
        logger.error(f"[setup_routes] Web directory not found: {web_dir}")
    
    logger.info("[setup_routes] All routes registered successfully")
