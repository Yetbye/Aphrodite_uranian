###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm_mimo import llm_response, llm_audio_response
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import asyncio
import sys
import subprocess
import torch
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc


app = Flask(__name__)
#sockets = Sockets(app)
opt = None
model = None
global_avatars = {} # avatar_id: payload
        

#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    logger.info(f"[build_avatar_session] sessionid={sessionid}, params={ {k:v for k,v in params.items() if k not in ('refaudio',)} }")
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id)
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    logger.debug(f"[build_avatar_session] sessionid={sessionid}, avatar_id={avatar_id}, ref_audio={'set' if ref_audio else 'none'}, ref_text={'set' if ref_text else 'none'}")
    if (avatar_id and avatar_id != opt.avatar_id):
        # Avoid reloading if already cached globally
        if avatar_id not in global_avatars:
            logger.info(f"[build_avatar_session] Loading new avatar: {avatar_id} for sessionid={sessionid}")
            global_avatars[avatar_id] = load_avatar(avatar_id)
            logger.info(f"[build_avatar_session] Avatar loaded: {avatar_id}")
        else:
            logger.debug(f"[build_avatar_session] Using cached avatar: {avatar_id} for sessionid={sessionid}")
        avatar_this = global_avatars[avatar_id]
    else:
        # Default avatar loaded at startup
        avatar_this = global_avatars.get(opt.avatar_id)
        logger.debug(f"[build_avatar_session] Using default avatar: {opt.avatar_id} for sessionid={sessionid}")
    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
        logger.info(f"[build_avatar_session] sessionid={sessionid}, ref_audio configured")
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        opt_this.customopt = json.loads(custom_config)
        logger.info(f"[build_avatar_session] sessionid={sessionid}, custom_config applied")

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    logger.info(f"[build_avatar_session] sessionid={sessionid}, avatar_session created with model={opt.model}")
    return avatar_session

async def offer(request):
    logger.info(f"[offer] Received WebRTC offer request from {request.remote}")
    try:
        response = await rtc_manager.handle_offer(request)
        logger.info(f"[offer] WebRTC offer handled successfully for {request.remote}")
        return response
    except Exception as e:
        logger.error(f"[offer] WebRTC offer failed for {request.remote}: {e}")
        raise

async def on_shutdown(app):
    logger.info("[on_shutdown] Application shutting down, closing RTC manager...")
    await rtc_manager.shutdown()
    logger.info("[on_shutdown] RTC manager shutdown complete")



def main():
    global rtc_manager, opt, model,load_avatar
    # 解析命令行参数
    from config import parse_args
    opt = parse_args()
    logger.info(f"[main] Parsed args: model={opt.model}, avatar_id={opt.avatar_id}, transport={opt.transport}, listenport={opt.listenport}")

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
        'flashhead':  'avatars.flashhead_avatar',
        'liveact':    'avatars.liveact_avatar',
    }
    import importlib
    logger.info(f"[main] Loading avatar module for model: {opt.model}")
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(f"[main] Avatar module loaded: {_avatar_modules[opt.model]}")
    logger.info(opt)

    logger.info(f"[main] Starting model loading for {opt.model}...")
    if opt.model == 'musetalk':
        model = load_model()
        logger.info("[main] MuseTalk model loaded successfully")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        logger.info(f"[main] Default avatar loaded: {opt.avatar_id}")
        warm_up(opt.batch_size,model)
        logger.info("[main] MuseTalk warm_up completed")
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        logger.info("[main] Wav2Lip model loaded successfully")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        logger.info(f"[main] Default avatar loaded: {opt.avatar_id}")
        warm_up(opt.batch_size,model,256)
        logger.info("[main] Wav2Lip warm_up completed")
    elif opt.model == 'ultralight':
        model = load_model(opt)
        logger.info("[main] Ultralight model loaded successfully")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        logger.info(f"[main] Default avatar loaded: {opt.avatar_id}")
        warm_up(opt.batch_size,global_avatars[opt.avatar_id],160)
        logger.info("[main] Ultralight warm_up completed")
    elif opt.model == 'flashhead':
        if getattr(opt, 'launch_backend', False):
            # 用户要求在代码中拉起独立环境的推理后端进程
            env_name = getattr(opt, 'backend_env', 'flashhead')
            script_path = os.path.join(os.path.dirname(__file__), 'avatars', 'flashhead_core', 'flash_head', 'inference.py')
            cmd = f"conda run -n {env_name} python {script_path} --ckpt_dir {opt.flashhead_ckpt} --wav2vec_dir {opt.wav2vec_dir}"
            logger.info(f"[main] Starting FlashHead backend via subprocess: {cmd}")
            subprocess.Popen(cmd, shell=True)
            logger.info("[main] FlashHead backend subprocess started")
            # 在这里，我们只加载接口 wrapper，实际的 heavy 推理交给了刚刚拉起的进程或者 API
        else:
            model = load_model(opt)
            logger.info("[main] FlashHead model loaded successfully")
            global_avatars[opt.avatar_id] = load_avatar(opt, opt.avatar_id)
            logger.info(f"[main] Default avatar loaded: {opt.avatar_id}")
            warm_up(opt.batch_size, model)
            logger.info("[main] FlashHead warm_up completed")
    elif opt.model == 'liveact':
        if getattr(opt, 'launch_backend', False):
            # 用户要求在代码中拉起独立环境的推理后端进程
            env_name = getattr(opt, 'backend_env', 'liveact')
            script_path = os.path.join(os.path.dirname(__file__), 'avatars', 'liveact_core', 'demo.py') # 假设demo.py为服务端入口
            cmd = f"conda run -n {env_name} python {script_path} --ckpt_dir {opt.liveact_ckpt} --wav2vec_dir {opt.wav2vec_dir}"
            logger.info(f"[main] Starting LiveAct backend via subprocess: {cmd}")
            subprocess.Popen(cmd, shell=True)
            logger.info("[main] LiveAct backend subprocess started")
            # 在这里，我们只加载接口 wrapper，实际的 heavy 推理交给了刚刚拉起的进程或者 API
        else:
            model = load_model(opt)
            logger.info("[main] LiveAct model loaded successfully")
            global_avatars[opt.avatar_id] = load_avatar(opt, opt.avatar_id)
            logger.info(f"[main] Default avatar loaded: {opt.avatar_id}")
            warm_up(opt.batch_size, model)
            logger.info("[main] LiveAct warm_up completed")

    # init rtc manager
    logger.info("[main] Initializing session manager builder and RTC manager...")
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
    logger.info("[main] RTC manager initialized")
    # share avatar_sessions (RTCManager handles it but routes.py expects it)

    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # session 0 for virtualcam
        logger.info("[main] Transport is virtualcam/rtmp, creating session 0...")
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()
        logger.info("[main] Session 0 render thread started")

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync["llm_audio_response"] = llm_audio_response

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)

    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync)

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        logger.info(f"[main] HTTP server started on 0.0.0.0:{opt.listenport}")
        if opt.transport=='rtcpush':
            logger.info(f"[main] RTCPush mode: starting {opt.max_session} push sessions...")
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                logger.debug(f"[main] RTCPush starting session {k}, url={push_url}")
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
            logger.info("[main] RTCPush sessions started")
        loop.run_forever()
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
    
    
    
