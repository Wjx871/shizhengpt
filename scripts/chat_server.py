"""Small local-only research chat UI, using the unchanged base model."""
import argparse
import base64
import io
import json
import pathlib
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from qwen_vl_utils import process_vision_info
from infer_image import Engine, ROOT
from cautious_chat import run as cautious_v1
from cautious_chat_v4 import run as cautious_run, VERSION as CAUTIOUS_VERSION
from chat_history import restore as restore_history
from chat_api import API_VERSION, normalize_request, format_response, model_info, openapi_spec

WEB = ROOT / 'web'
MAX_BODY = 12 * 1024 * 1024
SESSIONS = {}
LOCK = threading.Lock()
ENGINE = None


def chat(payload):
    text = payload.get('text', '')
    if not isinstance(text, str) or len(text) > 6000:
        raise ValueError('文字最多 6000 字。')
    encoded = payload.get('image')
    if not text.strip() and not encoded:
        raise ValueError('请输入文字或选择图片。')
    mode = payload.get('mode', 'guarded')
    if mode not in ('guarded', 'guarded_v1', 'raw'):
        raise ValueError('回答模式无效。')
    cap=payload.get('max_new_tokens',768)
    if isinstance(cap,bool) or not isinstance(cap,int) or not 1<=cap<=1024:
        raise ValueError('max_new_tokens 必须是 1 到 1024 的整数。')
    sid = payload.get('session_id')
    if sid is not None and (not isinstance(sid,str) or not sid or len(sid)>80):
        raise ValueError('session_id 必须是有效的会话字符串。')
    now = time.time()
    for key, old in list(SESSIONS.items()):
        if now - old['updated'] > 7200:
            old['temp'].cleanup()
            del SESSIONS[key]
    if sid is not None and sid not in SESSIONS:
        raise ValueError('对话已过期，请开始新对话。')
    if sid and SESSIONS[sid]['mode'] != mode:
        raise ValueError('切换回答模式后请开始新对话。')
    sid = sid or str(uuid.uuid4())
    new = sid not in SESSIONS
    session = SESSIONS.get(sid) or {'messages': [], 'temp': tempfile.TemporaryDirectory(prefix='shizhen-chat-'), 'updated': now, 'mode': mode, 'user_texts': [], 'latest_image': None}
    path = None
    try:
        content = []
        if encoded:
            if not isinstance(encoded, str):
                raise ValueError('图片格式无效。')
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError('图片最多 8 MB。')
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ('JPEG','PNG','WEBP'):
                    raise ValueError('图片仅支持 JPEG、PNG 或 WebP。')
                if im.width * im.height > 24000000:
                    raise ValueError('图片像素过大，请缩小后重试。')
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((1600, 1600))
                path = pathlib.Path(session['temp'].name) / (str(uuid.uuid4()) + '.jpg')
                im.save(path, quality=95)
            content.append({'type': 'image', 'image': str(path), 'max_pixels': ENGINE.config['max_pixels']})
        content.append({'type': 'text', 'text': text.strip() or '请描述这张图片中可以直接观察到的内容。'})
        messages = session['messages'] + [{'role': 'user', 'content': content}]
        if len(messages) > 29:
            raise ValueError('本轮对话已达到 15 次回复，请开始新对话。')
        start = time.perf_counter()
        if mode in ('guarded', 'guarded_v1'):
            user_texts=session['user_texts']+[text.strip()]
            if sum(len(s) for s in user_texts)>16000:
                raise ValueError('对话内容过长，请开始新对话。')
            workflow=cautious_v1 if mode=='guarded_v1' else cautious_run
            guard=workflow(lambda prompt,image=None: ENGINE.infer(prompt,image,max_new_tokens=512),user_texts,path or session['latest_image'])
            answer=guard['answer']
            session['messages']=messages+[{'role':'assistant','content':[{'type':'text','text':answer}]}]
            session['user_texts']=user_texts
            session['latest_image']=str(path) if path else session['latest_image']
            session['updated']=time.time();SESSIONS[sid]=session
            # Full candidate trace stays in this in-memory session for review;
            # rejected free-form image observations are never rendered to users.
            session['last_guard_trace']=guard['trace']
            return {'session_id':sid,'answer':answer,'latency_sec':round(time.perf_counter()-start,1),'truncated':False,
                    'mode':mode,'guard_status':guard['status'],'workflow_version':guard['version'],'model_invocations':guard['model_invocations']}
        template = ENGINE.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = ENGINE.processor(text=[template], images=images, videos=videos, padding=True, return_tensors='pt')
        if inputs.input_ids.shape[1] > 12000:
            raise ValueError('对话内容过长，请开始新对话。')
        inputs = inputs.to(ENGINE.model.device)
        with torch.inference_mode():
            ids = ENGINE.model.generate(**inputs, max_new_tokens=cap, do_sample=False)
        trimmed = ids[:, inputs.input_ids.shape[1]:]
        answer = ENGINE.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        session['messages'] = messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}]
        session['updated'] = time.time()
        SESSIONS[sid] = session
        return {'session_id': sid, 'answer': answer, 'latency_sec': round(time.perf_counter() - start, 1), 'truncated': trimmed.shape[1] >= cap, 'mode':mode, 'guard_status':'raw_unchecked',
                'usage':{'input_tokens':int(inputs.input_ids.shape[1]),'output_tokens':int(trimmed.shape[1])}}
    except Exception:
        if path:
            path.unlink(missing_ok=True)
        if new:
            session['temp'].cleanup()
        raise


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, data):
        raw = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        if code==503:self.send_header('Retry-After','2')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ('/api/health','/healthz'):
            return self.respond(200, {'ready': True, 'model': 'ShizhenGPT-7B-VL', 'weights_version':'base-bf16',
                'default_mode':'guarded','api_v1_default_mode':'raw','api_version':API_VERSION,'workflow_version':CAUTIOUS_VERSION})
        if self.path=='/v1/models':
            info=model_info(CAUTIOUS_VERSION)
            info['runtime']={'dtype':str(next(ENGINE.model.parameters()).dtype),'device':str(ENGINE.model.device),
                'device_name':torch.cuda.get_device_name(0),'model_class':type(ENGINE.model).__name__,
                'lora_loaded':bool(getattr(ENGINE.model,'peft_config',None)),
                'attention':ENGINE.model.config._attn_implementation}
            return self.respond(200,info)
        if self.path=='/openapi.json':return self.respond(200,openapi_spec())
        if self.path in ('/app.js','/app.css'):
            raw=(WEB/self.path[1:]).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','application/javascript; charset=utf-8' if self.path.endswith('.js') else 'text/css; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-cache')
            self.end_headers();self.wfile.write(raw);return
        if self.path not in ('/', '/index.html'):
            return self.respond(404, {'error': '页面不存在'})
        raw = (WEB / 'index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path.startswith('/v1/'):
            return self.api_v1_post()
        if self.path not in ('/api/chat', '/api/reset', '/api/restore'):
            return self.respond(404, {'error': '接口不存在'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            # Restoring up to 15 image turns can exceed a single chat request.
            body_limit=170*1024*1024 if self.path=='/api/restore' else MAX_BODY
            if not 0 < length <= body_limit:
                raise ValueError('请求过大或为空。')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('请求格式无效。')
            if not LOCK.acquire(blocking=False):
                return self.respond(409, {'error': '模型正在回答，请稍后重试。'})
            try:
                if self.path == '/api/reset':
                    session = SESSIONS.pop(payload.get('session_id'), None)
                    if session:
                        session['temp'].cleanup()
                    result = {'ok': True}
                elif self.path == '/api/restore':
                    result = restore_history(payload,SESSIONS,ENGINE.config['max_pixels'])
                else:
                    result = chat(payload)
            finally:
                LOCK.release()
            self.respond(200, result)
        except (ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError) as exc:
            self.respond(400, {'error': str(exc)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.respond(500, {'error': '模型处理失败，请重试或查看服务器日志。'})

    def api_v1_post(self):
        request_id=str(uuid.uuid4())
        def fail(status,code,message):
            self.respond(status,{'request_id':request_id,'error':{'code':code,'message':message}})
        if self.path not in ('/v1/chat','/v1/sessions/reset'):
            return fail(404,'not_found','接口不存在。')
        try:
            if 'application/json' not in self.headers.get('Content-Type','').lower():
                return fail(400,'invalid_content_type','Content-Type 必须为 application/json。')
            length=int(self.headers.get('Content-Length','0'))
            if length>MAX_BODY:return fail(413,'request_too_large','请求最多 12 MB。')
            if length<=0:raise ValueError('请求体不能为空。')
            payload=json.loads(self.rfile.read(length))
            if self.path=='/v1/chat':
                payload=normalize_request(payload)
            elif not isinstance(payload,dict) or set(payload)!={'session_id'} or not isinstance(payload['session_id'],str) or not 1<=len(payload['session_id'])<=80:
                raise ValueError('清理会话时必须提供 session_id 字符串。')
            if not LOCK.acquire(blocking=False):
                return fail(503,'model_busy','模型正在处理其他请求，请稍后重试。')
            try:
                if self.path=='/v1/sessions/reset':
                    session=SESSIONS.pop(payload['session_id'],None)
                    if session:session['temp'].cleanup()
                    result={'request_id':request_id,'ok':True}
                else:
                    result=format_response(chat(payload),request_id)
            finally:
                LOCK.release()
            self.respond(200,result)
        except (ValueError,UnicodeError,Image.DecompressionBombError,Image.DecompressionBombWarning,UnidentifiedImageError) as exc:
            fail(400,'invalid_request',str(exc))
        except Exception:
            import traceback
            print('API request failed:',request_id,flush=True);traceback.print_exc()
            fail(500,'inference_failed','模型处理失败，请提供 request_id 联系服务维护者。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    ENGINE = Engine()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Chat ready on http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
