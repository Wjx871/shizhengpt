"""Restore browser-owned conversation history with bounded roles and images."""
import base64
import io
import pathlib
import tempfile
import time
import uuid
from PIL import Image, ImageOps


def restore(payload, sessions, max_pixels):
    mode=payload.get('mode','raw')
    if mode not in ('raw','guarded','guarded_v1'):raise ValueError('回答模式无效。')
    sid=payload.get('session_id')
    if sid is not None and (not isinstance(sid,str) or not 1<=len(sid)<=80):raise ValueError('会话编号无效。')
    if sid in sessions and sessions[sid]['mode']==mode:
        sessions[sid]['updated']=time.time()
        return {'session_id':sid,'restored':False}
    messages=payload.get('messages')
    if not isinstance(messages,list) or not 2<=len(messages)<=30 or len(messages)%2:
        raise ValueError('历史记录必须为最多15轮完整对话。')
    session={'messages':[],'temp':tempfile.TemporaryDirectory(prefix='shizhen-history-'),
             'mode':mode,'user_texts':[],'latest_image':None,'updated':time.time()}
    try:
        for i,item in enumerate(messages):
            role='user' if i%2==0 else 'assistant'
            if not isinstance(item,dict) or item.get('role')!=role:raise ValueError('历史消息角色或顺序无效。')
            text=item.get('text','')
            if not isinstance(text,str) or len(text)>(6000 if role=='user' else 24000):raise ValueError('历史消息文字过长。')
            content=[];encoded=item.get('image')
            if role=='assistant' and encoded:raise ValueError('助手消息不能包含上传图片。')
            if encoded:
                if not isinstance(encoded,str):raise ValueError('历史图片格式无效。')
                if encoded.startswith('data:'):
                    header,sep,encoded=encoded.partition(',')
                    if header not in ('data:image/jpeg;base64','data:image/png;base64','data:image/webp;base64') or not sep:
                        raise ValueError('历史图片类型无效。')
                data=base64.b64decode(encoded,validate=True)
                if len(data)>8*1024*1024:raise ValueError('历史图片过大。')
                with Image.open(io.BytesIO(data)) as im:
                    if im.format not in ('JPEG','PNG','WEBP') or im.width*im.height>24000000:raise ValueError('历史图片类型或大小无效。')
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1600,1600))
                    path=pathlib.Path(session['temp'].name)/(str(uuid.uuid4())+'.jpg');im.save(path,quality=95)
                content.append({'type':'image','image':str(path),'max_pixels':max_pixels})
                session['latest_image']=str(path)
            content.append({'type':'text','text':text or '请描述这张图片中可以直接观察到的内容。'})
            session['messages'].append({'role':role,'content':content})
            if role=='user':session['user_texts'].append(text)
        if sum(map(len,session['user_texts']))>16000:raise ValueError('历史对话内容过长，请新建对话。')
        sid=str(uuid.uuid4());sessions[sid]=session
        return {'session_id':sid,'restored':True}
    except Exception:
        session['temp'].cleanup()
        raise
