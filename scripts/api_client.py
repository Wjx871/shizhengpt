"""Python stdlib client for the custom ShizhenGPT REST API (no SDK required)."""
import argparse
import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request


class ShizhenClient:
    def __init__(self,base_url='http://127.0.0.1:7860',timeout=180):
        self.base_url=base_url.rstrip('/');self.timeout=timeout

    def request(self,path,payload=None):
        body=json.dumps(payload,ensure_ascii=False).encode('utf-8') if payload is not None else None
        req=urllib.request.Request(self.base_url+path,body,{'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail=exc.read().decode('utf-8',errors='replace')
            raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc

    def chat(self,text='',image=None,session_id=None,mode='raw',max_new_tokens=None):
        payload={'text':text,'mode':mode}
        if image:payload['image_base64']=base64.b64encode(pathlib.Path(image).read_bytes()).decode('ascii')
        if session_id:payload['session_id']=session_id
        if max_new_tokens is not None:payload['max_new_tokens']=max_new_tokens
        return self.request('/v1/chat',payload)

    def reset(self,session_id):
        return self.request('/v1/sessions/reset',{'session_id':session_id})


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description='调用曙光服务器上的 ShizhenGPT 图文接口')
    p.add_argument('--base-url',default=os.environ.get('SHIZHENGPT_BASE_URL','http://127.0.0.1:7860'))
    p.add_argument('--text',default='');p.add_argument('--image');p.add_argument('--session-id')
    p.add_argument('--mode',choices=['raw','guarded','guarded_v1'],default='raw')
    p.add_argument('--max-new-tokens',type=int);p.add_argument('--timeout',type=int,default=180);p.add_argument('--output')
    a=p.parse_args()
    if not a.text.strip() and not a.image:p.error('至少提供 --text 或 --image')
    client=ShizhenClient(a.base_url,a.timeout)
    try:result=client.chat(a.text,a.image,a.session_id,a.mode,a.max_new_tokens)
    except (RuntimeError,urllib.error.URLError,TimeoutError,OSError) as exc:p.exit(1,str(exc)+'\n')
    if a.output:
        out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
