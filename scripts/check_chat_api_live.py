"""Real-service smoke checks: text, image, fusion, sessions and error protocol."""
import base64
import concurrent.futures
import json
import pathlib
import threading
import urllib.error
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/chat_api_v1_checks'


def call(path,payload=None,raw=None,content_type='application/json'):
    body=raw if raw is not None else (json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None)
    req=urllib.request.Request('http://127.0.0.1:7860'+path,body,{'Content-Type':content_type})
    try:
        with urllib.request.urlopen(req,timeout=180) as r:return r.status,json.load(r),dict(r.headers)
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read()),dict(e.headers)


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];sessions=[]
    def check(path,payload=None,expected=200,**kwargs):
        status,result,headers=call(path,payload,**kwargs)
        assert status==expected,(path,status,result)
        if status>=400:assert result.get('request_id') and isinstance(result.get('error'),dict),result
        records.append({'path':path,'status':status,'result':result})
        if result.get('session_id'):sessions.append(result['session_id'])
        return result
    try:
        health=check('/healthz');assert health['api_v1_default_mode']=='raw'
        model=check('/v1/models');assert model['weights_version']=='base-bf16' and not model['runtime']['lora_loaded']
        spec=check('/openapi.json');assert '/v1/chat' in spec['paths']
        text=check('/v1/chat',{'text':'我的测试编号是青松73。请只回答已记住。','max_new_tokens':24})
        assert text['mode']=='raw' and text['guard_status']=='raw_unchecked' and text['usage']['output_tokens']>0
        recall=check('/v1/chat',{'text':'我刚才的测试编号是什么？只回答编号。','session_id':text['session_id'],'max_new_tokens':24})
        assert '青松73' in recall['answer'],recall
        check('/v1/sessions/reset',{'session_id':text['session_id']})
        check('/v1/chat',{'text':'继续','session_id':text['session_id']},expected=400)
        image=base64.b64encode((ROOT/'evaluation/images/face/public_astronaut.png').read_bytes()).decode()
        photo=check('/v1/chat',{'image_base64':image,'max_new_tokens':64});assert photo['answer'] and photo['mode']=='raw'
        fusion=check('/v1/chat',{'text':'请用一句话描述图片，不做诊断。','image_base64':'data:image/png;base64,'+image,'max_new_tokens':64})
        assert fusion['answer'] and fusion['usage']['input_tokens']>0
        cautious=check('/v1/chat',{'text':'问诊前需要提供什么信息？','mode':'guarded'})
        assert cautious['mode']=='guarded' and cautious['usage'] is None and cautious['model_invocations']==0
        length=check('/v1/chat',{'text':'请详细解释问诊应提供哪些信息。','max_new_tokens':1})
        assert length['finish_reason']=='length' and length['truncated']
        for payload in [{},{'text':'你好','mode':'bad'},{'text':'你好','image':'wrong_field'},
                        {'image_base64':'YWJj'},{'text':'你好','session_id':[]},
                        {'text':'你好','mode':'guarded','max_new_tokens':32}]:
            check('/v1/chat',payload,expected=400)
        check('/v1/chat',raw=b'{',expected=400)
        check('/v1/chat',raw=b'hello',content_type='text/plain',expected=400)
        check('/v1/unknown',{},expected=404)
        # Both clients hit the same serialized model; one should receive 503.
        barrier=threading.Barrier(2)
        def concurrent_request():
            barrier.wait()
            return call('/v1/chat',{'text':'请只回答收到。','max_new_tokens':16})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:concurrent_request(),range(2)))
        assert sorted(s for s,_,_ in results)==[200,503],results
        for status,result,headers in results:
            records.append({'path':'/v1/chat','status':status,'result':result})
            if result.get('session_id'):sessions.append(result['session_id'])
            if status==503:assert result['error']['code']=='model_busy' and headers.get('Retry-After')=='2'
        (OUT/'results.json').write_text(json.dumps({'passed':True,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'passed':True,'http_checks':len(records),'weights':model['weights_version'],'runtime':model['runtime']},ensure_ascii=False))
    finally:
        for sid in set(sessions):call('/v1/sessions/reset',{'session_id':sid})


if __name__=='__main__':main()
