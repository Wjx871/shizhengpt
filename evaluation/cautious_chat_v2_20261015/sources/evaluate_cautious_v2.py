"""Frozen development regression comparison using the existing raw-model API."""
import base64
import hashlib
import json
import pathlib
import shutil
import time
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/cautious_chat_v2_20261015'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def request(path,payload):
    req=urllib.request.Request('http://127.0.0.1:7860'+path,json.dumps(payload,ensure_ascii=False).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)


def infer(prompt,image=None):
    encoded=base64.b64encode(pathlib.Path(image).read_bytes()).decode() if image else None
    reply=request('/api/chat',{'mode':'raw','text':prompt,'image':encoded})
    request('/api/reset',{'session_id':reply['session_id']})
    return {'raw_output':reply['answer'],'latency_sec':reply['latency_sec'],'truncated':reply['truncated'],
            'prompt':prompt,'image_path':image,'generation_max_new_tokens':768}


def run():
    from PIL import Image
    from cautious_chat import run as v1
    from cautious_chat_v2 import run as v2
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Do not overwrite frozen results')
    Image.new('RGB',(320,320),(0,0,0)).save(OUT/'black.png')
    Image.new('RGB',(320,320),(128,128,128)).save(OUT/'grey.png')
    cases=[
        {'id':'p01','texts':['请问，我最近三天轻微疲劳，休息后减轻。'],'expects_source':True},
        {'id':'p02','texts':['请帮我分析，我今天有点腹胀。'],'expects_source':True},
        {'id':'p03','texts':['我如果发热会是什么原因？'],'hypothetical':True},
        {'id':'p04','texts':['我最近轻微疲劳。','那这个药的标准剂量是多少？'],'drug_question':True},
        {'id':'p05','texts':['我近四天偶尔轻微疲劳，休息后减轻。'],'expects_source':True},
        {'id':'p06','texts':['我今天有发热。我今天没有发热。'],'conflict':True},
        {'id':'p07','texts':['泽珂素ZK-918的标准剂量是多少？'],'drug_question':True},
        {'id':'p08','texts':['问诊前需要提供什么信息？'],'faq':True},
        {'id':'p09','texts':['我肚子不舒服。'],'colloquial':True},
        {'id':'p10','texts':['请描述这张图片。'],'image':'evaluation/images/face/public_astronaut.png','photo':'colour'},
        {'id':'p11','texts':['请描述这张图片。'],'image':'evaluation/images/face/5467.jpg','photo':'partial'},
        {'id':'p12','texts':['请描述这张图片。'],'image':'evaluation/cautious_chat_v2_20261015/black.png','photo':'black'},
        {'id':'p13','texts':['请描述这张图片。'],'image':'evaluation/cautious_chat_v2_20261015/grey.png','photo':'grey'},
    ]
    for c in cases:
        if c.get('image'):c['image_sha256']=hashlib.sha256((ROOT/c['image']).read_bytes()).hexdigest()
    names=['cautious_chat.py','cautious_chat_v2.py','consultation_alignment_v4.py','consultation_alignment_v41.py','consultation_facts.py','face_quality.py','evaluate_cautious_v2.py']
    hashes={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in names}
    save(OUT/'protocol.json',{'cases':cases,'source_sha256':hashes,'frozen_unix':time.time(),
         'backend':'Existing base-model raw chat API, fresh sessions; same 768-token limit for both arms. Web guarded mode uses 512, separately smoke-tested.',
         'scope':'Development regressions, synthetic statements and known public images; no clinical labels. Photo facts computed from pixels, no added semantic vision capability.',
         'known_bugs_used_for_development':['polite-prefix report loss','hypothetical treated as fact','old symptoms reused for drug question','blanket photo description refusal']})
    (OUT/'sources').mkdir(exist_ok=True)
    for n in names:shutil.copyfile(ROOT/'scripts'/n,OUT/'sources'/n)
    rows=[]
    for case in cases:
        for arm,assistant in [('v1',v1),('v2',v2)]:
            print('START',case['id'],arm,flush=True)
            image=str(ROOT/case['image']) if case.get('image') else None
            result=assistant(infer,case['texts'],image)
            sources=result['trace'].get('symptom_sources',[])
            checks={}
            if case.get('expects_source'):checks['source_retained']=bool(sources)
            if case.get('hypothetical'):checks['not_patient_fact']=not sources
            if case.get('drug_question'):checks['no_drug_guess']='待核实的可能方向' not in result['answer'] and result['model_invocations']==0
            if case.get('conflict'):checks['clarify_conflict']=result['status']=='contradiction_abstained'
            if case.get('faq'):checks['useful_help']='开始时间' in result['answer']
            if case.get('colloquial'):checks['productive_clarification']=result['status']=='needs_clarification'
            if case.get('photo'):
                checks['measured_facts_present']='图像分辨率' in result['answer']
                checks['no_unchecked_visual_assertion']=all(p not in result['answer'] for p in ['佩戴头盔','照片为黑白','未见皮疹'])
                if case['photo'] in ('black','partial','grey'):checks['photo_detail_abstained']=result['status']=='image_abstained'
            row={'id':case['id'],'arm':arm,'checks':checks,'result':result};rows.append(row)
            save(OUT/'outputs'/f"{case['id']}_{arm}.json",row)
            print('DONE',case['id'],arm,result['status'],result['model_invocations'],checks,flush=True)
    summary={}
    for arm in ['v1','v2']:
        group=[r for r in rows if r['arm']==arm]
        summary[arm]={'cases':len(group),'cases_passing_declared_checks':sum(all(r['checks'].values()) for r in group),
                      'model_invocations':sum(r['result']['model_invocations'] for r in group),
                      'explanation_conditions':sum(r['result']['status']=='limited_answer' for r in group),
                      'measurement':'Functional development-regression checks, not hallucination rate or diagnostic accuracy'}
    save(OUT/'summary.json',summary);print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
