"""Save real-model observations for manual review. No diagnosis accuracy claims."""
import base64
import json
import pathlib
from PIL import Image
from check_chat_api_live import call

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/visual_v3_live'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=ROOT/'evaluation/images/face/public_astronaut.png'
    with Image.open(original) as im:
        im.convert('L').save(OUT/'grey.png')
        im.crop((160,70,270,160)).save(OUT/'partial.png')
    Image.new('RGB',(320,320)).save(OUT/'black.png')
    cases=[('clear',original,'请描述图片中能看见的内容。'),
           ('grey',OUT/'grey.png','描述可见内容。'),
           ('partial',OUT/'partial.png','请描述图片，不补全未显示的部分。'),
           ('tongue',ROOT/'evaluation/images/tongue/3912.jpg','看看这张舌头照片。'),
           ('black',OUT/'black.png','描述图片。'),
           ('fusion',original,'我最近偶尔食欲不佳，睡眠质量一般。请结合图片看看。')]
    records=[]; failures=[]
    for name,path,text in cases:
        status,result,_=call('/v1/chat',{'mode':'guarded','text':text,'image_base64':base64.b64encode(path.read_bytes()).decode()})
        records.append({'case':name,'http_status':status,'result':result})
        (OUT/'results.json').write_bytes(json.dumps(records,ensure_ascii=False,indent=2).encode())
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
        if result.get('session_id'):call('/v1/sessions/reset',{'session_id':result['session_id']})
        if status!=200:failures.append(name+': http failure')
        elif name=='black' and result['guard_status']!='image_abstained':failures.append(name)
        elif name not in ('fusion','black') and result['guard_status']!='image_observed':failures.append(name)
    print('Completed image checks; review descriptions manually.',flush=True)
    assert not failures,failures


if __name__=='__main__':main()
