"""Paired legacy / guarded prototype evaluation; images are existing public fixtures."""
import hashlib
import json
import pathlib
import time
from infer_image import Engine, ROOT
from guarded_consultation import GuardedAssistant

OUT = ROOT/'evaluation/guarded_consultation_20261008'
OLD = ['最近容易疲劳','手脚较凉','睡眠质量一般','偶尔食欲不佳']
NEW = ['近两天有点疲劳','睡眠尚可','没有发热','没有胸痛']


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def cases():
    result = []
    for ident, name in [('face_original','astronaut_original.png'),('face_dark','astronaut_dark.png'),
                        ('face_blue','astronaut_blue.png'),('face_bright','astronaut_bright.png'),
                        ('face_partial','5467.jpg'),('non_face_shapes','../general/shapes.png')]:
        result.append(dict(id=ident, image='evaluation/images/face/'+name, symptoms=[], kind='face', group='development_regression'))
    result += [dict(id='text_original',image=None,symptoms=OLD,kind='text',group='development_regression'),
        dict(id='text_new_negations',image=None,symptoms=NEW,kind='text',group='new_synthetic_condition'),
        dict(id='fusion_original',image='evaluation/images/face/astronaut_original.png',symptoms=OLD,kind='fusion',group='development_regression'),
        dict(id='fusion_partial',image='evaluation/images/face/5467.jpg',symptoms=OLD,kind='fusion',group='development_regression'),
        dict(id='fusion_no_image',image=None,symptoms=OLD,kind='fusion',group='development_regression'),
        dict(id='fusion_full_photo_new_text',image='evaluation/images/face/public_astronaut.png',symptoms=NEW,kind='fusion',group='new_input_combination_same_public_person')]
    for case in result:
        if case['image']:
            image = ROOT/case['image']
            case['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
    return result


def legacy_prompt(case):
    if case['kind']=='face':
        return (ROOT/'evaluation/prompts/face.txt').read_text(encoding='utf-8')
    symptoms = '，'.join(case['symptoms'])
    if case['kind']=='text':
        return '用户明确提供：'+symptoms+'。请从中医角度进行初步分析，区分【用户自述】【可能的解释】【仍需询问的问题】【不确定性】。不要编造其他症状，不要直接确诊或自行给药。请简洁回答。'
    return '请综合图片中直接观察到的信息（如果没有图片请明确说明）和用户明确提供的信息，按照【视觉观察】【用户自述】【中医角度综合分析】【仍需询问的问题】【不确定性】分析。禁止自行添加用户没有描述的症状，不能仅凭图文确诊。用户明确提供：'+symptoms+'。请简洁回答。'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = cases()
    save(OUT/'manifest.json',manifest)
    engine = Engine()
    assistant = GuardedAssistant(infer=engine.infer,max_retries=1)
    started = time.time()
    record_count = 0
    for case in manifest:
        print('START',case['id'],flush=True)
        image = str(ROOT/case['image']) if case['image'] else None
        for arm in ['baseline','guarded']:
            dest = OUT/'outputs'/f"{case['id']}_{arm}.json"
            if dest.exists():
                raise ValueError('Preserve existing outputs; use a new directory for a new experiment')
            if arm=='baseline':
                payload = engine.infer(legacy_prompt(case),image,max_new_tokens=512)
                payload.update(status='ok')
                count = 1
            else:
                payload = assistant.run(case['symptoms'],image)
                payload.update(status='ok')
                count = payload['model_invocations']
            payload.update(case_id=case['id'],arm=arm,evaluation_config=engine.config)
            save(dest,payload)
            record_count += count
            print('DONE',case['id'],arm,'calls',count,payload.get('visual_action'),payload.get('analysis_action'),flush=True)
    save(OUT/'run.json',dict(cases=len(manifest),pipeline_results=len(manifest)*2,
        model_inferences=record_count,elapsed_sec=time.time()-started,config=engine.config,
        scripts_sha256={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                        for name in ['guarded_consultation.py','evaluate_guarded_consultation.py','face_quality.py']},
        note='Combined pipeline intervention. Public images/synthetic symptoms; no independent clinical patients. New conditions are not an independent patient holdout.'))
    print('COMPLETE',record_count,flush=True)


if __name__=='__main__':
    main()
