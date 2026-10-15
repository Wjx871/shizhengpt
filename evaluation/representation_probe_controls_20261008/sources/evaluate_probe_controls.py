"""Follow-up controls prompted by initial proxy results; not an independent test."""
import hashlib
import json
import pathlib
import shutil
import time

from evaluate_representation_probe import PROMPT, prompt_for

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evaluation/representation_probe_20261008'
OUT=ROOT/'evaluation/representation_probe_controls_20261008'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def freeze():
    if (OUT/'manifest.json').exists():raise RuntimeError('Preserve frozen controls')
    original=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    cases=[]
    for case in original:
        prompt=PROMPT+'\n本次没有用户症状或补充问诊信息，只完成照片可见性与视觉观察。user_symptoms、additional_history、possible_explanations和questions均输出空列表，limitations只说明照片观察局限。\n视觉输入：已附本次原始测试图。'
        cases.append(dict(id=case['id']+'_visual_only',kind='visual_only',image=case['image'],image_sha256=case['image_sha256'],
            expected_visual_status=case['expected_visual_status'],symptoms=[],history=[],prompt=prompt))
    for case in [original[0],original[2]]:
        for expanded in [False,True]:
            prompt=prompt_for(case,'image',expanded).replace('视觉输入：已附本次原始测试图。','视觉输入：未提供图片或图像描述。visual_status均输出unknown，visual_observations为空。只基于文字问诊，不能新增视觉信息。')
            cases.append(dict(id=case['scenario']+('_text_expanded' if expanded else '_text_basic'),kind='text_only',image=None,
                expected_visual_status=dict(eyes='unknown',nose='unknown',mouth='unknown',headgear_worn='unknown'),
                symptoms=case['symptoms'],history=case['expanded_history'] if expanded else [],prompt=prompt))
    save(OUT/'manifest.json',cases)
    names=['evaluate_probe_controls.py','evaluate_representation_probe.py','infer_image.py']
    hashes={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in names}
    save(OUT/'protocol.json',dict(conditions=8,output_cap=768,no_retries=True,source_sha256=hashes,frozen_at_unix=time.time(),
        limitation='Follow-up on known proxy fixtures. Visual-only task removes symptoms but also narrows requested task; not a pure encoder isolation.'))
    for name in names:
        target=OUT/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'scripts'/name,target)
    print('FROZEN 4 visual-only + 4 text-only controls')


def run():
    from infer_image import Engine
    cases=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,sha in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==sha
    engine=Engine();started=time.time()
    for case in cases:
        dest=OUT/'outputs'/f"{case['id']}.json"
        if dest.exists():raise RuntimeError('Preserve raw control output')
        image=str(ROOT/case['image']) if case['image'] else None
        if image:assert hashlib.sha256(pathlib.Path(image).read_bytes()).hexdigest()==case['image_sha256']
        print('START',case['id'],flush=True)
        result=engine.infer(case['prompt'],image,max_new_tokens=protocol['output_cap'])
        result.update(case_id=case['id'],kind=case['kind'],evaluation_config=engine.config,clinical_accuracy_not_measured=True)
        save(dest,result);print('DONE',case['id'],result['output_tokens'],result['truncated'],flush=True)
    save(OUT/'run.json',dict(model_inferences=len(cases),elapsed_sec=time.time()-started,config=engine.config,source_sha256=protocol['source_sha256']))
    print('COMPLETE',len(cases),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else run()
