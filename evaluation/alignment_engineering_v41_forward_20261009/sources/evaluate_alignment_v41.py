"""Four forward proxy cases frozen after v4's early whole-answer failures."""
import argparse
import hashlib
import json
import pathlib
import time

from evaluate_alignment_v4 import infer, save

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/alignment_engineering_v41_forward_20261009'
SOURCES=['consultation_alignment_v4.py','consultation_alignment_v41.py','consultation_facts.py','guarded_consultation.py','face_quality.py','evaluate_alignment_v41.py']


def freeze():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    cases=[
        {'id':'f01','symptoms':['最近四天有点疲乏，休息后能减轻','睡眠正常','没有咳嗽和发热'],'image':None,'rubric_forbidden_phrases':['严重疲乏','睡眠质量差']},
        {'id':'f02','symptoms':['前天头疼，现在已经缓解','今天有轻微乏力','没有心慌'],'image':None,'rubric_forbidden_phrases':['头疼持续多久','严重乏力']},
        {'id':'f03','symptoms':['近六天睡眠良好','偶尔午后轻微疲劳','食欲正常'],'image':None,'rubric_forbidden_phrases':['睡眠质量一般','睡眠质量差','严重疲劳']},
        {'id':'f04','symptoms':['最近五天偶有腹胀','没有腹痛或发热','睡眠尚可'],'image':'evaluation/images/face/5467.jpg',
         'rubric_forbidden_phrases':['持续腹胀','睡眠质量差'],'expected_visual_status':{'eyes':'not_visible','nose':'not_visible','mouth':'not_visible'}},
    ]
    for c in cases:
        c['kind']='post_v4_forward_synthetic_proxy'
        if c['image']:c['image_sha256']=hashlib.sha256((ROOT/c['image']).read_bytes()).hexdigest()
    save(OUT/'manifest.json',cases)
    save(OUT/'protocol.json',{'source_sha256':{s:hashlib.sha256((ROOT/'scripts'/s).read_bytes()).hexdigest() for s in SOURCES},
         'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),'frozen_unix':time.time(),
         'comparison':'Schema-constrained base versus v4.1 component-retention workflow, same deterministic service/stage separation as v4',
         'development_context':'Designed after early v4 t01 and t03 failures. New synthetic phrasings; known image reused. Not independent clinical validation.',
         'limits':'Rule checks are bounded; component retention does not prove clinical validity. No weight changes.'})
    print('FROZEN_FORWARD',len(cases))


def run():
    from face_quality import analyze
    from consultation_alignment_v4 import run_case
    from consultation_alignment_v41 import run as run41
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()==protocol['manifest_sha256']
    for s,digest in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/'scripts'/s).read_bytes()).hexdigest()==digest,s
    for case in json.loads((OUT/'manifest.json').read_text(encoding='utf-8')):
        if case['image']:
            assert hashlib.sha256((ROOT/case['image']).read_bytes()).hexdigest()==case['image_sha256']
            case['image']=str(ROOT/case['image'])
        for arm in ['base','enhanced']:
            dest=OUT/'outputs'/f"{case['id']}_{arm}.json"
            if dest.exists():raise RuntimeError('Do not overwrite forward results')
            print('START',case['id'],arm,flush=True)
            result=run_case(infer,case) if arm=='base' else run41(infer,case,qc=analyze)
            result.update(case_id=case['id'],arm=arm,source_sha256=protocol['source_sha256'])
            save(dest,result)
            print('DONE',case['id'],arm,result['analysis_action'],result['model_invocations'],flush=True)
    save(OUT/'run.json',{'complete':True,'source_sha256':protocol['source_sha256']});print('COMPLETE_FORWARD',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    freeze() if args.freeze else run()
