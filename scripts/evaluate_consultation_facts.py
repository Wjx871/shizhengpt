"""Frozen synthetic comparison: v2 versus v3. Not clinical patient validation."""
import hashlib
import json
import pathlib
import shutil
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/consultation_facts_20261008'
SOURCES = ['guarded_consultation.py', 'guarded_consultation_v3.py', 'consultation_facts.py', 'face_quality.py', 'evaluate_consultation_facts.py']


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def freeze():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Manifest already frozen; do not overwrite')
    texts = [
        ('h01', ['近三天轻微乏力', '否认发烧', '睡眠正常']),
        ('h02', ['最近一周偶有腹胀', '没有恶心和呕吐', '食欲尚可']),
        ('h03', ['有时头晕', '没有心慌', '没有胸部疼痛']),
        ('h04', ['昨天头疼，今天已经缓解', '没有发热']),
        ('h05', ['近两周有时食欲下降', '并非每天都有', '没有腹痛']),
        ('h06', ['睡眠良好', '最近五天白天有点疲乏', '没有咳嗽']),
        ('h07', ['最近一个月偶有手足凉', '没有便秘和腹泻', '睡眠质量一般']),
        ('h08', ['近四天有点腹胀', '没有发烧但偶尔咳嗽', '睡眠尚可']),
    ]
    cases = [dict(id=ident, symptoms=symptoms, image=None, kind='text') for ident, symptoms in texts]
    cases.extend([
        dict(id='h09', symptoms=texts[1][1], image='evaluation/images/face/astronaut_jpeg.png', kind='fusion'),
        dict(id='h10', symptoms=texts[2][1], image='evaluation/images/face/5467.jpg', kind='fusion'),
        dict(id='r01', symptoms=[], image='evaluation/images/face/public_astronaut.png', kind='visual_regression'),
        dict(id='r02', symptoms=[], image='evaluation/images/face/astronaut_original.png', kind='visual_regression'),
    ])
    for case in cases:
        case['group'] = 'known_visual_regression' if case['kind']=='visual_regression' else 'frozen_new_synthetic_text_condition'
        if case['image']:
            case['image_sha256'] = hashlib.sha256((ROOT / case['image']).read_bytes()).hexdigest()
    hashes = {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() for name in SOURCES}
    save(OUT / 'manifest.json', cases)
    save(OUT / 'freeze.json', dict(source_sha256=hashes, frozen_at_unix=time.time(),
        protocol='No rule/prompt changes after real outputs. Synthetic text holdout only; repeated public images are not independent patients. Same model/config/caps/retry count in both arms.'))
    for name in SOURCES:
        target = OUT / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'scripts' / name, target)
    print('FROZEN', len(cases))


def run():
    from infer_image import Engine
    from guarded_consultation import GuardedAssistant
    from guarded_consultation_v3 import FactGuardedAssistant
    frozen = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    for name, expected in frozen['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() == expected, name
    cases = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    engine = Engine()
    assistants = [('v2', GuardedAssistant(infer=engine.infer)), ('v3', FactGuardedAssistant(infer=engine.infer))]
    start = time.time()
    calls = 0
    for case in cases:
        image = str(ROOT / case['image']) if case['image'] else None
        if image: assert hashlib.sha256(pathlib.Path(image).read_bytes()).hexdigest() == case['image_sha256']
        for arm, assistant in assistants:
            dest = OUT / 'outputs' / f"{case['id']}_{arm}.json"
            if dest.exists(): raise RuntimeError('Do not overwrite historical output')
            print('START', case['id'], arm, flush=True)
            result = assistant.run(case['symptoms'], image)
            result.update(case_id=case['id'], arm=arm, evaluation_config=engine.config)
            save(dest, result)
            calls += result['model_invocations']
            print('DONE', case['id'], arm, result['analysis_action'], result['visual_action'], 'calls', result['model_invocations'], flush=True)
    save(OUT / 'run.json', dict(cases=len(cases), model_inferences=calls, elapsed_sec=time.time()-start, config=engine.config,
                               source_sha256=frozen['source_sha256']))
    print('COMPLETE', calls, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else run()
