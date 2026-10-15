"""Prepare, run and score an image-only TongueDx2 evaluation. Labels never enter Engine."""
import argparse
import csv
import hashlib
import json
import pathlib
import re
import statistics
import tarfile
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATURES = dict(TonguePale='舌淡', TipSideRed='舌尖或舌边红', Spot='点刺',
                Ecchymosis='瘀斑', Crack='裂纹', Toothmark='齿痕',
                FurThick='苔厚', FurYellow='苔黄')
PROMPT = '''请仅依据本次提供的舌部照片判读可见舌象，逐项独立判断，不推断疾病、证候、脏腑状态或脉象，不编造病史。不使用文件名作为判读依据。
八项定义：TonguePale=舌体颜色偏淡；TipSideRed=舌尖或舌边发红；Spot=舌面点刺；Ecchymosis=舌面瘀斑；Crack=舌面裂纹；Toothmark=舌边齿痕；FurThick=舌苔厚；FurYellow=舌苔黄色。
每项只能填写整数1（可见该特征）、整数0（可判断无该特征）或null（图片不足以判断）。没有图片时八项必须全部null。不要把不确定写成0。
只输出一个JSON对象，包含features（上述八个英文键）、description（简短客观描述）和limitations（拍摄或判读局限字符串列表）。不要输出Markdown或额外解释。'''
DESCRIPTION_IDS = ['83', '90', '97', '104', '111', '118', '125', '130']
DESCRIPTION_PROMPT = '请仅根据图片中可以直接观察到的信息，客观描述舌象特征。依次分析：舌质颜色、舌体形态、舌苔颜色、舌苔厚薄、齿痕、裂纹、湿润程度、其他直接可见特征。严格分成【视觉观察】【中医解释】【不确定信息】。不要虚构用户症状，不要把中医理论解释写成疾病确诊。看不清的特征请明确说明。请简洁回答。'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(a):
    from PIL import Image
    images = sorted(a.images.glob('*.jpg'), key=lambda p: int(p.stem.split('_')[0]))
    rows = list(csv.DictReader(a.labels.open(encoding='utf-8-sig', newline='')))
    counts = Counter(r['id'] for r in rows)
    manifest, selected = [], []
    for image in images:
        ident = image.stem.split('_')[0]
        if counts[ident] != 1:
            raise ValueError(f'Expected one label for {ident}, found {counts[ident]}')
        row = next(r for r in rows if r['id'] == ident)
        if any(row[f] not in ('0', '1') for f in FEATURES):
            raise ValueError(f'Invalid label for {ident}')
        with Image.open(image) as im:
            size = list(im.size)
            im.verify()
        manifest.append(dict(id=ident, filename=image.name, sha256=sha(image), dimensions=size))
        selected.append({'id': ident, **{f: int(row[f]) for f in FEATURES}})
    if not images or len({r['sha256'] for r in manifest}) != len(images):
        raise ValueError('Empty image directory or exact duplicate images')
    a.output.mkdir(parents=True, exist_ok=True)
    save(a.output/'manifest.json', manifest)
    save(a.output/'labels.json', selected)
    prompt = a.prompt_file.read_text(encoding='utf-8') if getattr(a, 'prompt_file', None) else PROMPT
    (a.output/'prompt.txt').write_text(prompt, encoding='utf-8')
    save(a.output/'data_audit.json', dict(image_count=len(images), ids=[r['id'] for r in manifest],
        label_source=str(a.labels.resolve()), label_sha256=sha(a.labels),
        total_source_rows=len(rows), duplicate_ids_in_full_source=sum(n > 1 for n in counts.values()),
        selected_ids_all_unique=True, exact_duplicate_images=0,
        positive_counts={f: sum(r[f] for r in selected) for f in FEATURES},
        full_source_feature_counts={f: dict(Counter(r[f] for r in rows)) for f in FEATURES},
        image_source=str(a.images.resolve()), prompt_sha256=sha(a.output/'prompt.txt')))
    with tarfile.open(a.output/'inputs.tar.gz', 'w:gz') as tar:
        for image in images:
            tar.add(image, arcname='images/'+image.name)
        for name in ['manifest.json', 'prompt.txt']:
            tar.add(a.output/name, arcname=name)
    print(json.dumps({'prepared': len(images), 'output': str(a.output)}, ensure_ascii=False))


def parse(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        obj = json.loads(text)
        if 'features' not in obj and set(FEATURES).issubset(obj):
            obj = {'features': {f: obj[f] for f in FEATURES},
                   'description': obj.get('description', ''), 'limitations': obj.get('limitations', []),
                   'schema_note': 'Feature keys were at top level; losslessly normalized'}
        values = obj['features']
        if set(values) != set(FEATURES):
            raise ValueError('Missing or extra feature keys')
        if any(v is not None and (type(v) is not int or v not in (0, 1)) for v in values.values()):
            raise ValueError('Feature values must be integer 0/1 or null')
        return obj, None
    except (ValueError, KeyError, TypeError) as ex:
        return None, str(ex)


def run(a):
    from infer_image import Engine
    manifest = json.loads((a.output/'manifest.json').read_text(encoding='utf-8'))
    prompt = (a.output/'prompt.txt').read_text(encoding='utf-8')
    cases = manifest + [dict(id='no_image', filename=None)]
    engine = Engine()
    started = time.time()
    config = engine.config.copy()
    for case in cases:
        dest = a.output/'outputs'/f"{case['id']}.json"
        image = a.output/'images'/case['filename'] if case['filename'] else None
        if image and sha(image) != case['sha256']:
            raise ValueError(f"Image hash mismatch: {case['id']}")
        if dest.exists():
            prior = json.loads(dest.read_text(encoding='utf-8'))
            if (prior.get('status') == 'ok' and prior.get('prompt') == prompt
                    and prior.get('image_sha256') == case.get('sha256')
                    and prior.get('evaluation_config') == config):
                print('SKIP', case['id'], flush=True)
                continue
            raise ValueError(f'Existing incompatible or failed output: {dest}; preserve it and use a new run directory')
        print('START', case['id'], flush=True)
        try:
            result = engine.infer(prompt, str(image) if image else None)
            parsed, error = parse(result['raw_output'])
            result.update(id=case['id'], status='ok', parsed=parsed, parse_error=error, evaluation_config=config)
        except Exception as ex:
            result = dict(id=case['id'], status='error', error=f'{type(ex).__name__}: {ex}')
        save(dest, result)
        print('DONE', case['id'], result['status'], result.get('latency_sec'), result.get('parse_error'), flush=True)
    save(a.output/'run.json', dict(elapsed_sec=time.time()-started, model_load_sec=engine.load_sec,
         config=config, prompt_sha256=sha(a.output/'prompt.txt'), expected_inferences=len(cases)))


def ratio(n, d):
    return n/d if d else None


def describe(a):
    from infer_image import Engine
    manifest = json.loads((a.output/'manifest.json').read_text(encoding='utf-8'))
    engine = Engine()
    for case in manifest:
        if case['id'] not in DESCRIPTION_IDS:
            continue
        dest = a.output/'outputs'/f"description_{case['id']}.json"
        if dest.exists():
            raise ValueError(f'Preserve existing output: {dest}')
        image = a.output/'images'/case['filename']
        if sha(image) != case['sha256']:
            raise ValueError('Image hash mismatch')
        result = engine.infer(DESCRIPTION_PROMPT, str(image))
        result.update(id=case['id'], status='ok', evaluation_config=engine.config,
                      selection='Fixed spaced IDs; selection does not use feature labels')
        save(dest, result)
        print('DESCRIPTION_DONE', case['id'], result['latency_sec'], flush=True)


def score(a):
    manifest = json.loads((a.output/'manifest.json').read_text(encoding='utf-8'))
    labels = {r['id']: r for r in json.loads((a.output/'labels.json').read_text(encoding='utf-8'))}
    records, detailed = [], []
    for case in manifest:
        path = a.output/'outputs'/f"{case['id']}.json"
        r = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(status='not_run')
        if r.get('status') == 'ok':
            if r.get('image_sha256') != case['sha256']:
                raise ValueError('Output image mismatch')
            if r.get('prompt') != (a.output/'prompt.txt').read_text(encoding='utf-8'):
                raise ValueError('Output prompt mismatch')
            obj, err = parse(r['raw_output'])
            r.update(parsed=obj, parse_error=err)
        values = r.get('parsed', {}).get('features', {}) if r.get('parsed') else {}
        row = dict(id=case['id'], filename=case['filename'], status=r['status'], parse_error=r.get('parse_error'),
                   latency_sec=r.get('latency_sec'), truncated=r.get('truncated'), raw_output=r.get('raw_output', ''))
        for f in FEATURES:
            row['label_'+f] = labels[case['id']][f]
            row['pred_'+f] = values.get(f)
        records.append(r)
        detailed.append(row)
    metrics = []
    for f, name in FEATURES.items():
        pairs = [(r['label_'+f], r['pred_'+f]) for r in detailed]
        tp = sum(y == 1 and p == 1 for y, p in pairs)
        tn = sum(y == 0 and p == 0 for y, p in pairs)
        fp = sum(y == 0 and p == 1 for y, p in pairs)
        fn = sum(y == 1 and p == 0 for y, p in pairs)
        unknown = sum(p is None for y, p in pairs)
        positive = sum(y == 1 for y, p in pairs)
        sensitivity = ratio(tp, tp+fn)
        specificity = ratio(tn, tn+fp)
        metrics.append(dict(feature=f, name=name, n=len(pairs), positive=positive, tp=tp, tn=tn, fp=fp, fn=fn,
            unknown=unknown, coverage=ratio(len(pairs)-unknown,len(pairs)),
            strict_accuracy=ratio(tp+tn,len(pairs)), valid_accuracy=ratio(tp+tn,len(pairs)-unknown),
            precision=ratio(tp,tp+fp), recall=sensitivity, specificity=specificity,
            f1=ratio(2*tp,2*tp+fp+fn), balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            majority_baseline_accuracy=max(positive,len(pairs)-positive)/len(pairs)))
    for name, rows in [('predictions.csv', detailed), ('metrics.csv', metrics)]:
        with (a.output/name).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    control_path = a.output/'outputs/no_image.json'
    control = json.loads(control_path.read_text(encoding='utf-8')) if control_path.exists() else None
    if control and control.get('status') == 'ok':
        obj, err = parse(control['raw_output'])
        control.update(parsed=obj, parse_error=err)
    successful = [r for r in records if r['status']=='ok']
    result = dict(n=len(records), success=len(successful), missing=sum(r['status']=='not_run' for r in records),
        format_valid=sum(r.get('parsed') is not None for r in records),
        normalized_flat_schema=sum(bool((r.get('parsed') or {}).get('schema_note')) for r in records),
        truncated=sum(bool(r.get('truncated')) for r in records),
        exact_match=sum(all(r['pred_'+f] == r['label_'+f] for f in FEATURES) for r in detailed),
        strict_accuracy=sum(m['tp']+m['tn'] for m in metrics)/(len(records)*len(FEATURES)),
        majority_baseline_accuracy=sum(max(m['positive'],m['n']-m['positive']) for m in metrics)/(len(records)*len(FEATURES)),
        metrics=metrics, no_image_control=control,
        label_encoding_assumption='Provisional: 1 means feature present, 0 means absent; original annotation codebook not available.',
        label_agreement_status='provisional_pending_codebook',
        feature_prediction_coverage=sum(m['n']-m['unknown'] for m in metrics)/(len(records)*len(FEATURES)),
        accuracy_on_valid_predictions=ratio(sum(m['tp']+m['tn'] for m in metrics), sum(m['n']-m['unknown'] for m in metrics)),
        descriptive_runs=len(list((a.output/'outputs').glob('description_*.json'))),
        total_inferences=len(list((a.output/'outputs').glob('*.json'))),
        no_image_control_pass=bool(control and control.get('parsed') and all(
            v is None for v in control['parsed']['features'].values())),
        median_latency_sec=statistics.median(r['latency_sec'] for r in successful) if successful else None,
        metric_note='strict_accuracy: unknown, format failures and missing runs count as incorrect. Precision/recall/F1/balanced accuracy use only valid 0/1 predictions. Always interpret together with coverage.')
    save(a.output/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run','describe','score'])
    parser.add_argument('--images', type=pathlib.Path, default=ROOT.parent/'85-130 O')
    parser.add_argument('--labels', type=pathlib.Path, default=ROOT.parent/'TongueDx2原始标签总表_按编号排序.csv')
    parser.add_argument('--prompt-file', type=pathlib.Path, help='Optional evaluation prompt for prepare mode')
    parser.add_argument('--output', type=pathlib.Path, default=ROOT/'evaluation/tonguedx2_o_20261008')
    args = parser.parse_args()
    globals()[args.mode](args)
