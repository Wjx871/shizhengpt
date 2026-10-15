"""Two arms share instructions, image and input availability; CV facts are the intervention."""
import argparse
import csv
import hashlib
import json
import pathlib
import time
import traceback

from face_quality import analyze, annotate
from infer_image import Engine, ROOT

OUT = ROOT/'evaluation/route_b'


def compact_cv(result):
    keep = ['image_present', 'usable_for_technical_description', 'reject_reasons',
            'warnings', 'coordinate_frame', 'detected_face_bbox_xywh',
            'measurement_bbox_xywh', 'fixed_reference_roi_used', 'patches',
            'laplacian_variance', 'measurement_type', 'allow_medical_colour_inference']
    return {k: result[k] for k in keep if k in result}


def prepare():
    for folder in ['measurements', 'annotated', 'outputs']:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    original = analyze(ROOT/'evaluation/images/face/astronaut_original.png')
    reference = original['detected_face_bbox_xywh']
    cases = []
    for name in ['original', 'bright', 'dark', 'yellow', 'blue', 'blur', 'jpeg']:
        cases.append({'id':name, 'image':f'evaluation/images/face/astronaut_{name}.png',
                      'fixed_bbox':reference, 'symptoms':[]})
    for name, image in [('partial','evaluation/images/face/5467.jpg'),
                        ('diagram','evaluation/images/face/4923.jpg'),
                        ('shapes','evaluation/images/general/shapes.png')]:
        cases.append({'id':name, 'image':image, 'symptoms':[]})
    symptoms = json.load(open(ROOT/'configs/route_b_prompts.json'))['symptoms']
    for name, image in [('fusion_face','evaluation/images/face/astronaut_original.png'),
                        ('fusion_partial','evaluation/images/face/5467.jpg'),
                        ('fusion_no_image',None)]:
        cases.append({'id':name, 'image':image, 'symptoms':symptoms})
    for case in cases:
        path = ROOT/case['image'] if case['image'] else None
        facts = analyze(path, case.get('fixed_bbox'))
        case['cv_facts'] = facts
        (OUT/'measurements'/f"{case['id']}.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2))
        if path:
            annotate(path, facts, OUT/'annotated'/f"{case['id']}.jpg")
    full = ROOT/'evaluation/images/face/public_astronaut.png'
    facts = analyze(full)
    (OUT/'measurements/full_photo_detector_audit.json').write_text(json.dumps(facts, indent=2))
    annotate(full, facts, OUT/'annotated/full_photo_detector_audit.jpg')
    (OUT/'manifest.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    return cases


def parse_response(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(text)
        required = {'visual_facts', 'measurement_facts', 'user_symptoms', 'limitations', 'medical_status'}
        valid = isinstance(value, dict) and set(value) == required
        if valid:
            valid = all(isinstance(value[k], list) and all(isinstance(v, str) for v in value[k])
                        for k in ['visual_facts','measurement_facts','user_symptoms','limitations'])
            valid = valid and value['medical_status'] == 'unknown'
        return value, valid
    except (ValueError, TypeError):
        return None, False


def run(pilot=False):
    cases = prepare()
    if pilot:
        cases = [c for c in cases if c['id'] in ['original','partial','fusion_no_image']]
    prompt = json.load(open(ROOT/'configs/route_b_prompts.json'))['base']
    engine = Engine()
    start = time.time()
    records = []
    folder = OUT/('pilot_outputs' if pilot else 'outputs')
    folder.mkdir(exist_ok=True)
    repeats = 1 if pilot else 3
    for case in cases:
        for arm in ['prompt_only','cv_facts']:
            envelope = {'image_present': bool(case['image']), 'user_symptoms':case['symptoms'],
                        'cv_measurements':compact_cv(case['cv_facts']) if arm=='cv_facts' else None}
            full_prompt = prompt+'\n'+json.dumps(envelope, ensure_ascii=False)
            for repeat in range(1,repeats+1):
                ident = f"{case['id']}_{arm}_r{repeat}"
                print('START', ident, flush=True)
                try:
                    result = engine.infer(full_prompt, str(ROOT/case['image']) if case['image'] else None,
                                          max_new_tokens=512)
                    parsed, schema = parse_response(result['raw_output'])
                    result.update(test_id=ident, case_id=case['id'], arm=arm, repeat=repeat,
                                  status='ok', parsed_response=parsed, schema_valid=schema,
                                  symptoms_exact=isinstance(parsed,dict) and parsed.get('user_symptoms')==case['symptoms'],
                                  no_image_visual_empty=(isinstance(parsed,dict) and parsed.get('visual_facts')==[]) if not case['image'] else None,
                                  cv_gate=case['cv_facts']['usable_for_technical_description'],
                                  cv_latency_sec=case['cv_facts']['cv_latency_sec'])
                    if arm=='cv_facts':
                        # Deployment gate is deterministic; it does not rely on LLM obedience.
                        result['gated_pipeline_action'] = 'generate' if result['cv_gate'] else 'reject_before_generation'
                        result['gated_pipeline_result'] = parsed if result['cv_gate'] else {
                            'image_present':bool(case['image']), 'colour_assessment':'not_supported',
                            'reasons':case['cv_facts']['reject_reasons'], 'user_symptoms':case['symptoms'],
                            'medical_status':'unknown'}
                        result['ablation_bypassed_gate'] = not result['cv_gate']
                except Exception as exc:
                    traceback.print_exc()
                    result = {'test_id':ident, 'case_id':case['id'], 'arm':arm, 'repeat':repeat,
                              'status':'error', 'error':type(exc).__name__+': '+str(exc)}
                (folder/f'{ident}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                records.append(result)
                print('DONE',ident,result['status'],result.get('latency_sec'),
                      'schema',result.get('schema_valid'),'tokens',result.get('output_tokens'),flush=True)
    filename = 'pilot_records.json' if pilot else 'records.json'
    (OUT/filename).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    (OUT/('pilot_run.json' if pilot else 'run.json')).write_text(json.dumps({
        'elapsed_sec':time.time()-start,'model_load_sec':engine.load_sec,
        'records':len(records),'success':sum(r['status']=='ok' for r in records),
        'comparison':'matched prompt + metadata availability; only CV facts differ',
        'gated_reject_cases_are_llm_ablation_only':True},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        for c in prepare():
            print(c['id'],c['cv_facts']['usable_for_technical_description'],c['cv_facts']['reject_reasons'])
    else:
        run(args.pilot)
