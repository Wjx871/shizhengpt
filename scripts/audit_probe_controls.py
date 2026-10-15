"""Audit follow-up raw outputs; match visibility, not clinical diagnoses."""
import hashlib
import json
import pathlib

from guarded_consultation import parse_json
from audit_representation_probe import schema_reasons

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/representation_probe_controls_20261008'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol=read(OUT/'protocol.json');run=read(OUT/'run.json');cases=read(OUT/'manifest.json')
    assert run['model_inferences']==len(cases)==8 and run['source_sha256']==protocol['source_sha256']
    for name,sha in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==sha
        assert hashlib.sha256((OUT/'sources'/name).read_bytes()).hexdigest()==sha
    review=[]
    for case in cases:
        result=read(OUT/'outputs'/f"{case['id']}.json")
        assert result['prompt']==case['prompt'] and result['evaluation_config']==run['config']
        assert result['generation_config']['max_new_tokens']==768 and result['generation_config']['do_sample'] is False
        if case['image']:assert result['image_sha256']==case['image_sha256'] and result['image_grid_thw']
        else:assert not result['image_sha256'] and not result['image_path'] and result['image_grid_thw'] is None
        obj,error=parse_json(result['raw_output']);schema=[] if error else schema_reasons(obj)
        statuses=obj.get('visual_status',{}) if isinstance(obj,dict) else {}
        if not isinstance(statuses,dict):statuses={}
        matched=sum(statuses.get(k)==v for k,v in case['expected_visual_status'].items())
        abstentions=sum(statuses.get(k)=='unknown' and v!='unknown' for k,v in case['expected_visual_status'].items())
        contradicted=sum(statuses.get(k) in ['visible','not_visible','yes','no'] and statuses.get(k)!=v for k,v in case['expected_visual_status'].items())
        allowed={f'S{i+1}' for i in range(len(case['symptoms']))}|{f'I{i+1}' for i in range(len(case['history']))}
        used=[]
        if isinstance(obj,dict) and isinstance(obj.get('possible_explanations'),list):
            for item in obj['possible_explanations']:
                if isinstance(item,dict) and isinstance(item.get('basis_ids'),list):used.extend(item['basis_ids'])
        invalid_ids=[ident for ident in used if not isinstance(ident,str) or ident not in allowed]
        clinical_output_on_visual_only=bool(case['kind']=='visual_only' and isinstance(obj,dict) and (obj.get('possible_explanations') or obj.get('questions')))
        review.append(dict(case_id=case['id'],kind=case['kind'],symptoms=case['symptoms'],history=case['history'],
            parsed_output=obj,parse_error=error,schema_reasons=schema,raw_output=result['raw_output'],
            visual_items_matched=matched,abstentions_on_known_items=abstentions,unsupported_visual_assertions=contradicted,
            invalid_basis_ids=invalid_ids,clinical_output_on_visual_only=clinical_output_on_visual_only,
            symptoms_exact=isinstance(obj,dict) and obj.get('user_symptoms')==case['symptoms'],
            history_exact=isinstance(obj,dict) and obj.get('additional_history')==case['history'],truncated=result['truncated']))
    summary=dict(model_inferences=8,schema_ok=sum(not r['parse_error'] and not r['schema_reasons'] for r in review),
        truncated=sum(r['truncated'] for r in review),clinical_accuracy_measured=False,
        visual_only=dict(conditions=4,matched=sum(r['visual_items_matched'] for r in review if r['kind']=='visual_only'),total=16,
            unsupported_assertions=sum(r['unsupported_visual_assertions'] for r in review if r['kind']=='visual_only'),
            clinical_output_violation_cases=sum(r['clinical_output_on_visual_only'] for r in review),
            invalid_basis_id_cases=sum(bool(r['invalid_basis_ids']) for r in review if r['kind']=='visual_only'),
            abstentions=sum(r['abstentions_on_known_items'] for r in review if r['kind']=='visual_only')),
        text_only=dict(conditions=4,symptoms_exact=sum(r['symptoms_exact'] for r in review if r['kind']=='text_only'),
            history_exact=sum(r['history_exact'] for r in review if r['kind']=='text_only')),
        note='Follow-up on the same known proxy fixtures; visual-only request differs from joint task, so encoder/fusion attribution remains tentative.')
    (OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
