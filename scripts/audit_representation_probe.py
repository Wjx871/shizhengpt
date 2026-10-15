"""Engineering audit of a proxy probe; no medical accuracy score."""
import csv
import hashlib
import json
import pathlib

from guarded_consultation import parse_json
from evaluate_representation_probe import ARMS

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/representation_probe_20261008'
KEYS={'visual_status','visual_observations','user_symptoms','additional_history','possible_explanations','questions','limitations'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def string_list(value):
    return isinstance(value,list) and all(isinstance(s,str) for s in value)


def schema_reasons(obj):
    if not isinstance(obj,dict) or set(obj)!=KEYS:
        return ['schema_keys']
    reasons=[]
    visual=obj['visual_status']
    if not isinstance(visual,dict) or set(visual)!={'eyes','nose','mouth','headgear_worn'}:
        reasons.append('visual_status_keys')
    elif any(visual[k] not in ['visible','not_visible','unknown'] for k in ['eyes','nose','mouth']) or visual['headgear_worn'] not in ['yes','no','unknown']:
        reasons.append('visual_status_values')
    for key, maximum in [('visual_observations',2),('questions',3),('limitations',2)]:
        if not string_list(obj[key]) or len(obj[key])>maximum:
            reasons.append(key+'_type_or_length')
    for key in ['user_symptoms','additional_history']:
        if not string_list(obj[key]):reasons.append(key+'_type')
    explanations=obj['possible_explanations']
    if not isinstance(explanations,list) or len(explanations)>2:
        reasons.append('possible_explanations_type_or_length')
    else:
        for item in explanations:
            if not isinstance(item,dict) or set(item)!={'basis_ids','text'} or not string_list(item['basis_ids']) or not isinstance(item['text'],str):
                reasons.append('explanation_schema')
    return sorted(set(reasons))


def main():
    manifest=read(OUT/'manifest.json')
    protocol=read(OUT/'protocol.json')
    run=read(OUT/'run.json')
    prompts=read(OUT/'frozen_prompts.json')
    assert run['model_inferences']==16 and run['source_sha256']==protocol['source_sha256']
    for name, expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==expected
        assert hashlib.sha256((OUT/'sources'/name).read_bytes()).hexdigest()==expected
    rows=[]
    review=[]
    for case in manifest:
        for arm, mode, expanded in ARMS:
            ident=f"{case['id']}_{arm}"
            result=read(OUT/'outputs'/f'{ident}.json')
            assert result['prompt']==prompts[ident]
            assert result['evaluation_config']==run['config']
            assert result['generation_config']['max_new_tokens']==protocol['output_cap']
            assert result['generation_config']['do_sample'] is False
            assert result['arm']==arm and result['expanded_history']==expanded
            if mode=='image':
                assert result['image_sha256']==case['image_sha256'] and result['image_grid_thw']
            else:
                assert not result['image_path'] and not result['image_sha256'] and result['image_grid_thw'] is None
            obj,error=parse_json(result['raw_output'])
            schema=[] if error else schema_reasons(obj)
            valid=not error and not schema
            history=case['expanded_history'] if expanded else []
            correct=0
            unsupported_visual_assertions=0
            abstentions_on_known_items=0
            mismatch={}
            ids_used=[]
            invalid_ids=[]
            if isinstance(obj,dict):
                status=obj.get('visual_status')
                if isinstance(status,dict):
                    for key, expected in case['expected_visual_status'].items():
                        if status.get(key)==expected:correct+=1
                        else:
                            actual=status.get(key)
                            mismatch[key]=dict(expected=expected,actual=actual)
                            if actual=='unknown' and expected!='unknown':abstentions_on_known_items+=1
                            elif actual in ['visible','not_visible','yes','no']:unsupported_visual_assertions+=1
                else:mismatch={'missing_visual_status':True}
                allowed={f'S{i+1}' for i in range(len(case['symptoms']))}|{f'I{i+1}' for i in range(len(history))}
                explanations=obj.get('possible_explanations',[])
                if isinstance(explanations,list):
                    for item in explanations:
                        if isinstance(item,dict) and string_list(item.get('basis_ids')):
                            ids_used.extend(item['basis_ids'])
                    invalid_ids=sorted(set(ids_used)-allowed)
            row=dict(case_id=case['id'],arm=arm,visual_mode=mode,expanded_history=expanded,
                json_parse_ok=not error,schema_ok=valid,truncated=result['truncated'],
                symptoms_exact=bool(isinstance(obj,dict) and obj.get('user_symptoms')==case['symptoms']),
                history_exact=bool(isinstance(obj,dict) and obj.get('additional_history')==history),
                nonmedical_visual_items_matched=correct,nonmedical_visual_items_total=4,
                unsupported_visual_assertions=unsupported_visual_assertions,
                abstentions_on_known_visual_items=abstentions_on_known_items,
                used_history_id=any(ident.startswith('I') for ident in ids_used),
                invalid_basis_ids=';'.join(invalid_ids),parse_error=error or '',schema_reasons=';'.join(schema),
                output_tokens=result['output_tokens'],latency_sec=result['latency_sec'])
            rows.append(row)
            review.append(dict(case_id=case['id'],arm=arm,expected_visual_status=case['expected_visual_status'],
                input_symptoms=case['symptoms'],input_history=history,visual_mode=mode,
                provided_description=case['description'] if mode=='description' else None,
                parsed_output=obj,parse_error=error,schema_reasons=schema,visual_mismatches=mismatch,
                raw_output=result['raw_output']))
    summary=dict(conditions=len(rows),model_inferences=16,clinical_accuracy_measured=False,
        description_source='assistant visual review, not doctor',arms={})
    for arm,_,_ in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        summary['arms'][arm]=dict(conditions=4,json_parse_ok=sum(r['json_parse_ok'] for r in selected),
            schema_ok=sum(r['schema_ok'] for r in selected),symptoms_exact=sum(r['symptoms_exact'] for r in selected),
            history_exact=sum(r['history_exact'] for r in selected),
            nonmedical_visual_items_matched=sum(r['nonmedical_visual_items_matched'] for r in selected),
            nonmedical_visual_items_total=16,used_history_id_cases=sum(r['used_history_id'] for r in selected),
            unsupported_visual_assertions=sum(r['unsupported_visual_assertions'] for r in selected),
            abstentions_on_known_visual_items=sum(r['abstentions_on_known_visual_items'] for r in selected),
            invalid_basis_id_cases=sum(bool(r['invalid_basis_ids']) for r in selected),
            truncated=sum(r['truncated'] for r in selected),total_model_latency_sec=round(sum(r['latency_sec'] for r in selected),2))
    summary['interpretation']='Visibility/headgear checks are nonmedical engineering labels. Description arms disclose these facts, so higher agreement does not establish diagnostic accuracy or independent visual generalization.'
    (OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
