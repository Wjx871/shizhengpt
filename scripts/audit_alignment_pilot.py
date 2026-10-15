"""Task-specific audit, with raw transfer regressions kept separate."""
import csv
import hashlib
import json
import pathlib
import re

from guarded_consultation import parse_json

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/alignment_lora_pilot_20261009'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def score_case(case,obj):
    gold=case['answer'];task=case['task']
    score=dict(full_target_exact=obj==gold,matched_fields=0,total_fields=0,quote_exact=0,quote_total=0,
               source_ids_exact=0,source_ids_total=0,visual_unsupported_assertions=0,invalid_basis_ids=[],
               symptom_degree_risks=[],regression_schema_compatible=False,task_schema_ok=False,
               explanation_present=False,questions_present=False)
    if task=='vision':
        score['total_fields']=6
        if isinstance(obj,dict):
            score['task_schema_ok']=set(obj)=={'image_type','visual_status','clinical_inference'} and isinstance(obj.get('visual_status'),dict) and set(obj['visual_status'])=={'eyes','nose','mouth','headgear_worn'}
            score['matched_fields']+=obj.get('image_type')==gold['image_type']
            score['matched_fields']+=obj.get('clinical_inference')==[]
            status=obj.get('visual_status',{})
            if isinstance(status,dict):
                for key,value in gold['visual_status'].items():
                    # Semantic scoring tolerates the base model placing headgear
                    # at root; strict schema remains separately visible.
                    actual=status.get(key,obj.get(key));score['matched_fields']+=actual==value
                    if actual in ['visible','not_visible','yes','no'] and actual!=value:
                        score['visual_unsupported_assertions']+=1
    elif task=='symptoms':
        score['total_fields']=len(gold['facts'])*6
        score['quote_total']=score['source_ids_total']=len(gold['facts'])
        score['task_schema_ok']=isinstance(obj,dict) and set(obj)=={'facts'} and isinstance(obj.get('facts'),list)
        # Compare actual records even if base returned a root array. Do not turn
        # an envelope-format failure into 108 semantic errors or modify raw text.
        actual=obj.get('facts',[]) if isinstance(obj,dict) else obj if isinstance(obj,list) else []
        if not isinstance(actual,list):actual=[]
        for index,fact in enumerate(gold['facts']):
            output=actual[index] if index<len(actual) and isinstance(actual[index],dict) else {}
            # Source identity plus the five semantic/copy fields, per record.
            for key in ['source_id','quote','polarity','severity','frequency','duration']:
                score['matched_fields']+=output.get(key)==fact[key]
            score['quote_exact']+=output.get('quote')==fact['quote']
            score['source_ids_exact']+=output.get('source_id')==fact['source_id']
    elif task=='citation':
        score['total_fields']=3;score['quote_total']=score['source_ids_total']=1
        if isinstance(obj,dict):
            score['task_schema_ok']=set(obj)=={'corrected_claim','basis'} and isinstance(obj.get('basis'),list)
            score['matched_fields']+=obj.get('corrected_claim')==gold['corrected_claim']
            basis=obj.get('basis',[])
            if isinstance(basis,list) and len(basis)==1 and isinstance(basis[0],dict):
                match_id=basis[0].get('source_id')==gold['basis'][0]['source_id']
                match_quote=basis[0].get('quote')==gold['basis'][0]['quote']
                score['matched_fields']+=match_id+match_quote
                score['source_ids_exact']+=match_id;score['quote_exact']+=match_quote
    else:
        score['full_target_exact']=None
        if isinstance(obj,dict):
            keys={'visual_status','visual_observations','user_symptoms','additional_history','possible_explanations','questions','limitations'}
            score['regression_schema_compatible']=set(obj)==keys
            score['explanation_present']=bool(obj.get('possible_explanations'))
            score['questions_present']=bool(obj.get('questions'))
            text=json.dumps(obj,ensure_ascii=False)
            # Narrow literal check only; raw outputs must also be reviewed.
            if re.search(r'睡眠(?:质量)?差',text):score['symptom_degree_risks'].append('sleep_generally_to_poor')
            explanations=obj.get('possible_explanations',[])
            allowed=set(re.findall(r'(?m)^([SI]\d+)：',case['prompt']))
            if isinstance(explanations,list):
                for item in explanations:
                    if isinstance(item,dict) and isinstance(item.get('basis_ids'),list):
                        score['invalid_basis_ids'].extend(str(i) for i in item['basis_ids'] if i not in allowed)
    return score


def main():
    protocol=read(OUT/'protocol.json');training=read(OUT/'training_run.json');evaluation=read(OUT/'evaluation_run.json')
    for name,expected in protocol['source_sha256'].items():
        assert sha(ROOT/'scripts'/name)==sha(OUT/'sources'/name)==expected
    assert sha(OUT/'train.jsonl')==protocol['train_jsonl_sha256']
    assert sha(OUT/'test.jsonl')==protocol['test_jsonl_sha256']
    assert sha(OUT/'adapter/adapter_model.safetensors')==training['adapter_sha256']
    assert training['optimizer_steps']==64 and training['base_gradients_none'] and training['base_sentinels_unchanged']
    cases=[json.loads(line) for line in (OUT/'test.jsonl').read_text(encoding='utf-8').splitlines()]+read(OUT/'regression.json')
    assert evaluation['model_inferences']==len(cases)*2==40
    scored=[];review=[]
    for case in cases:
        for arm in ['base','lora']:
            result=read(OUT/'outputs'/f"{case['id']}_{arm}.json")
            assert result['prompt']==case['prompt'] and result['evaluation_config']==evaluation['config']
            assert result['generation_config']['do_sample'] is False
            assert result['generation_config']['max_new_tokens']==(768 if case['task']=='regression' else 512)
            if case.get('image_sha256'):assert result['image_sha256']==case['image_sha256']
            if not case['image']:assert not result['image_sha256'] and result['image_grid_thw'] is None
            obj,error=parse_json(result['raw_output'])
            score=score_case(case,obj)
            score.update(id=case['id'],task=case['task'],arm=arm,json_parse_ok=not error,truncated=result['truncated'],
                output_tokens=result['output_tokens'],latency_sec=result['latency_sec'])
            scored.append(score)
            review.append(dict(id=case['id'],task=case['task'],arm=arm,gold=case['answer'],output=obj,
                parse_error=error,raw_output=result['raw_output'],scores=score))
    summary=dict(training_steps=64,evaluation_inferences=40,clinical_accuracy_measured=False,arms={})
    for arm in ['base','lora']:
        summary['arms'][arm]={}
        for task in ['vision','symptoms','citation','regression']:
            rows=[r for r in scored if r['arm']==arm and r['task']==task]
            summary['arms'][arm][task]=dict(cases=len(rows),target_exact=sum(r['full_target_exact'] is True for r in rows),
                matched_fields=sum(r['matched_fields'] for r in rows),total_fields=sum(r['total_fields'] for r in rows),
                quote_exact=sum(r['quote_exact'] for r in rows),quote_total=sum(r['quote_total'] for r in rows),
                source_ids_exact=sum(r['source_ids_exact'] for r in rows),source_ids_total=sum(r['source_ids_total'] for r in rows),
                unsupported_visual_assertions=sum(r['visual_unsupported_assertions'] for r in rows),
                task_schema_ok=sum(r['task_schema_ok'] for r in rows),
                json_parse_ok=sum(r['json_parse_ok'] for r in rows),truncated=sum(r['truncated'] for r in rows),
                regression_schema_compatible=sum(r['regression_schema_compatible'] for r in rows),
                literal_sleep_degree_change_cases=sum(bool(r['symptom_degree_risks']) for r in rows),
                invalid_basis_id_cases=sum(bool(r['invalid_basis_ids']) for r in rows))
            if task=='regression':
                summary['arms'][arm][task].update(explanation_present=sum(r['explanation_present'] for r in rows),
                    questions_present=sum(r['questions_present'] for r in rows))
    summary['checkpoint_selection']='One fixed final checkpoint. Improvement decisions require per-task and raw regression review; no automatic global promotion.'
    summary['limitations']='Synthetic extraction/correction targets and two held-out image assets; existing proxy regressions are development cases, not clinical validation.'
    summary['scoring_review']='Exact output target and schema reported separately from semantic field matches. Root fact arrays and root headgear fields are read for semantic comparison without changing original outputs.'
    (OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=scored[0].keys());writer.writeheader();writer.writerows(scored)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
