"""Re-score preserved candidates after correcting a question-rule false positive."""
import copy
import hashlib
import json
from guarded_consultation import ROOT, parse_json, validate_analysis, validate_visual
from evaluate_guarded_consultation import save

OUT=ROOT/'evaluation/guarded_consultation_final_20261008'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    accepted=0
    for case in manifest:
        path=OUT/'outputs'/f"{case['id']}_guarded.json"
        backup=OUT/'outputs'/f"{case['id']}_guarded_pre_rule_review.json"
        if not backup.exists():save(backup,json.loads(path.read_text(encoding='utf-8')))
        result=copy.deepcopy(json.loads(backup.read_text(encoding='utf-8')))
        assert result['user_symptoms']==case['symptoms']
        if result['visual_facts']:
            assert not validate_visual(dict(visual_facts=result['visual_facts'],limitations=[]))
        if case['symptoms']:
            selected=None
            selected_index=None
            for index,attempt in enumerate(result['analysis_attempts']):
                attempt['prior_guard_reasons']=attempt['reasons']
                obj,error=parse_json(attempt['model_result']['raw_output'])
                reasons=[error] if error else validate_analysis(obj,case['symptoms'])
                if attempt['model_result'].get('truncated'):reasons.append('output_truncated')
                attempt['reasons']=reasons
                if not reasons:selected,selected_index=obj,index+1
            result['possible_explanations']=selected['possible_explanations'] if selected else []
            result['questions']=selected['questions'] if selected else []
            result['limitations']=['照片技术描述未经医学核验；症状原文由程序保留。']
            if not result['image_present']:result['limitations'].append('没有图片，未进行视觉分析。')
            elif not result['cv_facts']['usable_for_technical_description']:result['limitations'].append('图片质控拒绝，未进行面部观察。')
            result['limitations'].extend(selected['limitations'] if selected else ['问诊回答未通过检查，未展示为有效分析。'])
            result['analysis_action']='generated_analysis_requires_review' if selected else 'analysis_output_rejected_after_retry'
            result['selected_analysis_attempt']=selected_index
            result['analysis_revalidated_without_new_generation']=True
            accepted+=bool(selected)
        result['requires_review']=bool(result['visual_facts'] or result['possible_explanations'] or result['questions'])
        save(path,result)
    save(OUT/'rule_review.json',dict(symptom_cases=6,accepted_analysis_cases=accepted,new_model_inferences=0,
        reason='Unknown symptoms may be asked as questions; only presupposed changed sleep severity is blocked.',
        script_sha256=hashlib.sha256((ROOT/'scripts/guarded_consultation.py').read_bytes()).hexdigest(),
        backups_retained=True))
    print('RECHECK_COMPLETE',accepted,'of 6 symptom analyses; no new model generations')


if __name__=='__main__':main()
