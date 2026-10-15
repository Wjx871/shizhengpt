"""Post-experiment replay; not a new independent experiment or new inference."""
import hashlib
import json
import pathlib
import shutil

from guarded_consultation_v31 import retain_checked_questions

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evaluation/consultation_facts_20261008'
OUT=ROOT/'evaluation/consultation_facts_v31_replay_20261008'


def main():
    cases=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    retained=full=partial=0
    for case in cases:
        dest=OUT/'outputs'/f"{case['id']}_v31.json"
        if dest.exists():raise RuntimeError('Preserve replay history')
        path=SOURCE/'outputs'/f"{case['id']}_v3.json"
        original=json.loads(path.read_text(encoding='utf-8'))
        result=retain_checked_questions(original)
        assert result['analysis_attempts']==original['analysis_attempts']
        assert result['model_invocations']==original['model_invocations']
        assert result['user_symptoms']==case['symptoms']
        result['replayed_from']=str(path)
        result['source_output_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        result['new_model_inferences_in_replay']=0
        retained+=bool(case['symptoms'] and result['questions'])
        full+=result['analysis_action']=='generated_analysis_requires_review'
        partial+=result['analysis_action']=='questions_only_after_explanation_rejection_requires_review'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(cases=len(cases),symptom_cases=10,full_analysis_cases=full,questions_only_cases=partial,
        cases_with_questions=retained,new_model_inferences=0,
        note='Post-output engineering replay on existing v3 candidates. Not independent validation of v3.1.',
        script_sha256=hashlib.sha256((ROOT/'scripts/guarded_consultation_v31.py').read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['guarded_consultation_v31.py','review_question_fallback.py']:
        target=OUT/'sources'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'scripts'/name,target)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
