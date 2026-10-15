"""Reuse unchanged V2 visual stage; re-run six symptom analyses with question checks."""
import copy
import hashlib
import json
import time
from infer_image import Engine, ROOT
from guarded_consultation import GuardedAssistant, validate_analysis
from evaluate_guarded_consultation import save

PREVIOUS=ROOT/'evaluation/guarded_consultation_v2_20261008'
OUT=ROOT/'evaluation/guarded_consultation_final_20261008'


def main():
    manifest=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    save(OUT/'manifest.json',manifest)
    engine=Engine()
    assistant=GuardedAssistant(infer=engine.infer,max_retries=1)
    calls=0
    started=time.time()
    for case in manifest:
        dest=OUT/'outputs'/f"{case['id']}_guarded.json"
        if dest.exists():raise ValueError('Preserve existing output')
        previous=PREVIOUS/'outputs'/f"{case['id']}_guarded.json"
        result=copy.deepcopy(json.loads(previous.read_text(encoding='utf-8')))
        result['visual_reused_from']=str(previous)
        if case['symptoms']:
            print('START',case['id'],flush=True)
            obj,attempts=assistant.checked_generation(assistant.analysis_prompt(result),None,
                lambda value:validate_analysis(value,case['symptoms']),512)
            result['analysis_attempts']=attempts
            result['possible_explanations']=obj['possible_explanations'] if obj else []
            result['questions']=obj['questions'] if obj else []
            # Replace analysis limitations; retain deterministic input/visual limitations.
            result['limitations']=['照片技术观察未经医学核验；用户症状由程序原样保留。']
            result['limitations'].extend(obj['limitations'] if obj else ['问诊回答未通过检查，未展示为有效分析。'])
            result['analysis_action']='generated_analysis_requires_review' if obj else 'analysis_output_rejected_after_retry'
            calls+=len(attempts)
        result['new_inferences_in_final_run']=len(result['analysis_attempts']) if case['symptoms'] else 0
        result['model_invocations']=len(result['visual_attempts'])+len(result['analysis_attempts'])
        result['model_latency_sec']=sum(a['model_result']['latency_sec'] for stage in ['visual_attempts','analysis_attempts'] for a in result[stage])
        result['requires_review']=bool(result['visual_facts'] or result['possible_explanations'] or result['questions'])
        result['arm']='guarded_final'
        result['evaluation_config']=engine.config
        save(dest,result)
        print('DONE',case['id'],result['analysis_action'],'new_calls',result['new_inferences_in_final_run'],flush=True)
    save(OUT/'run.json',dict(cases=len(manifest),new_model_inferences=calls,elapsed_sec=time.time()-started,
        config=engine.config,visual_stage_reused_from=str(PREVIOUS),
        script_sha256=hashlib.sha256((ROOT/'scripts/guarded_consultation.py').read_bytes()).hexdigest(),
        changed='Question presupposition checks and neutral questioning instructions; six analyses regenerated.'))
    print('COMPLETE',calls,flush=True)


if __name__=='__main__':main()
