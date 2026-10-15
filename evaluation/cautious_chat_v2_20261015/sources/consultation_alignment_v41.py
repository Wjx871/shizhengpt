"""Keep independently checked answer components after whole-answer rejection.

Post-v4 development change. Replay is not an independent validation run.
"""
import copy

from consultation_alignment_v4 import fact_ledger, list_strings, question_checks, run_case, validate_analysis


def retain_valid_parts(result):
    result=copy.deepcopy(result)
    result['workflow_version']='v4.1_independent_component_retention'
    result['component_checks']=[]
    if result['analysis_action'] in ('full_answer_requires_review','no_symptoms'):
        return result
    symptoms=result['symptoms_original'];ledger=fact_ledger(symptoms);best=None
    for index,attempt in enumerate(result['analysis_attempts']):
        obj=attempt['parsed']
        if attempt['model_result'].get('truncated') or not isinstance(obj,dict) or set(obj)!={'possible_explanations','questions','limitations'}:
            continue
        explanations=[];questions=[];checks={'attempt':index+1,'explanations':[],'questions':[]}
        if isinstance(obj['possible_explanations'],list) and len(obj['possible_explanations'])<=2:
            for item_index,item in enumerate(obj['possible_explanations']):
                reasons=validate_analysis({'possible_explanations':[item],'questions':[],'limitations':[]},symptoms)
                reasons=[r for r in reasons if r!='no_followup_questions']
                checks['explanations'].append({'item':item_index+1,'reasons':reasons})
                if not reasons:explanations.append(item)
        if list_strings(obj['questions'],3):
            for question_index,q in enumerate(obj['questions']):
                reasons=question_checks(q,ledger)
                checks['questions'].append({'item':question_index+1,'reasons':reasons})
                if not reasons:questions.append(q)
        result['component_checks'].append(checks)
        # Retain one candidate's components, never splice conflicting answers
        # from separate attempts. Prefer explanations, then more checked questions.
        rank=(len(explanations),len(questions))
        if (explanations or questions) and (best is None or rank>best['rank']):
            best={'rank':rank,'explanations':explanations,'questions':questions,'attempt':index+1}
    if best:
        result['analysis']={'possible_explanations':best['explanations'],'questions':best['questions'],
                            'limitations':['完整候选未通过检查；此处仅保留逐项通过有限核对的内容，医学解释仍需复核。']}
        result['analysis_action']='partial_checked_content_requires_review'
        result['selected_component_attempt']=best['attempt']
    return result


def run(infer,case,qc=None):
    return retain_valid_parts(run_case(infer,case,enhanced=True,qc=qc))
