"""Evidence-first engineering prototype; no weight changes or clinical verifier."""
import copy
import json
import re

from consultation_facts import extract_facts, mentions, question_risks
from guarded_consultation import parse_json, forbidden_visual

ANALYSIS_SCHEMA = '''只输出JSON对象：
{"possible_explanations":[{"hypothesis":"需要进一步核实的可能解释","evidence":[{"source_id":"S1","quote":"逐字引用原文中的依据"}],"missing_info":["需要补充的信息"]}],"questions":["追问"],"limitations":["局限"]}。
最多2个解释，每个最多2条证据和2条缺失信息，最多3个追问和2条局限。示例是格式，不是答案。
解释不是确诊，不提供药物剂量。没有充分信息时解释可为空，但有症状时应提出有用问题。'''
VISUAL_SCHEMA = '''只输出JSON对象：{"visual_status":{"eyes":"visible/not_visible/unknown","nose":"visible/not_visible/unknown","mouth":"visible/not_visible/unknown"},"observations":["简短观察"],"limitations":["局限"]}。
每项状态只选visible、not_visible或unknown，最多2条观察和2条局限。只观察照片，不做问诊和诊断。'''


def fact_ledger(symptoms):
    records = copy.deepcopy(extract_facts(symptoms))
    for record in records:
        previous = []
        for clause in record['clauses']:
            if clause['findings']:
                previous = clause['findings']
            elif len(previous) == 1 and re.fullmatch(r'(?:今天|现在|目前)?(?:已|已经)缓解', clause['text'].strip()):
                # Resolve only one unambiguous preceding mention in the same source.
                previous[0]['polarity'] = 'resolved'
                previous[0]['state_evidence'] = {k: clause[k] for k in ['text', 'start', 'end']}
    return records


def question_checks(question, records):
    reasons = question_risks(question, records)
    for term, match in mentions(question):
        prefix = question[:match.start()]
        if re.search(r'是否|有没有|有无|如果|若', prefix):
            continue
        sources = [r for r in records if any(f['term']==term for c in r['clauses'] for f in c['findings'])]
        if term == 'sleep_quality' and sources:
            if match.group() not in ''.join(r['raw'] for r in sources):
                reasons.append('sleep_quality_presupposition_changed')
        states = {f['polarity'] for r in sources for c in r['clauses'] for f in c['findings'] if f['term']==term}
        if states and 'reported' not in states and re.search(r'多久|持续|加重|原因|程度', question[match.start():]):
            reasons.append('inactive_symptom_presupposed')
    return sorted(set(reasons))


def list_strings(value, maximum):
    return isinstance(value, list) and len(value)<=maximum and all(isinstance(s,str) and s.strip() for s in value)


def validate_analysis(obj, symptoms):
    if not isinstance(obj,dict) or set(obj)!={'possible_explanations','questions','limitations'}:
        return ['analysis_schema_keys']
    if not list_strings(obj['questions'],3) or not list_strings(obj['limitations'],2) or not isinstance(obj['possible_explanations'],list) or len(obj['possible_explanations'])>2:
        return ['analysis_schema_types']
    reasons=[]; ledger=fact_ledger(symptoms); sources={r['id']:r for r in ledger}
    if symptoms and not obj['questions']:
        reasons.append('no_followup_questions')
    for item in obj['possible_explanations']:
        if not isinstance(item,dict) or set(item)!={'hypothesis','evidence','missing_info'} or not isinstance(item['hypothesis'],str) or not item['hypothesis'].strip() or not list_strings(item['missing_info'],2):
            reasons.append('explanation_schema');continue
        if not isinstance(item['evidence'],list) or not 1<=len(item['evidence'])<=2:
            reasons.append('missing_evidence');continue
        for evidence in item['evidence']:
            if not isinstance(evidence,dict) or set(evidence)!={'source_id','quote'}:
                reasons.append('evidence_schema');continue
            record=sources.get(evidence['source_id']) if isinstance(evidence['source_id'],str) else None
            quote=evidence['quote']
            if record is None or not isinstance(quote,str) or not quote.strip() or quote not in record['raw']:
                reasons.append('evidence_quote_not_in_source');continue
            findings=[f for c in record['clauses'] for f in c['findings'] if f['evidence']['text'] in quote]
            if findings and all(f['polarity']!='reported' for f in findings):
                reasons.append('inactive_symptom_used_as_positive_basis')
        # A bounded check: hypotheses may name conditions, but must not intensify
        # known symptoms. Unknown accompanying symptoms belong in missing_info.
        for changed in ('睡眠质量差','睡眠质量一般','严重疲劳','严重乏力','长期食欲下降','持续腹胀'):
            if changed in item['hypothesis'] and changed not in ''.join(symptoms):
                reasons.append('hypothesis_changes_known_symptom')
    for question in obj['questions']:
        reasons.extend(question_checks(question,ledger))
    return sorted(set(reasons))


def validate_visual(obj):
    if not isinstance(obj,dict) or set(obj)!={'visual_status','observations','limitations'}:
        return ['visual_schema_keys']
    status=obj['visual_status']
    if not isinstance(status,dict) or set(status)!={'eyes','nose','mouth'} or any(v not in ('visible','not_visible','unknown') for v in status.values()) or not list_strings(obj['observations'],2) or not list_strings(obj['limitations'],2):
        return ['visual_schema_types']
    text=' '.join(obj['observations'])
    reasons=forbidden_visual(text)
    if re.search(r'无(?:明显)?(?:皮疹|肿胀|畸形)|正常|健康|皮肤细纹|毛孔',text):
        reasons.append('unverified_clinical_or_fine_detail_claim')
    # No claim that absence was independently visually verified.
    return reasons


def prompts(symptoms, enhanced=False):
    base='这是研究用合成问诊输入。根据以下用户信息给出可能解释和追问。\n'+ANALYSIS_SCHEMA
    base+='\n原始症状：'+json.dumps({f'S{i+1}':s for i,s in enumerate(symptoms)},ensure_ascii=False)
    if enhanced:
        base+='\n先核对事实：'+json.dumps(fact_ledger(symptoms),ensure_ascii=False)
        base+='\n症状原文最权威；保持程度、频率、时间与否定。已缓解不适不能当作当前患病依据。证据quote逐字引用原文，不改写、不编造，source_id必须对应此原文。'
        base+='\n未知或可能伴随症状只能写进missing_info或以“是否有……”追问，不能在hypothesis中认定已存在。不把睡眠良好或一般说成差。历史不适可询问是否再次出现，不能预设仍持续。'
    return base


def checked(infer,prompt,image,validator,retries=1):
    attempts=[]
    for number in range(retries+1):
        request=prompt
        if attempts:
            request+='\n上次未通过有限检查，原因：'+json.dumps(attempts[-1]['reasons'],ensure_ascii=False)+'。请重新输出完整JSON，纠正上述问题。'
        raw=infer(request,image)
        obj,error=parse_json(raw['raw_output'])
        reasons=[error] if error else validator(obj)
        if raw.get('truncated'):reasons.append('truncated')
        attempts.append({'model_result':raw,'parsed':obj,'reasons':reasons})
        if not reasons:return obj,attempts
    return None,attempts


def run_case(infer,case,enhanced=False,qc=None):
    result={'symptoms_original':case['symptoms'],'visual':None,'analysis':None,'visual_attempts':[],'analysis_attempts':[],
            'visual_action':'no_image','analysis_action':'no_symptoms','clinical_verification':False}
    image=case.get('image')
    if image:
        result['qc']=qc(image) if enhanced else None
        if enhanced and not result['qc']['usable_for_technical_description']:
            result['visual_action']='qc_abstention'
        else:
            prompt=VISUAL_SCHEMA
            if enhanced:
                prompt+='\n图片仅通过基础质控；不要猜测未显示部位，不判断无异常，不解释真实肤色。分辨率不足的细节明确写无法判断。'
            obj,attempts=checked(infer,prompt,image,validate_visual if enhanced else lambda _:[],retries=1 if enhanced else 0)
            result['visual']=obj;result['visual_attempts']=attempts
            result['visual_action']='limited_checks_passed_unverified' if obj else 'output_abstention'
    if case['symptoms']:
        obj,attempts=checked(infer,prompts(case['symptoms'],enhanced),None,
                             lambda obj:validate_analysis(obj,case['symptoms']) if enhanced else [],retries=1 if enhanced else 0)
        result['analysis']=obj;result['analysis_attempts']=attempts
        result['analysis_action']='full_answer_requires_review' if obj else 'full_answer_abstention'
        if enhanced and obj is None:
            for candidate in reversed(attempts):
                obj=candidate['parsed']
                if not isinstance(obj,dict) or candidate['model_result'].get('truncated'):continue
                questions=obj.get('questions')
                if list_strings(questions,3) and questions and not any(question_checks(q,fact_ledger(case['symptoms'])) for q in questions):
                    result['analysis']={'possible_explanations':[],'questions':questions,'limitations':['完整解释未通过有限检查，仅保留经过单独检查的追问。']}
                    result['analysis_action']='questions_only_requires_review';break
    attempts=result['visual_attempts']+result['analysis_attempts']
    result['model_invocations']=len(attempts)
    result['model_latency_sec']=sum(a['model_result']['latency_sec'] for a in attempts)
    return result
