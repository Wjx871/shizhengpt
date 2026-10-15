"""V3 visible image observations and preserved V2 text routing.

The main workflow below is copied from v1 to preserve historical experiments.
No learned confidence or clinical correctness guarantee is added.
"""
import re
from consultation_alignment_v4 import fact_ledger, mentions, validate_analysis
from consultation_alignment_v41 import run as aligned_run
from cautious_chat import (
    contradictions, extract_sources as v1_extract_sources,
    explanation_checks as v1_explanation_checks,
    question_checks as v1_question_checks,
)

from visual_observation_v3 import observe

VERSION='cautious_chat_v3_20261015'


def extract_sources(user_texts):
    normalized=[]
    for text in user_texts:
        clauses=[]
        for clause in re.split(r'[。\n；;]',text):
            hits=mentions(clause)
            # A hypothetical before the first symptom is not a patient fact.
            if hits and re.search(r'如果|假设|假如|要是',clause[:min(m.start() for _,m in hits)]):
                continue
            prefix=re.match(r'^\s*(?:你好[，,！! ]*)?(?:请问|请帮我(?:分析|看看|判断)|麻烦(?:帮我)?(?:看一下|看看|分析一下)?)[，,：: ]*',clause)
            if prefix and re.match(r'(?:我|本人)(?:最近|近|这|今天|昨天|有|没有|感觉|经常|偶尔)',clause[prefix.end():]):
                clause=clause[prefix.end():]  # exact substring, not a paraphrase
            clauses.append(clause)
        normalized.append('。'.join(clauses))
    result=v1_extract_sources(normalized)
    assert all(any(s in t for t in user_texts) for s in result)
    return result


def question_checks(question,records):
    reasons=v1_question_checks(question,records)
    if re.search(r'既然|因为你|你(?:已确诊|患有|属于)|肯定|一定是|无需就医|建议.*(?:服用|用药)|每天.*(?:毫克|mg)',question,re.I):
        reasons.append('unsupported_assertion_or_treatment_in_question')
    if not re.search(r'[？?]|是否|有没有|有无|什么|哪些|哪里|多久|何时|如何|怎样|多少|情况|具体表现',question):
        reasons.append('not_a_clear_followup_question')
    return sorted(set(reasons))


def explanation_checks(item,symptoms):
    return v1_explanation_checks(item,symptoms)


def medication_request(text):
    return bool(re.search(r'剂量|用量|处方|吃什么药|药.*(?:作用|禁忌|怎么用)|怎么服',text)
                and re.search(r'多少|标准|请|怎么|如何|能否|建议|[？?]|呢',text))


def run(infer,texts,image=None):
    latest=texts[-1].strip() if texts else ''
    result={'version':VERSION,'status':'needs_information','trace':{},'answer':'','model_invocations':0}
    sections=[]
    if image:
        photo_text,result['status'],visual_trace,calls=observe(infer,image)
        result['trace'].update(visual_trace)
        result['model_invocations']+=calls
        sections.append(photo_text)
    result['trace']['original_user_texts']=list(texts)
    if medication_request(latest):
        sections.append('**这个用药问题目前无法可靠回答。** 当前没有经过核查的药品说明书或指南支持此问题；不能根据先前的症状猜测药物身份、剂量或处方。请核实完整药名并向医生或药师确认。')
        result['status']='knowledge_abstained'
        result['answer']='\n\n'.join(sections);return result
    if re.search(r'问诊.*(?:提供|准备)|(?:提供|准备).*信息',latest) and not mentions(latest):
        sections.append('可以先提供：主要不适、开始时间、严重程度、出现频率、诱因及缓解因素，再补充既往病史、用药和已有检查结果。不确定的信息可以直接说不清楚。')
        result['status']='informational'
        result['answer']='\n\n'.join(sections);return result
    # Recall exactly user-authored test identifiers, without model invention.
    recall=bool(re.search(r'编号.*(?:什么|多少)|(?:什么|多少).*编号',latest))
    identifiers=re.findall(r'(?:测试编号|编号)(?:是|为|：|:)[ \t]*([^。\n，,？！?]+)','\n'.join(texts[:-1] if recall else texts))
    if identifiers and recall:
        sections.append('你提供的编号是：'+identifiers[-1]+'。');result['status']='source_recall'
        result['answer']='\n\n'.join(sections);return result
    if identifiers and re.search(r'(?:测试编号|编号)(?:是|为|：|:)',latest):
        sections.append('已记录你提供的编号：'+identifiers[-1]+'。');result['status']='source_recall'
        result['answer']='\n\n'.join(sections);return result
    if re.fullmatch(r'(你好|您好|谢谢|谢谢你)[！!。\s]*',latest):
        sections.append('你好，可以告诉我你的不适和持续时间；信息不够时，我会先询问。');result['status']='informational'
        result['answer']='\n\n'.join(sections);return result
    symptoms=extract_sources(texts);result['trace']['symptom_sources']=symptoms
    if not symptoms:
        if re.search(r'我.*(?:不舒服|不适|难受)',latest) and not re.search(r'如果|假设|假如|要是',latest):
            sections.append('**需要进一步说明症状。** 我还不能准确识别你说的不适，暂时无法判断原因。具体是哪个部位、什么感觉、从什么时候开始、是否影响日常活动？')
            result['status']='needs_clarification'
        elif latest and not image:
            sections.append('**目前无法可靠回答。** 当前没有识别到明确的症状事实，也没有可核查的资料支持这个问题的结论。请补充具体情况或提供可靠资料；我不会据此猜测病因、证候或照片中不可见的内容。')
        elif not image:
            sections.append('请提供具体问题或症状。')
        result['answer']='\n\n'.join(sections);return result
    conflicts=contradictions(symptoms)
    if conflicts:
        sections.append('**需要先核实信息。** 你提供的症状中有同一不适既报告存在、又报告不存在的情况。请说明目前以哪一条为准，以及是否发生过变化；核实前无法据此分析。')
        result['trace']['contradictions']=conflicts;result['status']='contradiction_abstained'
        result['answer']='\n\n'.join(sections);return result
    # Questions such as "那怎么办" can use preceding verbatim facts. New
    # unrelated requests must not be answered using unrelated old symptoms.
    if latest and not mentions(latest) and not re.search(r'那|这些|这个|继续|补充|分析|原因|情况|怎么办|判断|解释|描述',latest):
        sections.append('**这个问题目前无法可靠回答。** 当前流程只支持基于症状原文的有限问诊核对，没有可核查资料支持该问题。')
        result['answer']='\n\n'.join(sections);return result
    raw=aligned_run(infer,{'symptoms':symptoms,'image':None})
    result['trace']['aligned_result']=raw;result['model_invocations']+=raw['model_invocations']
    analysis=raw.get('analysis') or {};kept=[];dropped=[]
    for item in analysis.get('possible_explanations',[]):
        reasons=explanation_checks(item,symptoms)
        if reasons:dropped.append({'item':item,'reasons':reasons})
        else:kept.append(item)
    result['trace']['additional_rejections']=dropped
    questions=[q for q in analysis.get('questions',[]) if not question_checks(q,fact_ledger(symptoms))]
    sections.append('**你提供的信息**\n'+'\n'.join('- '+s for s in symptoms))
    if kept:
        lines=['**待核实的可能方向**']
        for item in kept:
            lines.append('- '+item['hypothesis'])
            lines.extend('  - 依据：'+e['source_id']+'「'+e['quote']+'」' for e in item['evidence'])
        lines.append('这些方向尚未经过医学核验，不能据此确定病因或证候。')
        sections.append('\n'.join(lines));result['status']='limited_answer'
    else:
        sections.append('**目前无法可靠确定病因或证候。** 现有信息不足，或候选解释没有通过依据和不确定性检查。')
        result['status']='explanation_abstained'
    if questions:sections.append('**还需要补充**\n'+'\n'.join(f'{i+1}. {q}' for i,q in enumerate(questions)))
    else:sections.append('请补充不适开始时间、是否影响日常活动，以及既往病史和用药情况。')
    result['answer']='\n\n'.join(sections)
    return result
