"""Bounded abstention workflow, not a calibrated confidence/clinical verifier."""
import re
from consultation_alignment_v4 import fact_ledger, mentions, question_checks, validate_analysis
from consultation_alignment_v41 import run as aligned_run

VERSION='cautious_chat_v1_20261013'


def photo_check(path):
    import numpy as np
    from PIL import Image
    from face_quality import analyze
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'))
    grey=rgb.mean(axis=2)
    reasons=[]
    if grey.mean()<35:reasons.append('too_dark')
    if (grey>=245).mean()>.45:reasons.append('overexposed')
    if min(rgb.shape[:2])<120:reasons.append('resolution_too_low')
    face=analyze(path)
    reasons.extend(face['reject_reasons'])
    return {'image_mean_brightness':round(float(grey.mean()),1),'reasons':sorted(set(reasons)),
            'face_qc':face,'passed':not reasons}


def extract_sources(user_texts):
    # No LLM rewrite. Source strings remain exactly as entered. Bounded lexical
    # recognition misses symptoms; omitted input remains available in the trace.
    texts=list(user_texts)
    correction=None
    for i,text in enumerate(texts):
        if re.search(r'更正|之前说错|刚才说错|纠正一下',text):correction=i
    if correction is not None:texts=texts[correction:]
    result=[]
    for text in texts:
        for clause in re.split(r'[。\n；;]',text):
            if mentions(clause) and not re.match(r'\s*(?:请|能否|如何|什么|是否|有没有|假设|如果)',clause):
                result.append(clause)
    return result


def contradictions(symptoms):
    states={}
    for record in fact_ledger(symptoms):
        for clause in record['clauses']:
            for f in clause['findings']:states.setdefault(f['term'],set()).add(f['polarity'])
    return sorted(k for k,v in states.items() if {'reported','negative'}<=v)


def explanation_checks(item,symptoms):
    reasons=validate_analysis({'possible_explanations':[item],'questions':[],'limitations':[]},symptoms)
    reasons=[r for r in reasons if r!='no_followup_questions']
    if reasons:return reasons
    text=item['hypothesis']
    if not re.search(r'可能|需.*(?:核实|确认|考虑)|待(?:核实|确认|排)|尚不能',text):
        reasons.append('uncertainty_missing')
    if re.search(r'确诊|肯定|一定|必然|就是|证明|可以排除|已经排除|无需(?:检查|就医)',text):
        reasons.append('definitive_claim')
    for term,m in mentions(text):
        cited=[symptoms[int(e['source_id'][1:])-1] for e in item['evidence']]
        source=' '.join(cited)
        if not any(t==term for t,_ in mentions(source)):
            reasons.append('unsupported_symptom_in_hypothesis')
        if term=='sleep_quality' and m.group() not in source:
            reasons.append('sleep_degree_changed')
    # Exact quotation alone is not enough: evidence must contain a recognized,
    # active symptom, rather than a question/instruction or only a normal status.
    for e in item['evidence']:
        matching=mentions(e['quote'])
        if not matching or all(t=='sleep_quality' and re.search(r'良好|正常',m.group()) for t,m in matching):
            reasons.append('no_active_symptom_in_quote')
    return sorted(set(reasons))


def run(infer,texts,image=None):
    latest=texts[-1].strip() if texts else ''
    result={'version':VERSION,'status':'needs_information','trace':{},'answer':'','model_invocations':0}
    sections=[]
    if image:
        qc=photo_check(image);result['trace']['image_qc']=qc
        if not qc['passed']:
            sections.append('**图片无法可靠判断。** 这张图片没有通过清晰度、曝光或面部完整性检查。请在自然光下重新拍摄清晰、完整的正面照片；如果是舌头或其他部位，这版面部核对流程不能可靠分析。')
            result['status']='image_abstained'
        else:
            sections.append('**图片信息尚不能确认。** 基础面部定位和画质检查已通过，但目前没有独立核验肤色、病变和各部位细节的能力，无法可靠判断这些内容，也不能仅凭照片给出诊断。')
            result['status']='visual_details_abstained'
        # Deliberately do not display unchecked free-form model image prose.
        # This narrows output capability; it is not improved visual recognition.
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
        if latest:
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
    result['trace']['aligned_result']=raw;result['model_invocations']=raw['model_invocations']
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
