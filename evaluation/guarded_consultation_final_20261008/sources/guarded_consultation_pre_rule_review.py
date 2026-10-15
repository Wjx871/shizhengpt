"""Prototype guardrails for face + text input. Checks are not medical verification."""
import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
VISUAL_PROMPT = '''只描述本次照片直接可见的面部技术信息，不做中医辨证。照片通过基础质控，不代表所有部位都可见。
不得猜测年龄、性别、职业、疾病、证候、健康状态、舌象、脉象、身体活动或未提供的症状；不得补全未显示的部位。
颜色仅描述照片，不代表真实肤色。分辨率不足时避免细微皮肤判断。不复述程序数值。
只输出JSON，两个键：visual_facts（简短字符串列表，最多3项），limitations（字符串列表，最多3项）。看不清就明确说明。'''
ANALYSIS_PROMPT = '''依据输入中S编号对应的用户原始症状，给出需要进一步核实的可能解释及追问。图片观察仅是未核验的技术描述，禁止据此做面色辨证。
不得将可能解释写成确诊。不得新增或改写用户症状、程度、频率、否定或持续时间。不要重复转述用户症状，使用S编号引用依据。
追问也不能预设未提供的事实。例如用户说睡眠质量一般，不应问睡眠质量差具体表现是什么，应中性询问睡眠质量一般具体指什么。
不要新增视觉事实，不要猜测年龄、性别、脉象或舌象。不要提供药物、剂量或治疗方案。
只输出JSON，三个键：possible_explanations（最多3个对象，每个对象有basis_ids字符串列表及explanation字符串），questions（最多5个字符串），limitations（最多3个字符串）。
basis_ids只能使用输入中实际存在的S编号，不允许以照片颜色作为辨证依据。questions可以询问尚不清楚的症状，不能把问题当成既有事实。信息不足时可以不给出可能解释，但仍应提出有用问题。'''


def parse_json(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        return json.loads(text), None
    except (ValueError, TypeError) as ex:
        return None, 'invalid_json: '+str(ex)


def string_list(value, maximum):
    return isinstance(value, list) and len(value) <= maximum and all(isinstance(v, str) and v.strip() for v in value)


def forbidden_visual(text):
    rules = {
        'unsupported_demographics': r'\d+\s*岁|中年|老年|青年|男性|女性|男人|女人|职业|宇航员|年龄',
        'unsupported_tongue_pulse_motion': r'舌象|舌苔|舌质|脉象|脉搏|活动正常|运动正常',
        'medical_inference_from_photo': r'脾虚|气虚|阳虚|阴虚|血虚|肝郁|血瘀|湿热|寒凝|病理|病变|疾病|健康状态|气血不足',
        'unsupported_body_region': r'手部|手掌|指甲|腹部|胸部|腿部|足部|背部',
    }
    return [name for name, pattern in rules.items() if re.search(pattern, text)]


def symptom_rewrite_risks(text, symptoms):
    joined = '\n'.join(symptoms)
    rules = []
    if any(s in joined for s in ['睡眠质量一般','睡眠尚可','睡眠良好']) and re.search(r'睡眠(?:质量)?(?:差|较差|不好|不佳)|睡眠浅|失眠|多梦', text):
        rules.append('sleep_severity_changed_or_added')
    if '偶尔食欲不佳' in joined and re.search(r'长期食欲|食欲(?:很差|极差|明显下降)|食欲减退严重', text):
        rules.append('appetite_frequency_or_severity_changed')
    if '没有发热' in joined or '无发热' in joined:
        if re.search(r'(?:有|出现|伴|存在)发热', text):
            rules.append('fever_negation_changed')
    if '没有胸痛' in joined or '无胸痛' in joined:
        if re.search(r'(?:有|出现|伴|存在)胸痛', text):
            rules.append('chest_pain_negation_changed')
    return rules


def validate_visual(obj):
    if not isinstance(obj, dict) or set(obj) != {'visual_facts', 'limitations'}:
        return ['visual_schema_keys']
    if not string_list(obj['visual_facts'], 3) or not string_list(obj['limitations'], 3):
        return ['visual_schema_types']
    return forbidden_visual(' '.join(obj['visual_facts']))


def validate_analysis(obj, symptoms):
    if not isinstance(obj, dict) or set(obj) != {'possible_explanations', 'questions', 'limitations'}:
        return ['analysis_schema_keys']
    if not isinstance(obj['possible_explanations'], list) or len(obj['possible_explanations']) > 3:
        return ['explanation_schema']
    if not string_list(obj['questions'], 5) or not string_list(obj['limitations'], 3):
        return ['analysis_list_types']
    allowed = {f'S{i+1}' for i in range(len(symptoms))}
    reasons = []
    for item in obj['possible_explanations']:
        if not isinstance(item, dict) or set(item) != {'basis_ids','explanation'}:
            return ['explanation_item_keys']
        if not string_list(item['basis_ids'], len(allowed)) or not item['basis_ids'] or not set(item['basis_ids']) <= allowed:
            reasons.append('unknown_or_missing_symptom_basis')
        if not isinstance(item['explanation'], str) or not item['explanation'].strip():
            return ['explanation_text_type']
        text = item['explanation']
        if text == '需要核实的可能解释':
            reasons.append('unfilled_template_placeholder')
        reasons += symptom_rewrite_risks(text, symptoms)
        if re.search(r'确诊|可以确定|肯定是|患有|你有|患者有|患者出现|患者表现|症状包括', text):
            reasons.append('unsupported_patient_fact_or_certainty')
        if re.search(r'面色|肤色|唇色|舌象|舌苔|脉象|照片|额头|脸颊|鼻部|眼周|\d+\s*岁|男性|女性', text):
            reasons.append('new_visual_or_demographic_claim_in_analysis')
        if not re.search(r'可能|需|待|不足|无法|不能|因素', text):
            reasons.append('explanation_not_marked_uncertain')
    if not obj['questions'] and not obj['possible_explanations']:
        reasons.append('empty_analysis')
    if not obj['questions']:
        reasons.append('missing_followup_questions')
    for question in obj['questions']:
        explicit_conditional = bool(re.search(r'是否|有没有|有无|会不会|吗|是不是|假如|如果|若', question))
        if not explicit_conditional:
            reasons += ['question_presupposes_'+name for name in symptom_rewrite_risks(question, symptoms)]
            joined=' '.join(symptoms)
            for term in ['发热','胸痛']:
                if ('没有'+term in joined or '无'+term in joined) and re.search(term+r'(?:持续|多久|几天)', question):
                    reasons.append('question_reverses_negative_'+term)
    return sorted(set(reasons))


class GuardedAssistant:
    def __init__(self, infer=None, max_retries=1):
        if not 0 <= max_retries <= 2:
            raise ValueError('max_retries must be 0..2')
        self._infer = infer
        self._engine = None
        self.max_retries = max_retries

    def infer(self, prompt, image=None, max_new_tokens=512):
        if self._infer is not None:
            return self._infer(prompt, image, max_new_tokens=max_new_tokens)
        if self._engine is None:
            from infer_image import Engine
            self._engine = Engine()
        return self._engine.infer(prompt, image, max_new_tokens=max_new_tokens)

    def checked_generation(self, prompt, image, validator, cap):
        attempts = []
        for number in range(self.max_retries+1):
            request = prompt
            if attempts:
                request += '\n上次输出未通过检查，原因：'+json.dumps(attempts[-1]['reasons'], ensure_ascii=False)
                request += '\n请重新回答，严格使用规定JSON字段和类型。不要补充任何输入未提供的事实。'
            raw = self.infer(request, image, max_new_tokens=cap)
            obj, error = parse_json(raw['raw_output'])
            reasons = [error] if error else validator(obj)
            if raw.get('truncated'):
                reasons.append('output_truncated')
            attempts.append(dict(attempt=number+1, reasons=reasons, model_result=raw))
            if not reasons:
                return obj, attempts
        return None, attempts

    @staticmethod
    def analysis_prompt(result):
        metadata = '\n用户原始症状（仅这些内容为已提供事实）：\n'+ '\n'.join(f'{ident}：{text}' for ident,text in result['source_symptoms'].items())
        metadata += '\n图片处理状态：'+result['visual_action']
        if result['visual_facts']:
            metadata += '\n未核验的照片技术观察，仅供说明照片条件，不作为证候依据：'+'；'.join(result['visual_facts'])
        metadata += '\n请输出实际分析，格式为：{"possible_explanations":[{"basis_ids":["S1"],"explanation":"需要核实的可能解释"}],"questions":[],"limitations":[]}。示例是格式说明，不是答案；请根据原始症状填写，提出有用追问。'
        return ANALYSIS_PROMPT+metadata

    def run(self, symptoms, image=None):
        if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
            raise ValueError('symptoms must be a list of exact strings')
        from face_quality import analyze
        original = list(symptoms)
        facts = analyze(image)
        result = dict(user_symptoms=original, image_present=bool(image), cv_facts=facts,
            medical_status='unknown', model_description_verified=False,
            guard_checks_are_medical_verification=False, visual_attempts=[], analysis_attempts=[],
            visual_facts=[], possible_explanations=[], questions=[], limitations=[],
            source_symptoms={f'S{i+1}': s for i,s in enumerate(original)})
        if not image:
            result['visual_action'] = 'no_image_visual_generation_skipped'
            result['limitations'].append('未提供图片，未进行视觉分析。')
        elif not facts['usable_for_technical_description']:
            result['visual_action'] = 'qc_rejected_visual_generation_skipped'
            result['limitations'].append('图片未通过面部质控，未生成面部观察；有症状时仍可做文字追问。')
        else:
            metadata = '\n本次已提供照片。图片左右仅指画面左右，不能确认人物解剖左右。照片颜色没有校准。'
            if 'low_resolution_for_fine_facial_features' in facts['warnings']:
                metadata += '面部区域分辨率偏低，应避免细节断言。'
            metadata += '\n请填写实际观察结果，输出格式为：{"visual_facts":[],"limitations":[]}。不要复述输入状态。'
            obj, attempts = self.checked_generation(VISUAL_PROMPT+metadata,
                                                    str(image), validate_visual, 256)
            result['visual_attempts'] = attempts
            if obj:
                result['visual_action'] = 'technical_description_passed_limited_checks_unverified'
                result['visual_facts'] = obj['visual_facts']
                result['limitations'].extend(obj['limitations'])
            else:
                result['visual_action'] = 'visual_output_rejected_after_retry'
                result['limitations'].append('图片回答未通过格式或已知风险检查，未展示该回答。')
        if original:
            obj, attempts = self.checked_generation(self.analysis_prompt(result),
                None, lambda value: validate_analysis(value, original), 512)
            result['analysis_attempts'] = attempts
            if obj:
                result['analysis_action'] = 'generated_analysis_requires_review'
                result['possible_explanations'] = obj['possible_explanations']
                result['questions'] = obj['questions']
                result['limitations'].extend(obj['limitations'])
            else:
                result['analysis_action'] = 'analysis_output_rejected_after_retry'
                result['limitations'].append('模型分析未通过检查，保留症状原文并建议补充信息。')
        else:
            result['analysis_action'] = 'no_symptoms_analysis_skipped'
        result['model_invocations'] = len(result['visual_attempts'])+len(result['analysis_attempts'])
        result['model_latency_sec'] = sum(a['model_result']['latency_sec'] for stage in ['visual_attempts','analysis_attempts'] for a in result[stage])
        result['requires_review'] = bool(result['visual_facts'] or result['possible_explanations'] or result['questions'])
        assert result['user_symptoms'] == original
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--symptoms-json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-retries',type=int,default=1)
    args = parser.parse_args()
    symptoms = json.loads(pathlib.Path(args.symptoms_json).read_text(encoding='utf-8')) if args.symptoms_json else []
    result = GuardedAssistant(max_retries=args.max_retries).run(symptoms, args.image)
    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(dest),'visual_action':result['visual_action'],
                      'analysis_action':result['analysis_action'],'model_invocations':result['model_invocations']},ensure_ascii=False))


if __name__ == '__main__':
    main()
