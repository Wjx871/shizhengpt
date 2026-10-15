"""Fact and question checks extending preserved v2. No learned visual verifier."""
import argparse
import json
import pathlib
import re

from guarded_consultation import GuardedAssistant as V2, validate_analysis, validate_visual
from consultation_facts import extract_facts, explanation_risks, question_risks


def validate_fact_analysis(obj, symptoms):
    reasons = validate_analysis(obj, symptoms)
    # Only inspect well-shaped objects; preserve v2 schema errors without exceptions.
    if any('schema' in r or 'type' in r or 'keys' in r for r in reasons):
        return reasons
    records = extract_facts(symptoms)
    for item in obj['possible_explanations']:
        reasons.extend(explanation_risks(item, records))
    for question in obj['questions']:
        reasons.extend(question_risks(question, records))
    return sorted(set(reasons))


def validate_scoped_visual(obj, low_resolution=False):
    reasons = validate_visual(obj)
    if any('schema' in r for r in reasons):
        return reasons
    for text in obj['visual_facts']:
        if re.search(r'头盔|宇航服|服装|衣服|职业', text):
            reasons.append('outside_face_description_scope')
        if re.search(r'皮疹|肿胀|畸形|病变|正常|健康|异常色泽', text):
            reasons.append('clinical_or_normality_claim_unverified')
        if low_resolution and re.search(r'纹理|毛孔|细纹|色斑|斑点|干裂|皮损', text):
            reasons.append('fine_detail_claim_low_resolution')
    return sorted(set(reasons))


class FactGuardedAssistant(V2):
    def run(self, symptoms, image=None):
        self._symptoms = list(symptoms)
        result = super().run(symptoms, image)
        result['guard_version'] = 'v3_fact_slots_and_question_prefix'
        result['symptom_facts'] = extract_facts(symptoms)
        result['fact_extraction_is_complete'] = False
        result['visual_verification_method'] = 'scope_and_resolution_rules_only'
        result['limitations'].append('事实槽位仅覆盖有限词表；图像断言未经过独立视觉核验。')
        return result

    def analysis_prompt(self, result):
        # Exact source spans are advisory metadata; source text remains authoritative.
        prompt = super().analysis_prompt(result)
        prompt += '\n有限词表提取的事实槽位（遗漏处以症状原文为准）：'+json.dumps(extract_facts(self._symptoms), ensure_ascii=False)
        prompt += '\n只解释用户明确报告的当前症状，否定或已缓解的症状不能作为现有患病依据。不重复转述症状。'
        prompt += '\n优先开放追问，例如“睡眠质量一般具体指什么？”；未知症状用“是否有……”询问，不能问“胸痛是否持续”或“发热多久”来预设它存在。解释不要补充原文没有的严重程度、频率和病程。'
        return prompt

    def checked_generation(self, prompt, image, validator, cap):
        if image:
            # v2 already appends a deterministic low-resolution warning to the prompt.
            check = lambda obj: validate_scoped_visual(obj, '分辨率偏低' in prompt)
            prompt += '\n只描述面部粗略可见轮廓、朝向或遮挡，不描述头盔服装，不判断无皮疹、无肿胀、无畸形或正常。看不清的细节放入limitations并写“无法判断”。'
        else:
            check = lambda obj: validate_fact_analysis(obj, self._symptoms)
        return super().checked_generation(prompt, image, check, cap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--symptoms-json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    symptoms = json.loads(pathlib.Path(args.symptoms_json).read_text(encoding='utf-8')) if args.symptoms_json else []
    result = FactGuardedAssistant(max_retries=args.max_retries).run(symptoms, args.image)
    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(dest), visual_action=result['visual_action'],
                          analysis_action=result['analysis_action'], model_invocations=result['model_invocations']), ensure_ascii=False))


if __name__ == '__main__':
    main()
