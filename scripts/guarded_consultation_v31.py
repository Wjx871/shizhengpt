"""Post-v3 availability fix: independently check questions before partial fallback.

Frozen v3 prompts and rules remain unchanged. Explanation failures are retained
in raw records and never promoted to accepted explanations by this fallback.
"""
import argparse
import copy
import json
import pathlib

from guarded_consultation import parse_json
from guarded_consultation_v3 import FactGuardedAssistant, validate_fact_analysis


def retain_checked_questions(result):
    result = copy.deepcopy(result)
    result['guard_version'] = 'v3.1_checked_question_fallback'
    result['partial_fallback_checks'] = []
    if result['analysis_action'] != 'analysis_output_rejected_after_retry':
        return result
    for index in range(len(result['analysis_attempts'])-1, -1, -1):
        attempt = result['analysis_attempts'][index]
        model = attempt['model_result']
        obj, error = parse_json(model['raw_output'])
        if error or model.get('truncated') or not isinstance(obj, dict) or set(obj) != {'possible_explanations', 'questions', 'limitations'}:
            reasons = ['cannot_extract_complete_structured_questions']
            questions = []
        else:
            questions = obj['questions']
            # Check questions as a separate answer, with no unsupported explanations
            # or model-authored limitations carried over from a failed candidate.
            reasons = validate_fact_analysis(dict(possible_explanations=[], questions=questions, limitations=[]), result['user_symptoms'])
        result['partial_fallback_checks'].append(dict(source_analysis_attempt=index+1, reasons=reasons))
        if reasons:
            continue
        result['questions'] = list(questions)
        result['possible_explanations'] = []
        result['analysis_action'] = 'questions_only_after_explanation_rejection_requires_review'
        result['selected_questions_from_analysis_attempt'] = index+1
        result['limitations'] = [s for s in result['limitations'] if s != '模型分析未通过检查，保留症状原文并建议补充信息。']
        result['limitations'].append('完整分析未通过检查；仅保留单独通过有限检查的追问，不展示被拒绝的解释。')
        result['requires_review'] = True
        break
    return result


class QuestionFallbackAssistant(FactGuardedAssistant):
    def run(self, symptoms, image=None):
        return retain_checked_questions(super().run(symptoms, image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--symptoms-json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    symptoms = json.loads(pathlib.Path(args.symptoms_json).read_text(encoding='utf-8')) if args.symptoms_json else []
    result = QuestionFallbackAssistant(max_retries=args.max_retries).run(symptoms, args.image)
    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(dest), analysis_action=result['analysis_action'], model_invocations=result['model_invocations']), ensure_ascii=False))


if __name__ == '__main__':
    main()
