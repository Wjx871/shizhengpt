"""Component acceptance must not turn failed explanations into valid answers."""
import copy
import json

from guarded_consultation_v31 import retain_checked_questions


def fixture(questions, explanation='需要核实的可能解释', truncated=False):
    obj=dict(possible_explanations=[dict(basis_ids=['S1'],explanation=explanation)],questions=questions,limitations=['模型自写的限制说明'])
    return dict(user_symptoms=['没有胸痛', '近两天有点疲劳'], analysis_action='analysis_output_rejected_after_retry',
        possible_explanations=[],questions=[], limitations=['模型分析未通过检查，保留症状原文并建议补充信息。'],
        requires_review=False, model_invocations=1,
        analysis_attempts=[dict(reasons=['unfilled_template_placeholder'], model_result=dict(raw_output=json.dumps(obj,ensure_ascii=False),truncated=truncated))])


def main():
    original=fixture(['近期作息如何？', '是否有头晕？'])
    backup=copy.deepcopy(original)
    result=retain_checked_questions(original)
    assert original==backup and result['analysis_attempts']==original['analysis_attempts']
    assert result['analysis_action']=='questions_only_after_explanation_rejection_requires_review'
    assert result['questions']==['近期作息如何？', '是否有头晕？'] and not result['possible_explanations']
    assert '模型自写的限制说明' not in result['limitations']
    assert result['model_invocations']==1
    for questions, truncated in [(['胸痛是否持续？'],False),([],False),(['近期作息如何？'],True),(['严重疲劳多久了？'],False)]:
        result=retain_checked_questions(fixture(questions,truncated=truncated))
        assert result['analysis_action']=='analysis_output_rejected_after_retry' and not result['questions']
    print('PASS: safe component fallback, immutable raw records, no failed explanations, premise checks, empty/truncated rejection, zero added calls')


if __name__=='__main__':
    main()
