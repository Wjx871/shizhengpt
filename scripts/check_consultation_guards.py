"""Deterministic negative controls, no real model needed."""
import json
import sys
import types
from guarded_consultation import GuardedAssistant, validate_visual, validate_analysis


def main():
    symptoms=['最近容易疲劳','手脚较凉','睡眠质量一般','偶尔食欲不佳']
    bad=dict(possible_explanations=[dict(basis_ids=['S3'],explanation='睡眠质量差可能导致疲劳')],questions=['持续多久？'],limitations=[])
    good=dict(possible_explanations=[dict(basis_ids=['S1'],explanation='疲劳可能与休息及生活方式因素有关，需进一步核实')],questions=['近期作息和活动量如何？'],limitations=['尚需补充信息'])
    assert 'sleep_severity_changed_or_added' in validate_analysis(bad,symptoms)
    assert not validate_analysis(good,symptoms)
    assert 'question_presupposes_sleep_severity_changed_or_added' in validate_analysis({**good,'questions':['睡眠质量差具体表现是什么？']},symptoms)
    assert not validate_analysis({**good,'questions':['是否存在失眠或多梦？']},symptoms)
    assert not validate_analysis({**good,'questions':['睡眠质量一般具体指入睡困难、易醒还是多梦？']},symptoms)
    bad_basis={**good,'possible_explanations':[dict(basis_ids=['S99'],explanation='可能与休息因素有关')]}
    assert 'unknown_or_missing_symptom_basis' in validate_analysis(bad_basis,symptoms)
    assert 'unsupported_demographics' in validate_visual(dict(visual_facts=['11岁女性'],limitations=[]))
    assert 'unsupported_tongue_pulse_motion' in validate_visual(dict(visual_facts=['舌体活动正常'],limitations=[]))
    original_module=sys.modules.get('face_quality')
    sys.modules['face_quality']=types.SimpleNamespace(analyze=lambda image:dict(
        usable_for_technical_description=False,reject_reasons=['fixture_rejection'],warnings=[]))
    calls=[]
    def fake_infer(prompt,image=None,max_new_tokens=None):
        calls.append(image)
        return dict(raw_output=json.dumps(bad if len(calls)==1 else good,ensure_ascii=False),latency_sec=0.0,truncated=False)
    try:
        result=GuardedAssistant(infer=fake_infer,max_retries=1).run(symptoms,None)
        assert calls==[None,None] and result['visual_facts']==[] and result['user_symptoms']==symptoms
        assert result['analysis_action']=='generated_analysis_requires_review'
        assert result['analysis_attempts'][0]['reasons']==['sleep_severity_changed_or_added']
        calls.clear()
        result=GuardedAssistant(infer=fake_infer).run([], 'rejected_fixture.png')
        assert not calls and result['model_invocations']==0
        assert result['visual_action']=='qc_rejected_visual_generation_skipped'
        result=GuardedAssistant(infer=lambda *a,**k:dict(raw_output='not JSON',latency_sec=0.0,truncated=False)).run(symptoms,None)
        assert result['analysis_action']=='analysis_output_rejected_after_retry' and result['user_symptoms']==symptoms
    finally:
        if original_module is None:sys.modules.pop('face_quality',None)
        else:sys.modules['face_quality']=original_module
    print('PASS: symptom rewrite detection, source IDs, demographic/motion claims, bounded retry, no-image routing, rejected-image zero calls, failed-output fallback')


if __name__=='__main__':main()
