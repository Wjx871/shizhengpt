"""Development controls for v3; separate from frozen model comparison cases."""
import json
import sys
import types

from consultation_facts import extract_facts, evidence_terms, question_risks
from guarded_consultation_v3 import FactGuardedAssistant, validate_fact_analysis, validate_scoped_visual


def response(text='可能与作息因素有关，需补充信息', question='近期作息如何？', ids=None):
    return dict(possible_explanations=[dict(basis_ids=ids or ['S1'], explanation=text)], questions=[question], limitations=[])


def main():
    symptoms = ['近两天有点疲劳', '没有发热和胸痛', '偶尔食欲不佳', '上周头痛已缓解']
    facts = extract_facts(symptoms)
    states = evidence_terms(facts)
    assert states['fatigue'] == {'reported'}
    assert states['fever'] == states['chest_pain'] == {'negative'}
    assert states['headache'] == {'resolved'}
    assert evidence_terms(extract_facts(['没有发热但有咳嗽']))['cough'] == {'reported'}
    assert evidence_terms(extract_facts(['不发热']))['fever'] == {'negative'}
    for record in facts:
        for clause in record['clauses']:
            for item in clause['findings'] + [v for items in clause['qualifiers'].values() for v in items]:
                evidence = item.get('evidence', item)
                assert record['raw'][evidence['start']:evidence['end']] == evidence['text']
    for question in ['胸痛是否持续？', '发热多久了？', '心悸是否每天出现？', '头痛何时加重？']:
        assert question_risks(question, facts), question
    for question in ['是否有心悸？', '有无恶心？', '如果出现胸痛，是否伴随其他不适？', '最近作息如何？']:
        assert not question_risks(question, facts), question
    assert not validate_fact_analysis(response(), symptoms)
    assert 'explanation_unsupported_symptom_fever' in validate_fact_analysis(response('发热可能与感染有关', ids=['S2']), symptoms)
    assert 'explanation_added_severity' in validate_fact_analysis(response('严重疲劳可能与休息有关'), symptoms)
    assert 'explanation_added_time' in validate_fact_analysis(response('疲劳一个月可能与作息有关'), symptoms)
    assert validate_fact_analysis(response(question='心悸是否持续？'), symptoms)
    assert validate_fact_analysis(response(question='严重疲劳持续多久？'), symptoms)
    assert not validate_fact_analysis(response(question='疲劳是否严重？'), symptoms)
    assert 'negative_or_resolved_only_basis' in validate_fact_analysis(response(ids=['S2']), symptoms)
    assert validate_scoped_visual(dict(visual_facts=['没有明显皮疹'], limitations=[]))
    assert validate_scoped_visual(dict(visual_facts=['佩戴头盔'], limitations=[]))
    assert validate_scoped_visual(dict(visual_facts=['皮肤毛孔清晰'], limitations=[]), True)
    assert not validate_scoped_visual(dict(visual_facts=['面部朝向画面正前方'], limitations=['细节无法判断']), True)
    for malformed in [None, [], {}, {'possible_explanations': [], 'questions': 'bad', 'limitations': []}]:
        assert validate_fact_analysis(malformed, symptoms)
    original = sys.modules.get('face_quality')
    sys.modules['face_quality'] = types.SimpleNamespace(analyze=lambda image:dict(usable_for_technical_description=False, warnings=[]))
    calls = []
    def infer(prompt, image=None, max_new_tokens=None):
        calls.append(image)
        bad = response(question='发热是否持续？')
        return dict(raw_output=json.dumps(bad if len(calls)==1 else response(), ensure_ascii=False), latency_sec=0, truncated=False)
    try:
        result = FactGuardedAssistant(infer=infer).run(symptoms, 'qc_rejected_fixture')
        assert calls == [None, None] and result['user_symptoms'] == symptoms
        assert result['analysis_action'] == 'generated_analysis_requires_review'
        assert result['symptom_facts'] == facts and not result['visual_facts']
    finally:
        if original is None: sys.modules.pop('face_quality', None)
        else: sys.modules['face_quality'] = original
    print('PASS: exact spans, negation, resolved state, degree/time, local question premise, visual scope, schema, retry, text fallback')


if __name__ == '__main__':
    main()
