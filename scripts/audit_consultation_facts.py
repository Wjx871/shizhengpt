"""Audit frozen paired run. Rule flags are not independently rated hallucinations."""
import csv
import hashlib
import json
import pathlib
import statistics

from consultation_facts import extract_facts
from guarded_consultation import parse_json
from guarded_consultation_v3 import validate_fact_analysis, validate_scoped_visual

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/consultation_facts_20261008'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest = read(OUT / 'manifest.json')
    run = read(OUT / 'run.json')
    frozen = read(OUT / 'freeze.json')
    assert run['source_sha256'] == frozen['source_sha256']
    for name, sha in frozen['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((OUT / 'sources' / name).read_bytes()).hexdigest() == sha
    rows = []
    review = []
    invocations = 0
    for case in manifest:
        pair = []
        for arm in ['v2', 'v3']:
            result = read(OUT / 'outputs' / f"{case['id']}_{arm}.json")
            assert result['user_symptoms'] == case['symptoms']
            assert result['source_symptoms'] == {f'S{i+1}': s for i, s in enumerate(case['symptoms'])}
            assert result['evaluation_config'] == run['config']
            if case['image']: assert result['cv_facts']['image_sha256'] == case['image_sha256']
            if not result['image_present'] or not result['cv_facts']['usable_for_technical_description']:
                assert not result['visual_attempts'] and not result['visual_facts']
            low_resolution = 'low_resolution_for_fine_facial_features' in result['cv_facts']['warnings']
            visual_flags = validate_scoped_visual(dict(visual_facts=result['visual_facts'], limitations=[]), low_resolution)
            analysis_flags = []
            if case['symptoms'] and result['analysis_action']=='generated_analysis_requires_review':
                analysis_flags = validate_fact_analysis({k: result[k] for k in ['possible_explanations', 'questions']} | dict(limitations=[]), case['symptoms'])
            if arm=='v3':
                assert not visual_flags and not analysis_flags
                assert result['symptom_facts'] == extract_facts(case['symptoms'])
                for record in result['symptom_facts']:
                    for clause in record['clauses']:
                        assert record['raw'][clause['start']:clause['end']] == clause['text']
                        for finding in clause['findings']:
                            evidence=finding['evidence']
                            assert record['raw'][evidence['start']:evidence['end']] == evidence['text']
            attempts = result['visual_attempts'] + result['analysis_attempts']
            assert len(attempts) == result['model_invocations']
            for stage, cap in [('visual_attempts',256), ('analysis_attempts',512)]:
                assert len(result[stage]) <= 2
                assert all(a['model_result']['generation_config']['max_new_tokens']==cap for a in result[stage])
            invocations += len(attempts)
            row = dict(case_id=case['id'], arm=arm, group=case['group'], symptom_case=bool(case['symptoms']),
                analysis_retained=result['analysis_action']=='generated_analysis_requires_review',
                visual_retained=bool(result['visual_facts']), analysis_action=result['analysis_action'],
                visual_action=result['visual_action'], calls=len(attempts),
                retry_stages=sum(len(result[s])>1 for s in ['visual_attempts','analysis_attempts']),
                rejected_candidates=sum(bool(a['reasons']) for a in attempts),
                truncated_candidates=sum(bool(a['model_result'].get('truncated')) for a in attempts),
                final_analysis_rule_flags=';'.join(analysis_flags), final_visual_rule_flags=';'.join(visual_flags),
                model_latency_sec=result['model_latency_sec'], questions=len(result['questions']))
            rows.append(row)
            pair.append(dict(arm=arm, symptoms=case['symptoms'], visual_facts=result['visual_facts'],
                possible_explanations=result['possible_explanations'], questions=result['questions'],
                limitations=result['limitations'], analysis_rule_flags=analysis_flags, visual_rule_flags=visual_flags,
                candidate_checks=[dict(reasons=a['reasons'], raw_output=a['model_result']['raw_output']) for a in attempts]))
        review.append(dict(case_id=case['id'], outputs=pair))
    assert invocations == run['model_inferences']
    summary = dict(cases=len(manifest), symptom_cases=sum(bool(c['symptoms']) for c in manifest),
        model_inferences=invocations, source_sha256=frozen['source_sha256'], arms={})
    for arm in ['v2', 'v3']:
        selected = [r for r in rows if r['arm']==arm]
        summary['arms'][arm] = dict(calls=sum(r['calls'] for r in selected),
            retained_analysis_cases=sum(r['analysis_retained'] for r in selected),
            retained_visual_cases=sum(r['visual_retained'] for r in selected),
            retained_analysis_cases_with_v3_flags=sum(bool(r['final_analysis_rule_flags']) for r in selected),
            retained_visual_cases_with_v3_flags=sum(bool(r['final_visual_rule_flags']) for r in selected),
            retry_stages=sum(r['retry_stages'] for r in selected), rejected_candidates=sum(r['rejected_candidates'] for r in selected),
            truncated_candidates=sum(r['truncated_candidates'] for r in selected),
            model_latency_total_sec=round(sum(r['model_latency_sec'] for r in selected),2),
            median_case_model_latency_sec=round(statistics.median(r['model_latency_sec'] for r in selected),2))
    summary['interpretation'] = 'V3 flags measure its own finite rules, not independent hallucination accuracy. Inspect review.json manually; no clinical labels.'
    (OUT / 'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT / 'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT / 'comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
