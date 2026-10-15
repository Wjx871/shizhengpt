"""Audit saved regression evidence; performs no model inference."""
import csv
import hashlib
import json
import pathlib
import shutil

from guarded_consultation import validate_analysis, validate_visual, parse_json

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / 'evaluation/guarded_consultation_20261008'
FINAL = ROOT / 'evaluation/guarded_consultation_final_20261008'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest = read(FINAL / 'manifest.json')
    assert manifest == read(BASE / 'manifest.json')
    config = read(BASE / 'run.json')['config']
    rows = []
    for case in manifest:
        ident = case['id']
        baseline = read(BASE / 'outputs' / f'{ident}_baseline.json')
        result = read(FINAL / 'outputs' / f'{ident}_guarded.json')
        assert baseline['evaluation_config'] == result['evaluation_config'] == config
        assert result['user_symptoms'] == case['symptoms']
        assert result['source_symptoms'] == {f'S{i+1}': s for i, s in enumerate(case['symptoms'])}
        if case['image']:
            assert result['cv_facts']['image_sha256'] == case['image_sha256']
            assert baseline['image_sha256'] == case['image_sha256']
        assert baseline['generation_config']['max_new_tokens'] == 512
        skipped = not result['image_present'] or not result['cv_facts']['usable_for_technical_description']
        if skipped:
            assert not result['visual_attempts'] and not result['visual_facts']
        if result['visual_facts']:
            assert not validate_visual(dict(visual_facts=result['visual_facts'], limitations=[]))
        if case['symptoms']:
            selected = result['analysis_attempts'][result['selected_analysis_attempt'] - 1]
            obj, error = parse_json(selected['model_result']['raw_output'])
            assert not error and not validate_analysis(obj, case['symptoms'])
            assert result['questions'] == obj['questions']
            assert result['possible_explanations'] == obj['possible_explanations']
        attempts = result['visual_attempts'] + result['analysis_attempts']
        assert all(a['model_result']['generation_config']['max_new_tokens'] == 256 for a in result['visual_attempts'])
        assert all(a['model_result']['generation_config']['max_new_tokens'] == 512 for a in result['analysis_attempts'])
        rows.append(dict(case_id=ident, image_sha256=case.get('image_sha256', ''),
            baseline_truncated=bool(baseline.get('truncated')),
            symptoms_preserved=result['user_symptoms'] == case['symptoms'],
            visual_action=result['visual_action'], analysis_action=result['analysis_action'],
            visual_attempts=len(result['visual_attempts']), analysis_attempts=len(result['analysis_attempts']),
            visual_facts=len(result['visual_facts']), explanations=len(result['possible_explanations']),
            questions=len(result['questions']), rejected_candidates=sum(bool(a['reasons']) for a in attempts),
            guarded_truncated_candidates=sum(bool(a['model_result'].get('truncated')) for a in attempts)))
    script_hash = hashlib.sha256((ROOT / 'scripts/guarded_consultation.py').read_bytes()).hexdigest()
    assert script_hash == read(FINAL / 'rule_review.json')['script_sha256']
    summary = dict(cases=len(rows), real_model_calls=sum(read(ROOT / 'evaluation' / folder / 'run.json').get(key, 0)
        for folder, key in [('guarded_consultation_20261008', 'model_inferences'),
                            ('guarded_consultation_v2_20261008', 'model_inferences'),
                            ('guarded_consultation_final_20261008', 'new_model_inferences')]),
        accepted_symptom_cases=sum(bool(r['questions']) for r in rows),
        retained_visual_cases=sum(bool(r['visual_facts']) for r in rows),
        qc_rejected_image_cases=sum(r['visual_action'].startswith('qc_rejected') for r in rows),
        baseline_truncated_cases=sum(r['baseline_truncated'] for r in rows),
        final_candidate_records=sum(r['visual_attempts'] + r['analysis_attempts'] for r in rows),
        final_rejected_candidates=sum(r['rejected_candidates'] for r in rows),
        final_guarded_truncated_candidates=sum(r['guarded_truncated_candidates'] for r in rows),
        final_retry_cases=sum(r['visual_attempts'] > 1 or r['analysis_attempts'] > 1 for r in rows),
        script_sha256=script_hash, config=config,
        note='Regression evidence, not clinical accuracy. Final visual records are reused from v2. Revalidation made zero new generations.')
    (FINAL / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (FINAL / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(ROOT / 'scripts/guarded_consultation.py', FINAL / 'sources/guarded_consultation_reviewed.py')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
