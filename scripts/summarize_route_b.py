"""Mechanical fidelity metrics; no medical accuracy score."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'evaluation/route_b'
records = json.loads((out/'records.json').read_text(encoding='utf-8'))
summary = {}
for arm in ['prompt_only', 'cv_facts']:
    rows = [r for r in records if r['arm'] == arm]
    fusion = [r for r in rows if r['case_id'].startswith('fusion_')]
    groups = [[r for r in rows if r['case_id']==c] for c in sorted({r['case_id'] for r in rows})]
    summary[arm] = {
        'total': len(rows), 'success': sum(r['status']=='ok' for r in rows),
        'schema_valid': sum(r.get('schema_valid',False) for r in rows),
        'fusion_symptoms_exact': sum(r.get('symptoms_exact',False) for r in fusion),
        'fusion_total': len(fusion),
        'no_image_visual_empty': sum(r.get('no_image_visual_empty') is True for r in rows),
        'no_image_total': sum(r['case_id']=='fusion_no_image' for r in rows),
        'truncated': sum(r.get('truncated',False) for r in rows),
        'identical_repeat_groups': sum(len(g)==3 and len({r.get('raw_output') for r in g})==1 for g in groups),
        'groups': len(groups),
        'median_latency_sec': statistics.median(r['latency_sec'] for r in rows if r['status']=='ok'),
    }
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fields = ['test_id','case_id','arm','repeat','status','schema_valid','symptoms_exact',
          'no_image_visual_empty','cv_gate','latency_sec','output_tokens','truncated','raw_output']
with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
print(json.dumps(summary,indent=2))
for case in sorted({r['case_id'] for r in records}):
    for arm in ['prompt_only','cv_facts']:
        r=next(r for r in records if r['case_id']==case and r['arm']==arm)
        print(case,arm,r.get('raw_output'))
