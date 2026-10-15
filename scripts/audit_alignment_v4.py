"""Post-run structural metrics and predeclared phrase screens, no inference."""
import collections
import json
import pathlib
import statistics

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/alignment_engineering_v4_20261009'


def audit_answer(case,analysis,visual):
    sources={f'S{i+1}':s for i,s in enumerate(case['symptoms'])}
    metrics={'phrase_screen_hits':[], 'evidence_items':0,'invalid_evidence_items':0,'questions':0,'explanations':0,
             'visual_exact':0,'visual_unknown':0,'visual_explicit_mismatch':0,'visual_reference_items':len(case.get('expected_visual_status',{}))}
    if isinstance(analysis,dict):
        questions=analysis.get('questions',[])
        explanations=analysis.get('possible_explanations',[])
        metrics['questions']=len(questions) if isinstance(questions,list) else 0
        metrics['explanations']=len(explanations) if isinstance(explanations,list) else 0
        text=' '.join(q for q in questions if isinstance(q,str)) if isinstance(questions,list) else ''
        if isinstance(explanations,list):
            for item in explanations:
                if not isinstance(item,dict):continue
                text+=' '+str(item.get('hypothesis',''))
                for e in item.get('evidence',[]) if isinstance(item.get('evidence'),list) else []:
                    metrics['evidence_items']+=1
                    if not isinstance(e,dict):metrics['invalid_evidence_items']+=1;continue
                    source=sources.get(e.get('source_id')) if isinstance(e.get('source_id'),str) else None
                    quote=e.get('quote')
                    if not source or not isinstance(quote,str) or not quote or quote not in source:
                        metrics['invalid_evidence_items']+=1
        metrics['phrase_screen_hits']=[p for p in case['rubric_forbidden_phrases'] if p in text]
    status=visual.get('visual_status',{}) if isinstance(visual,dict) else {}
    if not isinstance(status,dict):status={}
    for key,expected in case.get('expected_visual_status',{}).items():
        actual=status.get(key,'unknown')
        if actual==expected:metrics['visual_exact']+=1
        elif actual not in ('visible','not_visible'):metrics['visual_unknown']+=1
        else:metrics['visual_explicit_mismatch']+=1
    return metrics


def main(directory=None):
    global OUT
    if directory:
        OUT=pathlib.Path(directory)
    cases=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));rows=[];review=[]
    for case in cases:
        for arm in ['base','enhanced']:
            path=OUT/'outputs'/f"{case['id']}_{arm}.json"
            result=json.loads(path.read_text(encoding='utf-8'))
            analysis_first=result['analysis_attempts'][0]['parsed'] if result['analysis_attempts'] else None
            visual_first=result['visual_attempts'][0]['parsed'] if result['visual_attempts'] else None
            metrics=audit_answer(case,result['analysis'],result['visual'])
            first=audit_answer(case,analysis_first,visual_first)
            row={'id':case['id'],'kind':case['kind'],'arm':arm,'has_symptoms':bool(case['symptoms']),
                 'analysis_action':result['analysis_action'],'visual_action':result['visual_action'],
                 'model_invocations':result['model_invocations'],'model_latency_sec':result['model_latency_sec'],
                 'retained':metrics,'first_candidate':first,
                 'rejected_candidates':sum(bool(a['reasons']) for a in result['analysis_attempts']+result['visual_attempts'])}
            rows.append(row)
            review.append({'id':case['id'],'arm':arm,'symptoms':case['symptoms'],'visual_reference':case.get('expected_visual_status'),
                           'retained_analysis':result['analysis'],'retained_visual':result['visual'],
                           'analysis_candidates':[{'parsed':a['parsed'],'reasons':a['reasons']} for a in result['analysis_attempts']],
                           'visual_candidates':[{'parsed':a['parsed'],'reasons':a['reasons']} for a in result['visual_attempts']],
                           'qc_reasons':result.get('qc',{}).get('reject_reasons') if result.get('qc') else [],'metrics':metrics})
    summaries={}
    for arm in ['base','enhanced']:
        group=[r for r in rows if r['arm']==arm]
        summary={'conditions':len(group),'symptom_conditions':sum(r['has_symptoms'] for r in group),
                 'full_analysis':sum(r['analysis_action']=='full_answer_requires_review' for r in group),
                 'nonempty_explanation_conditions':sum(r['retained']['explanations']>0 for r in group),
                 'question_conditions':sum(r['retained']['questions']>0 for r in group),
                 'questions_only_conditions':sum(r['analysis_action']=='questions_only_requires_review' for r in group),
                 'partial_checked_conditions':sum(r['analysis_action']=='partial_checked_content_requires_review' for r in group),
                 'visual_qc_abstentions':sum(r['visual_action']=='qc_abstention' for r in group),
                 'visual_output_abstentions':sum(r['visual_action']=='output_abstention' for r in group),
                 'model_invocations':sum(r['model_invocations'] for r in group),
                 'rejected_candidates':sum(r['rejected_candidates'] for r in group),
                 'model_latency_sec':round(sum(r['model_latency_sec'] for r in group),2),
                 'median_condition_latency_sec':round(statistics.median(r['model_latency_sec'] for r in group),2)}
        for view in ['first_candidate','retained']:
            summary[view]={key:sum(r[view][key] for r in group) for key in ['evidence_items','invalid_evidence_items','visual_exact','visual_unknown','visual_explicit_mismatch','visual_reference_items']}
            summary[view]['phrase_flag_conditions']=sum(bool(r[view]['phrase_screen_hits']) for r in group)
        summaries[arm]=summary
    for name,value in [('audit.json',{'summary':summaries,'rows':rows}),('review.json',review)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--directory');args=p.parse_args()
    main(args.directory)
