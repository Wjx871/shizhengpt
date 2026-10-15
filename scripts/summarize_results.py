import collections,csv,hashlib,json,pathlib,re,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'evaluation/results/results.csv',encoding='utf-8-sig')))
valid=[r for r in rows if r['status']=='ok'];groups=collections.defaultdict(list)
for r in valid:groups[r['category']].append(r)
summary={'total':len(rows),'successful':len(valid),'errors':[r for r in rows if r['status']!='ok'],'truncated_count':sum(r['truncated']=='True' for r in valid),'categories':{}}
for category,records in groups.items():
 values=[float(r['latency_sec']) for r in records];summary['categories'][category]={'runs':len(records),'median_latency_sec':statistics.median(values),'min_latency_sec':min(values),'max_latency_sec':max(values),'max_allocated_gib':max(int(r['peak_memory']) for r in records)/2**30,'median_output_tokens':statistics.median(int(r['output_tokens']) for r in records),'median_generation_tokens_per_sec':statistics.median(float(r['tokens_per_sec']) for r in records)}
repeated=collections.defaultdict(list)
for r in valid:repeated[r['test_id'].rsplit('_r',1)[0]].append(r)
summary['repeatability']={name:{'runs':len(rr),'distinct_outputs':len(set(r['raw_output'] for r in rr))} for name,rr in repeated.items() if len(rr)>1}
dataset=json.load(open(ROOT.parent/'datasets/TCM-Vision-Benchmark/TCM-Vision_Benchmark.json'));lookup={pathlib.Path(r['image'][0]).stem:r for r in dataset}
mc=[]
for r in groups['official_visual_mc']:
 ident=r['test_id'].split('_')[1];expected=lookup[ident]['answer_idx'];match=re.match(r'^\s*([ABCD])(?:\b|[.\s]|$)',r['raw_output']);pred=match.group(1) if match else None
 mc.append({'id':ident,'predicted':pred,'expected':expected,'answer_label':lookup[ident]['answer'],'raw_output':r['raw_output'],'correct':pred==expected})
summary['official_mc_selected_samples']=mc
summary['mc_correct_count']=sum(x['correct'] for x in mc);summary['mc_total']=len(mc)
summary['human_scores_filled']=any(r[k] for r in rows for k in ['visual_correctness','groundedness','tcm_relevance','consistency','explainability'])
(ROOT/'evaluation/results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
