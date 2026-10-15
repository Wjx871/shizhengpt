import collections,csv,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(name):return list(csv.DictReader(open(ROOT/'evaluation/results'/name,encoding='utf-8-sig')))
main=read('results.csv');control=read('prompt_controls.csv');permuted=read('mc_permutations.csv');allrows=main+control+permuted
lookup={c['test_id']:c for c in json.load(open(ROOT/'evaluation/mc_permutation_manifest.json'))};checks=[]
for r in permuted:
 c=lookup[r['test_id'].rsplit('_r',1)[0]];checks.append({'test_id':r['test_id'],'predicted':r['raw_output'].strip(),'expected':c['expected'],'correct':r['raw_output'].strip()==c['expected']})
original=json.load(open(ROOT/'evaluation/results/summary.json'))['official_mc_selected_samples'];per_image={}
for ident in ['3245','3248','3249','3263','3376','3912']:
 selected=[c for c in checks if f'_{ident}_' in c['test_id']];first=next(c for c in original if c['id']==ident);per_image[ident]={'correct_out_of_4':int(first['correct'])+sum(c['correct'] for c in selected),'original':first,'rotated':selected}
groups=collections.defaultdict(list)
for r in control:groups[r['test_id'].rsplit('_r',1)[0]].append(r)
rawfiles=[json.load(open(ROOT/'evaluation/outputs'/f"{r['test_id']}.json")) for r in allrows]
r={'formal_total':len(allrows),'formal_success':sum(r['status']=='ok' for r in allrows),'formal_truncated':sum(r['truncated']=='True' for r in allrows),'max_allocated_gib':max(int(r['peak_memory']) for r in allrows)/2**30,'max_reserved_gib':max(x['peak_reserved_bytes'] for x in rawfiles)/2**30,'control_repeatability':{k:{'runs':len(v),'distinct_outputs':len(set(x['raw_output'] for x in v))} for k,v in groups.items()},'mc_permuted_correct':sum(c['correct'] for c in checks),'mc_permuted_total':len(checks),'mc_combined_correct':sum(c['correct'] for c in checks)+sum(c['correct'] for c in original),'mc_combined_total':len(checks)+len(original),'mc_combined_best_constant_baseline_correct':max(collections.Counter(c['expected'] for c in checks).get(k,0)+sum(c['expected']==k for c in original) for k in 'ABCD'),'mc_per_image':per_image,'replacement_character_test_ids':[x['test_id'] for x in rawfiles if '\ufffd' in x['raw_output']]}
(ROOT/'evaluation/results/controls_summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
