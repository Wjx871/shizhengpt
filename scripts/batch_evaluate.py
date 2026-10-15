import argparse,csv,json,pathlib,time,traceback
from infer_image import Engine,ROOT
FIELDS=['test_id','category','image_path','prompt_id','dtype','generation_config','latency_sec','peak_memory','raw_output','visual_correctness','groundedness','tcm_relevance','consistency','explainability','hallucination','notes','preprocess_sec','generation_sec','output_tokens','tokens_per_sec','truncated','status']
def run(manifest,output):
 cases=json.loads(pathlib.Path(manifest).read_text());e=Engine();start=time.time();records=[]
 with open(output,'w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
  for case in cases:
   for repeat in range(1,case.get('repeats',3)+1):
    ident=case['test_id']+f'_r{repeat}';print('START',ident,flush=True)
    try:
     prompt=(ROOT/'evaluation/prompts'/f"{case['prompt_id']}.txt").read_text()+case.get('suffix','')
     image=str(ROOT/case['image_path']) if case.get('image_path') else None
     result=e.infer(prompt,image);result.update(test_id=ident,category=case['category'],prompt_id=case['prompt_id'],status='ok')
    except Exception as ex:
     traceback.print_exc();result={'test_id':ident,'category':case['category'],'status':'error','notes':type(ex).__name__+': '+str(ex)}
    (ROOT/'evaluation/outputs'/f'{ident}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    row={k:result.get(k,'') for k in FIELDS};row['generation_config']=json.dumps(row['generation_config'],ensure_ascii=False);writer.writerow(row);f.flush();records.append(result)
    print('DONE',ident,result.get('latency_sec'),result.get('status'),flush=True)
 (ROOT/'evaluation/results'/('run_summary.json' if pathlib.Path(output).name=='results.csv' else pathlib.Path(output).stem+'_summary.json')).write_text(json.dumps({'elapsed_sec':time.time()-start,'model_load_sec':e.load_sec,'load_allocated_bytes':e.load_allocated,'load_reserved_bytes':e.load_reserved,'cases':len(records),'success':sum(r['status']=='ok' for r in records)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(ROOT/'evaluation/manifest.json'));p.add_argument('--output',default=str(ROOT/'evaluation/results/results.csv'));a=p.parse_args();run(a.manifest,a.output)
