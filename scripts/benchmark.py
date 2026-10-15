import argparse,json,statistics
from infer_image import Engine,ROOT
p=argparse.ArgumentParser();p.add_argument('--image',default=str(ROOT/'evaluation/images/general/shapes.png'));p.add_argument('--runs',type=int,default=3);a=p.parse_args()
e=Engine();e.infer('请客观描述图片。',a.image,128)
r=[e.infer('请客观描述图片。',a.image,128) for _ in range(a.runs)]
result={'warmup_excluded':True,'runs':a.runs,'median_latency_sec':statistics.median(x['latency_sec'] for x in r),'peak_memory_bytes':max(x['peak_memory'] for x in r),'raw_runs':r};(ROOT/'evaluation/results/benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
