"""Raw base/LoRA comparison on frozen engineering tests and transfer regressions."""
import hashlib
import json
import pathlib
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/alignment_lora_pilot_20261009'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    from peft import PeftModel
    from infer_image import Engine
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['source_sha256'].items():assert sha(ROOT/'scripts'/name)==expected,name
    assert sha(OUT/'test.jsonl')==protocol['test_jsonl_sha256']
    rows=[json.loads(line) for line in (OUT/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    rows+=json.loads((OUT/'regression.json').read_text(encoding='utf-8'))
    assert len(rows)==20
    engine=Engine();started=time.time();calls=0
    for arm in ['base','lora']:
        if arm=='lora':
            engine.model=PeftModel.from_pretrained(engine.model,OUT/'adapter',is_trainable=False).eval()
            engine.model.config.use_cache=True
        for row in rows:
            dest=OUT/'outputs'/f"{row['id']}_{arm}.json"
            if dest.exists():raise RuntimeError('Preserve evaluation history')
            image=str(ROOT/row['image']) if row['image'] else None
            if row.get('image_sha256'):assert sha(ROOT/row['image'])==row['image_sha256']
            cap=768 if row['task']=='regression' else 512
            print('EVAL',arm,row['id'],flush=True)
            result=engine.infer(row['prompt'],image,max_new_tokens=cap)
            result.update(case_id=row['id'],arm=arm,task=row['task'],evaluation_config=engine.config)
            save(dest,result);calls+=1
            print('DONE',arm,row['id'],'tokens',result['output_tokens'],'truncated',result['truncated'],flush=True)
    save(OUT/'evaluation_run.json',dict(model_inferences=calls,cases=20,elapsed_sec=time.time()-started,config=engine.config,
         no_guarded_wrapper=True,no_retries=True,checkpoint_selection='fixed final training checkpoint',
         training_run_sha256=sha(OUT/'training_run.json'),protocol_sha256=sha(OUT/'protocol.json')))
    print('EVAL_COMPLETE',calls,flush=True)


if __name__=='__main__':main()
