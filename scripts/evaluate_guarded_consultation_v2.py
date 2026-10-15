"""Re-evaluate revised guarded arm against the preserved first baseline."""
import hashlib
import json
import time
from infer_image import Engine, ROOT
from guarded_consultation import GuardedAssistant
from evaluate_guarded_consultation import save

OUT = ROOT/'evaluation/guarded_consultation_v2_20261008'
FIRST = ROOT/'evaluation/guarded_consultation_20261008'


def main():
    manifest=json.loads((FIRST/'manifest.json').read_text(encoding='utf-8'))
    save(OUT/'manifest.json',manifest)
    engine=Engine()
    assistant=GuardedAssistant(infer=engine.infer,max_retries=1)
    started=time.time()
    calls=0
    for case in manifest:
        dest=OUT/'outputs'/f"{case['id']}_guarded.json"
        if dest.exists():raise ValueError('Preserve existing output: '+str(dest))
        print('START',case['id'],flush=True)
        image=str(ROOT/case['image']) if case['image'] else None
        if image and hashlib.sha256((ROOT/case['image']).read_bytes()).hexdigest()!=case['image_sha256']:
            raise ValueError('Image changed')
        payload=assistant.run(case['symptoms'],image)
        payload.update(status='ok',case_id=case['id'],arm='guarded_v2',evaluation_config=engine.config)
        save(dest,payload)
        calls+=payload['model_invocations']
        print('DONE',case['id'],'calls',payload['model_invocations'],payload['visual_action'],payload['analysis_action'],flush=True)
    save(OUT/'run.json',dict(cases=len(manifest),pipeline_results=len(manifest),model_inferences=calls,
        elapsed_sec=time.time()-started,config=engine.config,
        script_sha256=hashlib.sha256((ROOT/'scripts/guarded_consultation.py').read_bytes()).hexdigest(),
        changed='Natural-language input envelope, explicit output format, reject unfilled placeholder and missing questions',
        baseline_source=str(FIRST/'outputs')))
    print('COMPLETE',calls,flush=True)


if __name__=='__main__':main()
