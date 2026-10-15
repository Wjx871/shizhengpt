"""Run the evidence-first prototype against the existing local chat service."""
import argparse
import base64
import json
import pathlib

from consultation_alignment_v4 import run_case
from evaluate_alignment_v4 import request


def infer(prompt,image=None):
    encoded=base64.b64encode(pathlib.Path(image).read_bytes()).decode() if image else None
    response=request('/api/chat',{'mode':'raw','text':prompt,'image':encoded})
    request('/api/reset',{'session_id':response['session_id']})
    return {'raw_output':response['answer'],'latency_sec':response['latency_sec'],
            'truncated':response['truncated'],'prompt':prompt,'image_path':image}


if __name__=='__main__':
    from face_quality import analyze
    parser=argparse.ArgumentParser()
    parser.add_argument('--symptoms-json',help='UTF-8 JSON array of verbatim symptom strings')
    parser.add_argument('--image',help='Face photo, optional')
    parser.add_argument('--output',required=True)
    parser.add_argument('--version',choices=['v4','v41'],default='v41')
    args=parser.parse_args()
    symptoms=json.loads(pathlib.Path(args.symptoms_json).read_text(encoding='utf-8-sig')) if args.symptoms_json else []
    if not isinstance(symptoms,list) or not all(isinstance(s,str) and s.strip() for s in symptoms):
        parser.error('symptoms-json must contain a string array')
    if not symptoms and not args.image:parser.error('Supply symptoms or an image')
    case={'symptoms':symptoms,'image':args.image}
    if args.version=='v41':
        from consultation_alignment_v41 import run
        result=run(infer,case,qc=analyze)
    else:
        result=run_case(infer,case,enhanced=True,qc=analyze)
    output=pathlib.Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'analysis':result['analysis'],'visual':result['visual'],
                      'analysis_action':result['analysis_action'],'visual_action':result['visual_action']},ensure_ascii=False,indent=2))
