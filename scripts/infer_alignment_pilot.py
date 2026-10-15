"""Explicit opt-in adapter inference; does not change the original default entry."""
import argparse
import hashlib
import json
import pathlib


def main():
    from infer_image import Engine
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompt-file',required=True)
    parser.add_argument('--image')
    parser.add_argument('--adapter')
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-new-tokens',type=int,default=512)
    args=parser.parse_args()
    engine=Engine()
    if args.adapter:
        from peft import PeftModel
        engine.model=PeftModel.from_pretrained(engine.model,args.adapter,is_trainable=False).eval()
    result=engine.infer(pathlib.Path(args.prompt_file).read_text(encoding='utf-8'),args.image,max_new_tokens=args.max_new_tokens)
    result['adapter_path']=args.adapter
    result['adapter_sha256']=hashlib.sha256((pathlib.Path(args.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest() if args.adapter else None
    result['research_only']=True
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(path),adapter=args.adapter,truncated=result['truncated']),ensure_ascii=False))


if __name__=='__main__':main()
