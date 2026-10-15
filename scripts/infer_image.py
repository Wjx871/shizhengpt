"""Official Qwen2.5-VL pipeline; CUDA API denotes HIP on this DCU image."""
import argparse,hashlib,json,pathlib,time
import torch,yaml
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Engine:
 def __init__(self,config=None):
  self.config=config or yaml.safe_load((ROOT/'configs/inference.yaml').read_text())
  c=self.config;torch.manual_seed(c['seed']);torch.set_num_threads(8)
  if not torch.cuda.is_available():raise RuntimeError('HIP/DCU unavailable')
  if not (ROOT/'docs/sources/download_complete.json').exists():raise RuntimeError('Model download not verified complete')
  start=time.perf_counter()
  self.processor=AutoProcessor.from_pretrained(c['model_path'],local_files_only=True,use_fast=False,min_pixels=c['min_pixels'],max_pixels=c['max_pixels'])
  self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(c['model_path'],torch_dtype=getattr(torch,c['dtype']),device_map={'':'cuda:0'},attn_implementation=c['attention'],local_files_only=True).eval()
  torch.cuda.synchronize();self.load_sec=time.perf_counter()-start
  self.load_allocated=torch.cuda.memory_allocated();self.load_reserved=torch.cuda.memory_reserved()
  (ROOT/'docs/sources/loaded_backend.json').write_text(json.dumps({'attention':self.model.config._attn_implementation,'vision_attention':getattr(self.model.config.vision_config,'_attn_implementation',None),'dtype':str(next(self.model.parameters()).dtype),'device':str(self.model.device),'generation_defaults':self.model.generation_config.to_dict()},indent=2))
  print(json.dumps({'load_sec':self.load_sec,'load_allocated_bytes':self.load_allocated,'load_reserved_bytes':self.load_reserved,'dtype':str(next(self.model.parameters()).dtype),'device':str(self.model.device)},ensure_ascii=False),flush=True)
 def infer(self,prompt,image=None,max_new_tokens=None):
  content=[]
  if image:content.append({'type':'image','image':str(pathlib.Path(image).resolve()),'max_pixels':self.config['max_pixels']})
  content.append({'type':'text','text':prompt});messages=[{'role':'user','content':content}]
  start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  images,videos=process_vision_info(messages)
  inputs=self.processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt').to(self.model.device)
  torch.cuda.synchronize();prep_sec=time.perf_counter()-start
  gc={'max_new_tokens':max_new_tokens or self.config['max_new_tokens'],'do_sample':self.config['do_sample']}
  if gc['do_sample']:gc.update(temperature=0.7,top_p=0.9)
  generate_start=time.perf_counter()
  with torch.inference_mode():ids=self.model.generate(**inputs,**gc)
  torch.cuda.synchronize();generation_sec=time.perf_counter()-generate_start
  trimmed=ids[:,inputs.input_ids.shape[1]:]
  output=self.processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  token_count=trimmed.shape[1]
  return {'raw_output':output,'latency_sec':time.perf_counter()-start,'preprocess_sec':prep_sec,'generation_sec':generation_sec,'output_tokens':token_count,'tokens_per_sec':token_count/generation_sec,'peak_memory':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'dtype':self.config['dtype'],'generation_config':gc,'input_tokens':inputs.input_ids.shape[1],'image_grid_thw':inputs.image_grid_thw.tolist() if 'image_grid_thw' in inputs else None,'truncated':token_count>=gc['max_new_tokens'],'prompt':prompt,'image_path':image or '', 'image_sha256':hashlib.sha256(pathlib.Path(image).read_bytes()).hexdigest() if image else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--image');p.add_argument('--prompt',required=True);p.add_argument('--output',default=str(ROOT/'evaluation/outputs/single.json'));a=p.parse_args()
 e=Engine();result=e.infer(a.prompt,a.image);pathlib.Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['raw_output'])
if __name__=='__main__':main()
