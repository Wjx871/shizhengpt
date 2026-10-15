import json,pathlib
from safetensors import safe_open
ROOT=pathlib.Path(__file__).resolve().parents[1]; model=ROOT.parent/'models/ShizhenGPT-7B-VL'
index=json.loads((model/'model.safetensors.index.json').read_text());expected=index['weight_map'];observed={};dtype_counts={}
for shard in sorted(set(expected.values())):
 with safe_open(model/shard,framework='pt',device='cpu') as f:
  for key in f.keys():
   assert key not in observed,key
   observed[key]=shard;dtype=f.get_slice(key).get_dtype();dtype_counts[dtype]=dtype_counts.get(dtype,0)+1
assert observed==expected,'index/header tensor mismatch'
for name in ['config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','chat_template.json','vocab.json','merges.txt']:
 assert (model/name).stat().st_size>0,name
result={'tensors':len(observed),'shards':len(set(expected.values())),'dtype_tensor_counts':dtype_counts,'index_total_size_bytes':index['metadata']['total_size'],'complete':True}
(ROOT/'docs/sources/model_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
