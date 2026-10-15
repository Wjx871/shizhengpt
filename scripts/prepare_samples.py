import hashlib,json,pathlib,urllib.request,zipfile,collections
ROOT=pathlib.Path(__file__).resolve().parents[1]; cache=ROOT.parent/'datasets'/'TCM-Vision-Benchmark'; cache.mkdir(parents=True,exist_ok=True)
repo='FreedomIntelligence/TCM-Vision-Benchmark'
manifest=json.load(urllib.request.urlopen(f'https://modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision=master&Recursive=true',timeout=30))
(ROOT/'docs/sources/vision_dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
for f in manifest['Data']['Files']:
 if f['Path'] not in ['README.md','TCM-Vision_Benchmark.json','tcm_bench_images.zip']:continue
 dest=cache/f['Path']; temp=dest.with_suffix(dest.suffix+'.partial'); h=hashlib.sha256()
 url=f'https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision={f["Revision"]}&FilePath={f["Path"]}'
 with urllib.request.urlopen(url,timeout=60) as response, open(temp,'wb') as out:
  while True:
   data=response.read(8*1024*1024)
   if not data:break
   out.write(data);h.update(data)
 assert h.hexdigest()==f['Sha256'],f['Path'];temp.replace(dest);print('VERIFIED',dest.name,dest.stat().st_size,flush=True)
print((cache/'README.md').read_text(),flush=True)
data=json.loads((cache/'TCM-Vision_Benchmark.json').read_text());print('JSON type',type(data).__name__,flush=True)
if isinstance(data,dict):print('keys',list(data),flush=True)
else:print('first rows',data[:2],flush=True)
with zipfile.ZipFile(cache/'tcm_bench_images.zip') as z:
 print('ZIP files',z.namelist()[:30],flush=True)
