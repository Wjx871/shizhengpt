import concurrent.futures,hashlib,json,pathlib,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'models'/'ShizhenGPT-7B-VL'; DEST.mkdir(parents=True,exist_ok=True)
BASE='https://hf-mirror.com'; REPO='FreedomIntelligence/ShizhenGPT-7B-VL'
def getjson(url):
    return json.load(urllib.request.urlopen(url,timeout=30))
revision='f3a2f0985690c69f2cf60aca49f180c1da783541'
ms=getjson(f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true')
(ROOT/'docs/sources/modelscope_manifest.json').write_text(json.dumps(ms,indent=2))
msfiles={f['Path']:f for f in ms['Data']['Files']}
def download(f):
    name=f['Path']; target=DEST/name; expected=f['Sha256']; size=f['Size']
    for attempt in range(4):
        try:
            url=f'https://modelscope.cn/api/v1/models/{REPO}/repo?Revision={f["Revision"]}&FilePath={name}'
            with urllib.request.urlopen(url,timeout=60) as response, open(target.with_suffix(target.suffix+'.partial'),'wb') as out:
                h=hashlib.sha256(); total=0; start=time.time(); last=start
                while True:
                    block=response.read(8*1024*1024)
                    if not block:break
                    out.write(block); h.update(block); total+=len(block)
                    if time.time()-last>30:
                        print(name,total,'bytes',round(total/(time.time()-start)/1e6,2),'MB/s',flush=True); last=time.time()
            if size and total!=size:raise RuntimeError('size mismatch')
            if expected and h.hexdigest()!=expected:raise RuntimeError('HF SHA256 mismatch')
            alt=msfiles.get(name)
            if alt and h.hexdigest()!=alt['Sha256']:raise RuntimeError('ModelScope SHA256 mismatch')
            target.with_suffix(target.suffix+'.partial').replace(target)
            print('VERIFIED',name,total,h.hexdigest(),flush=True)
            return
        except Exception as e:
            print('RETRY',name,attempt,type(e).__name__,str(e),flush=True); time.sleep(2)
    raise RuntimeError('download failed: '+name)
files=[f for f in ms['Data']['Files'] if f['Type']=='blob' and not f['Path'].startswith('.')]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download,files))
index=json.loads((DEST/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    assert (DEST/name).is_file(),name
(ROOT/'docs/sources/download_complete.json').write_text(json.dumps({'hf_reference_revision':revision,'download_source':'ModelScope FreedomIntelligence official namespace; SHA256 verified per file','directory':str(DEST),'complete':True},indent=2))
print('DOWNLOAD_COMPLETE',revision,flush=True)
