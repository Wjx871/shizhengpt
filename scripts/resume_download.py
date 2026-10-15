import concurrent.futures,hashlib,json,os,pathlib,urllib.request,time
ROOT=pathlib.Path(__file__).resolve().parents[1];DEST=ROOT.parent/'models/ShizhenGPT-7B-VL'
manifest=json.load(open(ROOT/'docs/sources/modelscope_manifest.json'));files=[f for f in manifest['Data']['Files'] if f['Type']=='blob' and not f['Path'].startswith('.')]
def hashfile(p):
 h=hashlib.sha256()
 with open(p,'rb') as inp:
  for b in iter(lambda:inp.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
for f in files:
 target=DEST/f['Path']
 if target.exists():
  assert target.stat().st_size==f['Size'];assert hashfile(target)==f['Sha256'];print('VERIFIED_EXISTING',target.name,flush=True);continue
 partial=target.with_suffix(target.suffix+'.partial');prefix=partial.stat().st_size if partial.exists() else 0
 u=f'https://modelscope.cn/api/v1/models/FreedomIntelligence/ShizhenGPT-7B-VL/repo?Revision={f["Revision"]}&FilePath={f["Path"]}'
 ranges=[(start,min(start+128*1024*1024,f['Size'])-1) for start in range(prefix,f['Size'],128*1024*1024)]
 def segment(pair):
  start,end=pair;seg=DEST/(target.name+f'.segment-{start}')
  for attempt in range(4):
   try:
    req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=60) as response,open(seg,'wb') as out:
     assert response.status==206,response.status
     assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{f["Size"]}'
     while True:
      block=response.read(8*1024*1024)
      if not block:break
      out.write(block)
    assert seg.stat().st_size==end-start+1;print('SEGMENT_OK',target.name,start,end,flush=True);return seg
   except Exception as e:print('SEGMENT_RETRY',start,attempt,type(e).__name__,flush=True)
  raise RuntimeError('range failed')
 print('RESUMING',target.name,prefix,f['Size'],len(ranges),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:segments=list(pool.map(segment,ranges))
 with open(partial,'ab') as out:
  for seg in segments:
   with open(seg,'rb') as inp:
    for b in iter(lambda:inp.read(8*1024*1024),b''):out.write(b)
 assert partial.stat().st_size==f['Size'];assert hashfile(partial)==f['Sha256'];partial.replace(target)
 for seg in segments:seg.unlink()
 print('VERIFIED_RESUMED',target.name,flush=True)
(ROOT/'docs/sources/download_complete.json').write_text(json.dumps({'directory':str(DEST),'complete':True,'download_source':'ModelScope official FreedomIntelligence namespace','all_files_sha256_verified':True},indent=2))
print('DOWNLOAD_COMPLETE',flush=True)
