import json,os,pathlib,platform,shutil,subprocess,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def command(args):
 try:return subprocess.run(args,capture_output=True,text=True,timeout=30).stdout
 except Exception as e:return type(e).__name__+': '+str(e)
r={'python':platform.python_version(),'torch':torch.__version__,'cuda_field':torch.version.cuda,'hip':torch.version.hip,'accelerator_available':torch.cuda.is_available(),'dtk_real_path':os.path.realpath('/opt/dtk'),'disk':command(['df','-h']),'memory_host':command(['free','-h']),'device_usage':command(['rocm-smi']),'hipcc':command(['hipcc','--version']),'cpu':command(['lscpu'])}
for name in ['memory.max','memory/memory.limit_in_bytes','cpu.max','cpu/cpu.cfs_quota_us','cpu/cpu.cfs_period_us']:
 p=pathlib.Path('/sys/fs/cgroup')/name
 if p.exists():r[name]=p.read_text().strip()
if torch.cuda.is_available():
 r['devices']=[{'name':torch.cuda.get_device_name(i),'properties':str(torch.cuda.get_device_properties(i)),'free_total_bytes':torch.cuda.mem_get_info(i)} for i in range(torch.cuda.device_count())];r['bf16_reported']=torch.cuda.is_bf16_supported()
(ROOT/'docs/environment.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));(ROOT/'docs/environment.md').write_text('# SERVER ENVIRONMENT REPORT\n\n```json\n'+json.dumps(r,ensure_ascii=False,indent=2)+'\n```\n');print(json.dumps(r,ensure_ascii=False,indent=2))
