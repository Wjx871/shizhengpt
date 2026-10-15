import io,json,pathlib
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter,ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[1]
face=Image.open(ROOT/'evaluation/images/face/public_astronaut.png').convert('RGB').crop((150,20,305,192))
variants={'original':face,'bright':ImageEnhance.Brightness(face).enhance(1.15),'dark':ImageEnhance.Brightness(face).enhance(0.85),'blur':face.filter(ImageFilter.GaussianBlur(0.8))}
arr=np.asarray(face).astype('float32')
variants['yellow']=Image.fromarray(np.clip(arr*np.array([1.08,1.02,0.88]),0,255).astype('uint8'))
variants['blue']=Image.fromarray(np.clip(arr*np.array([0.88,1.00,1.08]),0,255).astype('uint8'))
buf=io.BytesIO();face.save(buf,format='JPEG',quality=35);buf.seek(0);variants['jpeg']=Image.open(buf).convert('RGB')
for name,im in variants.items():im.save(ROOT/f'evaluation/images/face/astronaut_{name}.png')
(ROOT/'evaluation/results/lighting_transformations.json').write_text(json.dumps({'source':'public_astronaut.png','manual_crop':[150,20,305,192],'brightness':[1.15,0.85],'yellow_rgb_multiplier':[1.08,1.02,0.88],'blue_rgb_multiplier':[0.88,1,1.08],'gaussian_blur_radius':0.8,'jpeg_quality':35,'geometry_unchanged':True},indent=2))
cases=[]
def add(ident,cat,prompt,image=None,repeats=3,suffix=''):
 cases.append({'test_id':ident,'category':cat,'prompt_id':prompt,'image_path':image or '', 'repeats':repeats,'suffix':suffix})
for ident in ['3249','3376','3263','3912']:add('tongue_'+ident,'tongue','tongue','evaluation/images/tongue/'+ident+'.jpg')
for name in variants:add('face_'+name,'face' if name=='original' else 'lighting_robustness','face',f'evaluation/images/face/astronaut_{name}.png')
add('face_partial','face','face','evaluation/images/face/5467.jpg')
add('face_diagram','negative_control','face','evaluation/images/face/4923.jpg')
add('text_symptoms','text','text');add('text_insufficient','text','text_insufficient')
for ident,img in [('no_image',None),('face','evaluation/images/face/astronaut_original.png'),('tongue_white','evaluation/images/tongue/3249.jpg'),('tongue_yellow','evaluation/images/tongue/3376.jpg'),('unrelated','evaluation/images/general/shapes.png')]:add('fusion_'+ident,'fusion_ablation','fusion',img)
for ident,img in [('shapes','evaluation/images/general/shapes.png'),('face','evaluation/images/face/astronaut_original.png'),('tongue','evaluation/images/tongue/3249.jpg')]:add('hallucination_'+ident,'hallucination','hallucination',img)
rows=json.load(open(ROOT.parent/'datasets/TCM-Vision-Benchmark/TCM-Vision_Benchmark.json'));selected=json.load(open(ROOT/'evaluation/selected_candidates.json'))
label_targets=['3249','3376','3263','3912','3245','3248']
for r in selected:
 ident=pathlib.Path(r['image'][0]).stem
 if ident in label_targets:
  pid='mc_'+ident;prompt=r['question']+'\n'+'\n'.join(k+'. '+v for k,v in r['options'].items())+'\n\u53ea\u56de\u7b54\u9009\u9879\u5b57\u6bcd\uff0c\u4e0d\u8981\u89e3\u91ca\u3002'
  (ROOT/'evaluation/prompts'/f'{pid}.txt').write_text(prompt,encoding='utf-8');add(pid,'official_visual_mc',pid,'evaluation/images/tongue/'+ident+'.jpg',1)
(ROOT/'evaluation/manifest.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2));print('MANIFEST',len(cases),'cases',sum(c['repeats'] for c in cases),'runs')
sheet=Image.new('RGB',(155*7,192),'white');d=ImageDraw.Draw(sheet)
for i,(name,im) in enumerate(variants.items()):sheet.paste(im,(i*155,0));d.text((i*155,175),name,fill='black')
sheet.save(ROOT/'evaluation/results/lighting_contact_sheet.jpg')
