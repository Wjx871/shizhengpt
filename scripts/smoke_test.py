import json
from PIL import Image,ImageDraw
from infer_image import Engine,ROOT
path=ROOT/'evaluation/images/general/shapes.png'
im=Image.new('RGB',(512,384),'white');d=ImageDraw.Draw(im);d.rectangle((40,80,190,230),fill='red');d.ellipse((290,80,440,230),fill='blue');im.save(path)
e=Engine();result=e.infer('请描述这张图片中的主要内容。',str(path),128)
result.update(model_load_sec=e.load_sec,load_allocated_bytes=e.load_allocated,load_reserved_bytes=e.load_reserved)
(ROOT/'evaluation/outputs/smoke_test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
