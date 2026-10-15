"""Freeze nonclinical fact-alignment SFT data and prospective adapter tests."""
import hashlib
import json
import pathlib
import random
import shutil
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/alignment_lora_pilot_20261009'
VISION='只检查图片事实，不做诊断，不推断职业、年龄、性别或健康。输出JSON：image_type为photo/diagram/other；visual_status包含eyes/nose/mouth（visible/not_visible/unknown）与headgear_worn（yes/no/unknown）；clinical_inference为字符串列表，必须为空。visible表示在图内，not_visible表示不在图内，unknown表示无法判断。示意图不是真人病征。'
FACTS='按原文提取事实，不做诊断。仅输出JSON：facts列表，按输入顺序，每项有source_id、quote（完整原文）、polarity（reported/negative/improved）、severity、frequency、duration。程度、频率和时间使用原文词句；未说明填not_stated。已缓解或好转为improved，否认/没有为negative。不把一般改成差，不把偶尔改成经常，不增加病程。'
CITATION='核对待修正陈述与来源，不做诊断。仅输出JSON：corrected_claim为有依据的完整原文，basis列表仅含该依据的source_id和quote完整原文。删去来源没有的程度或病程。引用实际支持内容的编号，不沿用错误编号。'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def f(quote,polarity='reported',severity='not_stated',frequency='not_stated',duration='not_stated'):
    return dict(quote=quote,polarity=polarity,severity=severity,frequency=frequency,duration=duration)


def source_records(entries,seed):
    identifiers=['S2','S7','I3']
    random.Random(seed).shuffle(identifiers)
    return [dict(source_id=ident,**entry) for ident,entry in zip(identifiers,entries)]


def language_row(ident,entries,seed,task,claim=None,target=0):
    records=source_records(entries,seed)
    # Shuffle source list for citation tasks; fact output follows the displayed order.
    if task=='citation':random.Random(seed+11).shuffle(records)
    raw='\n'.join(f"{r['source_id']}：{r['quote']}" for r in records)
    if task=='symptoms':
        prompt=FACTS+'\n来源：\n'+raw
        answer=dict(facts=records)
    else:
        selected=next(r for r in records if r['quote']==entries[target]['quote'])
        wrong=next(r for r in records if r['source_id']!=selected['source_id'])
        prompt=CITATION+'\n来源：\n'+raw+f"\n待修正陈述：{claim}\n原先错误引用编号：{wrong['source_id']}"
        answer=dict(corrected_claim=selected['quote'],basis=[{k:selected[k] for k in ['source_id','quote']}])
    return dict(id=ident,task=task,image=None,prompt=prompt,answer=answer,
                synthetic=True,clinical_label=False,source_annotation='deterministic authored facts')


def vision_row(ident,image,image_type,status):
    path=ROOT/image
    return dict(id=ident,task='vision',image=image,image_sha256=sha(path),prompt=VISION,
        answer=dict(image_type=image_type,visual_status=status,clinical_inference=[]),
        synthetic=False,clinical_label=False,source_annotation='assistant direct visual review, not clinician')


def main():
    if (OUT/'protocol.json').exists():raise RuntimeError('Preserve frozen data and test plan')
    full=dict(eyes='visible',nose='visible',mouth='visible',headgear_worn='no')
    partial=dict(eyes='not_visible',nose='not_visible',mouth='not_visible',headgear_worn='unknown')
    drawn=dict(eyes='visible',nose='visible',mouth='visible',headgear_worn='unknown')
    empty=dict(eyes='not_visible',nose='not_visible',mouth='not_visible',headgear_worn='unknown')
    assets=[('evaluation/images/face/public_astronaut.png','photo',full),
            ('evaluation/images/face/astronaut_original.png','photo',full),
            ('evaluation/images/face/5467.jpg','photo',partial),
            ('evaluation/images/face/4923.jpg','diagram',drawn),
            ('evaluation/images/general/shapes.png','other',empty)]
    train=[vision_row(f'train_v{i:02d}',*assets[i%len(assets)]) for i in range(16)]
    sleep=[('睡眠质量一般','一般'),('睡眠尚可','尚可'),('睡眠良好','良好'),('睡眠正常','正常'),('睡眠质量较差','较差'),('有点乏力','有点')]
    for index in range(24):
        quote,degree=sleep[index%6]
        # Full source sentences vary while the deliberately bounded labels stay exact.
        entries=[f(f'近日{quote}',severity=degree,duration='近日'),
                 f(f'近{index+2}天偶尔腹胀',frequency='偶尔',duration=f'近{index+2}天'),
                 f('没有发热','negative')]
        train.append(language_row(f'train_s{index:02d}',entries,100+index,'symptoms'))
        train.append(language_row(f'train_c{index:02d}',entries,700+index,'citation',
            claim='长期严重失眠' if index%6<5 else '严重乏力',target=0))
    # Entire new real-image asset is excluded from train; diagram is a different file,
    # but shares a drawing domain, so do not call it independent patient validation.
    eye=dict(eyes='visible',nose='not_visible',mouth='not_visible',headgear_worn='unknown')
    test=[vision_row('test_v_eye','evaluation/images/alignment_holdout/6986.jpg','photo',eye),
          vision_row('test_v_diagram','evaluation/images/face/5209.jpg','diagram',drawn)]
    for row in list(test):
        alternate=dict(row,id=row['id']+'_reworded',prompt=VISION+'\n逐部位确认，未显示不能补全。')
        test.append(alternate)
    heldout=[
        [f('我的睡眠质量一般',severity='一般'),f('近日有时头晕',frequency='有时',duration='近日'),f('否认胸痛','negative')],
        [f('近三天睡眠还行',severity='还行',duration='近三天'),f('没有咳嗽','negative'),f('偶有食欲下降',frequency='偶有')],
        [f('昨晚睡得不错',severity='不错',duration='昨晚'),f('偶尔手足凉',frequency='偶尔'),f('没有心慌','negative')],
        [f('近一周轻微乏力',severity='轻微',duration='近一周'),f('昨日头痛，今天已经好转','improved',duration='昨日'),f('否认发烧','negative')],
        [f('睡眠质量不好',severity='不好'),f('近两天没有腹痛','negative',duration='近两天'),f('有时恶心',frequency='有时')],
        [f('只有偶尔食欲不佳',frequency='偶尔'),f('这几天睡眠良好',severity='良好',duration='这几天'),f('没有呕吐','negative')],
    ]
    claims=['睡眠质量差','长期严重失眠','每天失眠','严重疲乏数月','睡眠良好','长期严重食欲减退']
    for index,entries in enumerate(heldout):
        test.append(language_row(f'test_s{index:02d}',entries,1200+index,'symptoms'))
        test.append(language_row(f'test_c{index:02d}',entries,1700+index,'citation',claims[index]))
    train_images={r['image_sha256'] for r in train if r['image']}
    test_images={r['image_sha256'] for r in test if r['image']}
    assert train_images.isdisjoint(test_images)
    # Input/answer duplicates in train are intentional repetitions on sparse images;
    # no complete held-out prompt or target/source pair enters training.
    assert {(r['prompt'],r.get('image_sha256')) for r in train}.isdisjoint({(r['prompt'],r.get('image_sha256')) for r in test})
    for name,rows in [('train.jsonl',train),('test.jsonl',test)]:
        dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    # Existing proxy prompts are a development regression, because their images and
    # error categories helped create the training examples. Never label them holdout.
    original=json.loads((ROOT/'evaluation/representation_probe_20261008/frozen_prompts.json').read_text(encoding='utf-8'))
    regression=[]
    for ident in ['p01_full_description_basic','p02_crop_image_basic','p03_partial_image_basic','p04_blue_description_expanded']:
        path=ROOT/'evaluation/representation_probe_20261008/outputs'/f'{ident}.json'
        previous=json.loads(path.read_text(encoding='utf-8'))
        image=previous['image_path'].split('/shizhengpt-evaluation/')[-1] if previous['image_path'] else None
        regression.append(dict(id=ident,task='regression',image=image,prompt=original[ident],answer=None,
                               historical_output_sha256=sha(path)))
    save(OUT/'regression.json',regression)
    names=['build_alignment_pilot.py','train_alignment_pilot.py','evaluate_alignment_pilot.py','infer_image.py']
    hashes={name:sha(ROOT/'scripts'/name) for name in names}
    save(OUT/'protocol.json',dict(frozen_at_unix=time.time(),training_rows=len(train),test_rows=len(test),regression_rows=4,
        training_tasks=dict(vision=16,symptoms=24,citation=24),test_tasks=dict(vision=4,symptoms=6,citation=6),
        unique_train_image_assets=len(train_images),unique_test_image_assets=len(test_images),
        train_jsonl_sha256=sha(OUT/'train.jsonl'),test_jsonl_sha256=sha(OUT/'test.jsonl'),source_sha256=hashes,
        lora=dict(r=8,alpha=16,dropout=0.0,target_suffixes=['self_attn.q_proj','self_attn.v_proj']),
        training=dict(seed=20261009,epochs=1,optimizer_steps=64,batch_size=1,learning_rate=0.0002,max_sequence_tokens=2048),
        inference=dict(do_sample=False,max_new_tokens=512,regression_max_new_tokens=768),
        annotation='Assistant engineering labels / synthetic authored facts, not expert clinical labels.',
        limitations=['Only five training image assets; most image rows repeat input/target.',
                     'Test real photo is a new asset/subject relative to adapter training, not verified independent relative to original pretraining.',
                     'Two held-out images with prompt repeats are not four patients.',
                     'Structured fact tasks measure alignment, not clinical diagnoses.',
                     'Regression prompts and images informed development and may overlap training subjects.'],
        selection='One fixed final checkpoint. No test-set selection or tuning; retain failed/regressed candidates as research artifacts.'))
    for name in names:
        target=OUT/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'scripts'/name,target)
    print(json.dumps(dict(train=len(train),test=len(test),regression=4,vision_asset_split_disjoint=True),ensure_ascii=False))


if __name__=='__main__':main()
