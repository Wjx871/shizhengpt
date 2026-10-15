"""Factorial image/text and limited/expanded synthetic history probe.

Descriptions are assistant visual reviews, NOT doctor-authored clinical findings.
No diagnostic truth labels; one raw model call per condition, no wrappers/retries.
"""
import hashlib
import json
import pathlib
import shutil
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/representation_probe_20261008'
ARMS = [('image_basic', 'image', False), ('description_basic', 'description', False),
        ('description_expanded', 'description', True), ('image_expanded', 'image', True)]
PROMPT = '''这是研究用合成问诊输入，图片人物与症状没有真实病例关联。仅依据本次提供的信息回答。
把图像或图像描述中的可见信息与用户症状分开。不得从照片猜测年龄、性别、职业、健康状态、脉象、舌象或真实肤色；未显示的部位不能补全。
如果图像描述代替图片，应明确观察依据是提供的描述，不要声称自己看到了图片。不得将信息不足写成确诊，不提供药物、剂量或治疗。
用户症状和补充问诊信息保持原字串；可能解释只能引用实际提供的S/I编号，不能根据服装或面色推断脏腑病变。
追问不能预设未知症状，也应考虑补充问诊信息；仍未明确的内容可以继续询问。
只输出一个JSON对象，字段为：
visual_status：对象，eyes/nose/mouth各为visible、not_visible或unknown，headgear_worn为yes、no或unknown；判定仅针对画面可见区域或提供的描述。
visual_observations：最多2条简短字符串；
user_symptoms：症状原文字符串列表；additional_history：实际提供的补充信息原文列表，未提供则为空；
possible_explanations：最多2个对象，每个只有basis_ids（S/I编号列表）及text（需要核实的可能解释字符串）；信息不足可为空；
questions：最多3条追问字符串；limitations：最多2条字符串。不要填写格式占位答案。'''


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def cases():
    fatigue = ['最近容易疲劳', '手脚较凉', '睡眠质量一般', '偶尔食欲不佳']
    fatigue_extra = ['这些不适约一周，工作忙时更明显', '最近每晚约睡6小时，有时熬夜',
                     '没有发热、胸痛或心慌', '休息后疲劳会减轻', '未提供既往病史、用药史、舌象、脉象或检查结果']
    abdomen = ['最近一周偶有腹胀', '没有恶心和呕吐', '食欲尚可']
    abdomen_extra = ['腹胀主要在吃得较多后出现，排气后减轻', '排便每天一次，未报告便秘或腹泻',
                     '没有腹痛或发热', '近期吃饭较快，未提供体重变化', '未提供既往病史、用药史、舌象、脉象或检查结果']
    expected_full = dict(eyes='visible', nose='visible', mouth='visible', headgear_worn='no')
    expected_partial = dict(eyes='not_visible', nose='not_visible', mouth='not_visible', headgear_worn='unknown')
    result = [
        dict(id='p01_full', scenario='synthetic_fatigue', image='evaluation/images/face/public_astronaut.png',
             description='公开人物静态照片。人物头部和上半身可见，头部朝向画面前方，眼睛、鼻子和嘴部可见，表情为微笑。头发可见，头上没有戴头盔；画面下方另有一个头盔。服装与背景不提供医学信息。照片未作医学校准，皮肤细微异常无法据此确认。',
             symptoms=fatigue, expanded_history=fatigue_extra, expected_visual_status=expected_full),
        dict(id='p02_crop', scenario='synthetic_fatigue', image='evaluation/images/face/astronaut_original.png',
             description='155×172像素的静态头部裁剪图。人物面部朝向画面前方，可见眼睛、鼻子、嘴部及微笑表情，头发可见，头部没有戴头盔。细节分辨率有限，不能据此确认细微皮肤异常，也不能排除异常。照片颜色未校准。',
             symptoms=fatigue, expanded_history=fatigue_extra, expected_visual_status=expected_full),
        dict(id='p03_partial', scenario='synthetic_abdominal_bloating', image='evaluation/images/face/5467.jpg',
             description='局部面部照片，画面主要显示一侧脸颊及耳部，照片中有两个黑色箭头，左部有明显明亮区域。眼睛、鼻子和嘴部不在画面内，头顶部不在画面内，因此不能判断是否戴头盔。不能描述对侧脸颊或补全未显示的五官，箭头区域的医学性质未知，照片颜色未校准。',
             symptoms=abdomen, expanded_history=abdomen_extra, expected_visual_status=expected_partial),
        dict(id='p04_blue', scenario='synthetic_fatigue', image='evaluation/images/face/astronaut_blue.png',
             description='155×172像素的静态头部裁剪图，照片经过偏蓝成像变换。人物面部朝向画面前方，可见眼睛、鼻子、嘴部及微笑表情，头发可见，头部没有戴头盔。不能将变换后的照片颜色当作人物真实肤色；细节分辨率有限，不能确认或排除细微皮肤异常。',
             symptoms=fatigue, expanded_history=fatigue_extra, expected_visual_status=expected_full),
    ]
    for case in result:
        case['image_sha256'] = hashlib.sha256((ROOT / case['image']).read_bytes()).hexdigest()
        case['description_source'] = 'assistant_direct_visual_review_non_clinician'
        case['clinical_ground_truth_available'] = False
        case['history_is_synthetic'] = True
    return result


def prompt_for(case, mode, expanded):
    text = PROMPT + '\n用户原始症状：\n' + '\n'.join(f'S{i+1}：{s}' for i, s in enumerate(case['symptoms']))
    history = case['expanded_history'] if expanded else []
    text += '\n补充问诊信息：\n' + ('\n'.join(f'I{i+1}：{s}' for i, s in enumerate(history)) if history else '未提供。')
    text += '\n视觉输入：' + ('已附本次原始测试图。' if mode == 'image' else '没有附图，仅提供以下人工核对的图像描述：' + case['description'])
    return text


def freeze():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Do not overwrite frozen protocol')
    manifest = cases()
    save(OUT / 'manifest.json', manifest)
    names = ['evaluate_representation_probe.py', 'infer_image.py']
    hashes = {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() for name in names}
    save(OUT / 'protocol.json', dict(frozen_at_unix=time.time(), cases=4, conditions=16, arms=ARMS,
        source_sha256=hashes, output_cap=768, max_retries=0, wrappers=False,
        primary_comparisons=['image_basic vs description_basic: change visual representation only',
                             'description_basic vs description_expanded: change history only',
                             'image_basic vs image_expanded: change history only'],
        limitations=['No doctor descriptions or clinical diagnostic reference labels.',
                     'Manual text reveals facts and removes uncertainty: oracle proxy, not equal visual difficulty.',
                     'Three image conditions are the same public person; there are only two synthetic symptom scenarios.',
                     'Expanded history is additional synthetic information, not complete clinical records.']))
    for name in names:
        target = OUT / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'scripts' / name, target)
    save(OUT / 'frozen_prompts.json', {f"{case['id']}_{arm}": prompt_for(case, mode, expanded)
        for case in manifest for arm, mode, expanded in ARMS})
    print('FROZEN 4 image conditions x 4 arms = 16 raw model calls; clinical proxy only')


def run():
    from infer_image import Engine
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() == expected, name
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    prompts = json.loads((OUT / 'frozen_prompts.json').read_text(encoding='utf-8'))
    engine = Engine()
    start = time.time()
    count = 0
    for case in manifest:
        image = str(ROOT / case['image'])
        assert hashlib.sha256(pathlib.Path(image).read_bytes()).hexdigest() == case['image_sha256']
        for arm, mode, expanded in ARMS:
            ident = f"{case['id']}_{arm}"
            path = OUT / 'outputs' / f'{ident}.json'
            if path.exists(): raise RuntimeError('Preserve historical inference')
            print('START', ident, flush=True)
            result = engine.infer(prompts[ident], image if mode == 'image' else None, max_new_tokens=protocol['output_cap'])
            result.update(case_id=case['id'], arm=arm, visual_mode=mode, expanded_history=expanded,
                          evaluation_config=engine.config, clinical_accuracy_not_measured=True)
            save(path, result)
            count += 1
            print('DONE', ident, 'tokens', result['output_tokens'], 'truncated', result['truncated'], flush=True)
    save(OUT / 'run.json', dict(model_inferences=count, elapsed_sec=time.time()-start, config=engine.config,
        source_sha256=protocol['source_sha256'], model_load_sec=engine.load_sec, conditions=16,
        all_single_call=True, no_retries=True, no_guarded_wrapper=True))
    print('COMPLETE', count, flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    freeze() if args.freeze else run()
