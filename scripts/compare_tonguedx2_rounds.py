"""Validate a paired prompt comparison and save a compact Chinese report."""
import csv
import json
import pathlib
import re
import statistics
from tonguedx2_o import FEATURES, parse, sha, save

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIRST = ROOT/'evaluation/tonguedx2_o_20261008'
SECOND = ROOT/'evaluation/tonguedx2_o_round2_20261008'


def expresses_all_unknown(text):
    """Qualitative diagnostic only; never coerce these values into scored predictions."""
    stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        obj = json.loads(stripped)
        values = obj.get('features', obj)
        return set(FEATURES).issubset(values) and all(values[f] is None or values[f] == 'null' for f in FEATURES)
    except (ValueError, TypeError, AttributeError):
        lines = stripped.splitlines()
        matches = [re.fullmatch(r'([A-Za-z]+):\s*null', line.strip()) for line in lines]
        return len(matches) == 8 and all(matches) and {m.group(1) for m in matches} == set(FEATURES)


def inspect(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    labels = {r['id']: r for r in json.loads((folder/'labels.json').read_text(encoding='utf-8'))}
    prompt = (folder/'prompt.txt').read_text(encoding='utf-8')
    predictions, raw, bad = {}, {}, []
    correct = valid = ones = zeros = 0
    for case in manifest:
        r = json.loads((folder/'outputs'/f"{case['id']}.json").read_text(encoding='utf-8'))
        assert r['status'] == 'ok' and not r['truncated']
        assert r['image_sha256'] == case['sha256'] and r['prompt'] == prompt
        obj, error = parse(r['raw_output'])
        if error:
            bad.append(case['id'])
        values = obj['features'] if obj else {f: None for f in FEATURES}
        predictions[case['id']] = values
        raw[case['id']] = r
        for f in FEATURES:
            if values[f] is not None:
                valid += 1
                ones += values[f] == 1
                zeros += values[f] == 0
                correct += values[f] == labels[case['id']][f]
    control = json.loads((folder/'outputs/no_image.json').read_text(encoding='utf-8'))
    assert control['status'] == 'ok' and not control['image_path'] and control['image_grid_thw'] is None
    assert control['prompt'] == prompt
    obj, error = parse(control['raw_output'])
    control_pass = bool(obj and all(v is None for v in obj['features'].values()))
    return dict(n=len(manifest), valid=valid, correct=correct, ones=ones, zeros=zeros,
        invalid_ids=bad, format_valid=len(manifest)-len(bad), coverage=valid/(len(manifest)*8),
        conditional_agreement=correct/valid if valid else None,
        strict_agreement=correct/(len(manifest)*8),
        median_latency_sec=statistics.median(r['latency_sec'] for r in raw.values()),
        no_image_pass=control_pass, no_image_raw=control['raw_output'],
        no_image_expresses_all_unknown=expresses_all_unknown(control['raw_output']),
        image_answers_expressing_all_unknown=sum(expresses_all_unknown(r['raw_output']) for r in raw.values()),
        predictions=predictions, raw=raw, manifest=manifest, labels=labels,
        prompt_sha256=sha(folder/'prompt.txt'))


def main():
    first, second = inspect(FIRST), inspect(SECOND)
    assert first['manifest'] == second['manifest'] and first['labels'] == second['labels']
    assert first['prompt_sha256'] != second['prompt_sha256']
    for ident in first['raw']:
        assert first['raw'][ident]['evaluation_config'] == second['raw'][ident]['evaluation_config']
    source = list(csv.DictReader((ROOT.parent/'TongueDx2原始标签总表_按编号排序.csv').open(encoding='utf-8-sig', newline='')))
    for ident, label in second['labels'].items():
        original = [r for r in source if r['id'] == ident]
        assert len(original) == 1
        assert all(int(original[0][f]) == label[f] for f in FEATURES)
    rows = []
    for ident in first['raw']:
        for f, name in FEATURES.items():
            p1, p2 = first['predictions'][ident][f], second['predictions'][ident][f]
            rows.append(dict(id=ident, feature=f, name=name, label=second['labels'][ident][f],
                             round1=p1, round2=p2, changed=p1 != p2))
    with (SECOND/'paired_predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compact = {name: {k: v for k, v in obj.items() if k not in ('predictions','raw','manifest','labels')}
               for name, obj in [('round1', first), ('round2', second)]}
    compact['checks'] = dict(same_48_originals=True, same_inference_config=True,
        same_labels_checked_against_source=True, prompts_different=True,
        compared_feature_positions=len(rows), changed_predictions=sum(r['changed'] for r in rows),
        label_encoding_confirmed=False)
    save(SECOND/'comparison.json', compact)
    fmt = lambda x: f'{x:.1%}' if x is not None else '无有效判断'
    table = ['| 项目 | 第一轮 | 第二轮 |', '|---|---:|---:|',
        f"| 结构化推理成功 | {first['n']}/48 | {second['n']}/48 |",
        f"| 八项数值均可解析的图片 | {first['format_valid']}/48 | {second['format_valid']}/48 |",
        f"| 有效 0/1 判断数 | {first['valid']}/384 | {second['valid']}/384 |",
        f"| 有效判断覆盖率 | {fmt(first['coverage'])} | {fmt(second['coverage'])} |",
        f"| 有效判断中与标签一致 | {first['correct']}/{first['valid']} | {second['correct']}/{second['valid']} |",
        f"| 有效判断的暂定标签符合率 | {fmt(first['conditional_agreement'])} | {fmt(second['conditional_agreement'])} |",
        f"| 输出 1 / 输出 0 | {first['ones']} / {first['zeros']} | {second['ones']} / {second['zeros']} |",
        f"| 无图 JSON 可解析且八项均为 null | {'通过' if first['no_image_pass'] else '未通过'} | {'通过' if second['no_image_pass'] else '未通过'} |",
        f"| 图片推理中位耗时 | {first['median_latency_sec']:.2f} 秒 | {second['median_latency_sec']:.2f} 秒 |"]
    feature_table = ['| 特征 | 第一轮有效数 / 暂定一致数 | 第二轮有效数 / 暂定一致数 |', '|---|---:|---:|']
    for f, name in FEATURES.items():
        entries = []
        for obj in (first, second):
            ps = [v[f] for v in obj['predictions'].values()]
            count = sum(v is not None for v in ps)
            agree = sum(values[f] is not None and values[f] == obj['labels'][ident][f]
                        for ident, values in obj['predictions'].items())
            entries.append(f'{count} / {agree}')
        feature_table.append(f'| {name} | {entries[0]} | {entries[1]} |')
    prompt = (SECOND/'prompt.txt').read_text(encoding='utf-8')
    text = '# O 原图第二轮：提示词对照\n\n'
    text += '2026-10-08。第二轮实际完成 48 张 O 原图与 1 次无图对照，共 49 次推理；没有使用 S。与第一轮使用相同图片、模型权重、推理配置及标签。本次改变了提示词，因此是同一批图片的提示词对照，不是新增患者验证。\n\n'
    text += '第二轮采用简化提示：只输出顶层八个键，区域可以看清而没有特征时填 0，仅在该区域看不清时填 null，并明确无图与模糊的区别。保留 null，没有强迫模型必须选择 0/1。\n\n'
    text += '\n'.join(table) + '\n\n'
    text += f"第二轮 {second['format_valid']} 张可解析图片回答均没有给出有效 0/1；共有 {second['image_answers_expressing_all_unknown']}/48 张在文字含义上八项均表示 null。不能把这种全部拒答理解为视觉识别正确或模型确认没有异常。\n\n"
    text += '第二轮其余 14 张的格式问题经逐份复核：9 张用 true/false 表达了共 17 项判断，2 张在特征值中输出中文描述，2 张输出字符串 "f"，1 张按逐行文字列举而非 JSON。严格评分没有把布尔值、描述或 "f" 自动改成整数。因此“0 个合规 0/1 判断”不等于所有原始回答都完全没有舌象信息。允许无歧义的布尔值转换可提取 17 项（4.4% 的特征位置），仍远低于第一轮的 63 项。中文描述仅供定性复核，不猜测数值。\n\n'
    if second['no_image_expresses_all_unknown']:
        text += '第二轮无图回答的八项在含义上均表示不确定，没有再次出现第一轮的舌淡阳性断言；但字符串 "null" 与 JSON 的 null 不是同一类型，因此格式检查仍不通过。这是单次提示词对照中的现象，不能宣称无图编造已被彻底解决。\n\n'
    text += '标签编码仍未确认，所有符合率暂按 1=特征存在、0=不存在计算，不能称为医学准确率。覆盖率提升仅说明给出的有效数值更多，不能单独证明判断更准确；两轮有效判断集合不同，条件符合率也有选择偏差。无图对照每轮只测一次，通过不意味着彻底消除幻觉。\n\n'
    text += '\n'.join(feature_table) + '\n\n'
    text += '## 第二轮无图原始回答\n\n```text\n' + second['no_image_raw'] + '\n```\n\n'
    text += '## 第二轮提示词\n\n```text\n' + prompt + '\n```\n\n'
    text += '## 验证与证据\n\n已核对 48 张原图哈希、两轮推理配置、标签源表匹配，以及 384 个特征位置的逐项变化。两轮原始回答分别保留，无覆盖。第二轮 49 次运行成功，无回答截断。\n\n'
    text += '[第二轮逐图结果](../evaluation/tonguedx2_o_round2_20261008/predictions.csv) · [逐项配对结果](../evaluation/tonguedx2_o_round2_20261008/paired_predictions.csv) · [比较汇总](../evaluation/tonguedx2_o_round2_20261008/comparison.json)\n'
    report = ROOT/'docs/tonguedx2_o_round2_report.md'
    report.write_text(text, encoding='utf-8')
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
