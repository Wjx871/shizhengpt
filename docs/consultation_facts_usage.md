# 症状事实与追问检查入口 v3

推荐研究入口 `scripts/guarded_consultation_v31.py`；冻结实验入口 `scripts/guarded_consultation_v3.py` 与原 v2 入口保留用于对照。新版本不改变模型权重。

在服务器 `/root/private_data/shizhengpt-evaluation` 下运行：

```bash
.venv/bin/python scripts/guarded_consultation_v31.py \
  --image evaluation/images/face/astronaut_original.png \
  --symptoms-json /path/to/symptoms.json \
  --output evaluation/v3_consultation_result.json
```

纯文字输入省略 `--image`。症状文件为 UTF-8 JSON 字符串列表。

## 输出与检查

- `user_symptoms`、`source_symptoms` 保留原文。
- `symptom_facts` 保存有限词表内的症状词、否定/报告/已缓解状态和原文位置；程度、频率、时间保存在对应分句中。原文位置为 Python 字符索引、左闭右开。
- 槽位是确定性词表提取结果，不是专家确认的完整语义解析。没有识别的内容不会消失，仍保存在原文中；不能把提取失败解释为用户没有某症状。
- 解释检查引用来源是否支持出现的症状词；仅有否定或已缓解症状的来源不能作为当前症状解释依据。新增程度、频率和病程有部分检查。
- 问句逐个检查症状前的询问条件。“是否有胸痛？”可以用于确认，“胸痛是否持续？”会预设胸痛存在；不能用全句包含“是否”来跳过检查。
- 视觉部分增加范围和低分辨率规则，阻止服装/头盔描述、部分正常/无异常判断和细节断言。这是限制输出范围，尚无独立视觉核验模型。
- 最多重试一次；被拒绝的原始回答仍保存在 attempts 中，只用于研究审阅。检查后的字段仍需人工复核。
- v3.1 完整分析失败时，对最后的完整候选追问单独检查；通过则只保留追问，`possible_explanations` 为空，状态为 `questions_only_after_explanation_rejection_requires_review`。不要把这个状态展示为完整分析成功。被拒绝的解释继续保存在原始记录中。

`check_consultation_facts.py` 为开发用构造案例检查。`evaluate_consultation_facts.py --freeze` 冻结新实验输入和源码；没有 `--freeze` 时在服务器运行成对 v2/v3 比较。实验禁止覆盖旧结果，运行前核验源码哈希。

`audit_consultation_facts.py` 生成逐条件 CSV、核对结果 JSON 和供人工复核的完整回答 `review.json`。规则标记不等于专家标注的幻觉率；新合成文本条件也不等于独立临床患者验证集。

`check_question_fallback.py` 验证追问单独保留逻辑；`review_question_fallback.py` 从冻结 v3 回答生成独立回放文件，禁止覆盖旧结果，不新增模型推理。完整实验及误拦截情况见 [consultation_facts_report.md](consultation_facts_report.md)。
