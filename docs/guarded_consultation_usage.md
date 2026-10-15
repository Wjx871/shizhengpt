# 面部图像与文本问诊保护入口

入口：`scripts/guarded_consultation.py`。这是现有原始 BF16 模型外围的程序优化，没有训练或修改权重。

## 实际处理

1. 原始症状写入程序字段 `user_symptoms` 和编号表 `source_symptoms`，不依赖模型复述。
2. 图片先走现有 `face_quality.analyze`；没有图片或图片质控拒绝时，不调用视觉模型。存在症状时继续文字分析。
3. 通过质控的图片先做技术观察。仅允许规定字段，检查部分已知人口学、病理、舌象、脉象、不可见身体区域和动态状态断言。
4. 第二步不输入图片，使用症状原文编号、视觉处理状态及通过有限检查的技术描述。可能解释必须引用已存在的 S 编号，不允许根据未经校准照片推断面色证候；图片观察没有被证明正确。
5. 检查已知症状改写模式、确定性断言、未知编号、新增视觉断言及输出格式。违规重试一次；仍失败时保留症状原文并返回检查失败原因，不展示被拒绝的回答为有效分析。
6. 原始模型回答保存在 `visual_attempts` / `analysis_attempts` 中供研究审阅。正式展示应使用通过检查后的字段，不能将失败的原始回答直接呈现为有效结论。

追问允许询问尚不明确的症状，例如“是否存在失眠或多梦？”，但不能将用户的“睡眠质量一般”预设为“睡眠质量差”。这仍是有限模式检查，不能保证所有问句都没有错误前提。

## 使用

在原服务器项目目录 `/root/private_data/shizhengpt-evaluation` 执行：

```bash
# 原图 + UTF-8 JSON 字符串列表形式的症状
.venv/bin/python scripts/guarded_consultation.py \
  --image evaluation/images/face/astronaut_original.png \
  --symptoms-json /path/to/symptoms.json \
  --output evaluation/consultation_result.json

# 只有文字，无视觉调用
.venv/bin/python scripts/guarded_consultation.py \
  --symptoms-json /path/to/symptoms.json \
  --output evaluation/text_consultation_result.json

# 图片质控拒绝且没有症状时，不加载模型
.venv/bin/python scripts/guarded_consultation.py \
  --image evaluation/images/face/5467.jpg \
  --output evaluation/rejected_face_result.json
```

症状文件示例：

```json
["最近容易疲劳", "手脚较凉", "睡眠质量一般", "偶尔食欲不佳"]
```

`--max-retries` 默认 1，允许 0–2。模型仅在真正需要生成时加载。

## 适用范围与局限

这是技术观察及辅助追问原型。没有专家确认的视觉事实、没有颜色校准，也没有疾病或证候准确率证明。原来的面色理论解释被限制，不能将减少输出内容解读为模型视觉能力提高。

规则主要覆盖此前出现的错误模式，不能检测所有近义改写、暗含前提的追问、照片事实误判或不合理理论解释。正脸眼部检测不等于逐区域可见性分割，不能保证额头、唇部等没有遮挡。通过格式和关键词检查仅表示未触发这些规则，所有保留的模型内容仍需复核。

拒答和格式失败会造成有用信息损失，因此必须同时报告最终保留的内容、拒答次数、重试次数和模型原始错误。程序保存症状原文是设计保证，不是模型症状理解能力的准确率。

## 检查与对照

`check_consultation_guards.py` 用人工构造的错误输出验证拦截、重试、退回和零视觉调用逻辑；它不计算临床幻觉率。

`evaluate_guarded_consultation.py` 对同一组 12 条件运行旧入口提示和新流程。旧组直接生成观察、解释和问诊，新组加入质控、分步、规则及重试；属于组合改动，不能单独归因于某一个模块。

图片是已有公开样本，症状是合成实验输入。新增症状条件和公开人物完整照片的新组合，不是独立临床患者测试集；不得宣称已验证跨患者泛化。

本次实测见 [guarded_consultation_report.md](guarded_consultation_report.md)。`recheck_guarded_consultation.py` 对保留候选使用修正后的规则重新检查，不新增模型生成；复核前结果和旧拒绝原因保留。`audit_guarded_consultation.py` 核对最终结果并生成对照 CSV/JSON，不进行模型推理。
