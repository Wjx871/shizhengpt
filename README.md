# ShizhenGPT-7B-VL 部署与望诊 / 问诊评测

在 K100_AI 64 GB、DTK 25.04.2、PyTorch 2.5.1 上验证官方 BF16 模型，研究图片、文本、图文联合推理。当前网页默认使用基础权重和谨慎回答流程；曾进行小规模 LoRA 代理实验，未替换默认模型，没有引入 RAG 或量化。

## 当前入口（2026-10-15 备份）

- 网页与服务：`web/index.html`、`scripts/chat_server.py`。见 [启动说明](docs/chat_web.md)。
- 默认谨慎回答：`scripts/cautious_chat.py`。见 [能力范围](docs/cautious_chat_usage.md) 和 [行为测试结果](docs/cautious_chat_report_20261013.md)。
- 可切换原始模型对照；信息不足时弃答是有限工程策略，不能保证零幻觉或诊断正确。
- 仓库备份代码、配置、文档及合成条件实验记录。模型权重、原始图片、舌诊数据、缓存与临时压缩包保持本地保存，见 [备份范围](docs/github_backup_20261015.md)。

下文保留初始部署和各历史实验入口，反映各阶段的研究设置。

### 谨慎回答 v2（独立优化分支）

`optimization/cautious-v2` 在固定备份后继续开发。新增礼貌前缀后的症状识别、假设语句与实际报告区分、用药问题的独立弃答、口语不适的补充追问，以及由程序测量的图片信息。默认入口为 `scripts/cautious_chat_v2.py`，v1 保留用于对照。详见 [v2 结果与限制](docs/cautious_chat_v2_report_20261015.md)。

服务器项目目录：`/root/private_data/shizhengpt-evaluation`。模型目录：`/root/private_data/models/ShizhenGPT-7B-VL`。

配置见 `configs/inference.yaml`；部署记录见 `docs/deployment.md`；环境实测见 `docs/environment.md`；评测结论见 `docs/evaluation_report.md`。

在服务器项目目录运行：

```bash
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/infer_image.py --image evaluation/images/tongue/3249.jpg --prompt '请客观描述舌象。'
.venv/bin/python scripts/infer_image.py --prompt '我最近容易疲劳，请给出需要追问的问题。'
.venv/bin/python scripts/batch_evaluate.py
.venv/bin/python scripts/benchmark.py
```

原始输出保存在 `evaluation/outputs/`，结果表位于 `evaluation/results/results.csv`。显存单位为 bytes，延迟单位为秒。人工评分列使用 1–5，未经人工评分时留空；这些评分不代表医学诊断准确率。

官方教材图片、图示与公开照片不代表真实临床采集分布。公开人物照片没有医学标签，不能称为“健康人”对照。图文联合中的症状是构造的实验条件，不是照片人物的真实自述。路线 A/B/C 依据实际结果选择。

保留系统 DCU 定制 PyTorch。项目 venv 复用系统依赖，仅补充缺少的 qwen-vl-utils。不要运行 torch 升级命令。

## Route B 面部质控原型

在服务器项目目录运行：

```bash
.venv/bin/python scripts/face_quality.py --image evaluation/images/face/astronaut_original.png --output evaluation/route_b/qc.json
.venv/bin/python scripts/infer_face_qc.py --image evaluation/images/face/astronaut_original.png --output evaluation/route_b/face_result.json
.venv/bin/python scripts/route_b_compare.py
.venv/bin/python scripts/summarize_route_b.py
```

`infer_face_qc.py` 在质控拒绝时不加载大模型。通过质控仅允许技术描述，模型输出始终标记为未经核验；程序的测量字段和原始症状保留在独立字段中。可用 `--symptoms-json` 传入 UTF-8 JSON 字符串列表。这个入口面向正脸照片；一般文本问诊和舌图继续使用 `infer_image.py`。

比较实验为了观察模型是否服从质控，仍对不合格图片生成答案，并明确标记为绕过门控的实验输出。正式入口不会这样运行。报告见 `docs/route_b_report.md`，结果见 `evaluation/route_b/`。像素采样区不是皮肤分割，颜色没有校准，不能用于面色辨证。
