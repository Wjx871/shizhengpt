# Deployment record

Project: `/root/private_data/shizhengpt-evaluation` (persistent NFS).
Model: `/root/private_data/models/ShizhenGPT-7B-VL` (persistent NFS).
Official namespace: FreedomIntelligence/ShizhenGPT-7B-VL. Hugging Face direct access failed; ModelScope official namespace used, every downloaded file checked against repository SHA256. Per-file revisions saved in `docs/sources/modelscope_manifest.json`. The Hugging Face reference revision was f3a2f0985690c69f2cf60aca49f180c1da783541; All four ModelScope weight shard SHA256 values and byte lengths were subsequently verified against the pinned Hugging Face revision through hf-mirror resolve HEAD x-linked-etag / x-linked-size metadata. Evidence: docs/sources/hf_weight_headers.json.

The GitHub official Image-Text-to-Text pipeline is saved in `docs/sources/github_README.md`. Qwen2_5_VLForConditionalGeneration + AutoProcessor + process_vision_info + chat template + trimmed generated IDs are used unchanged in substance.

System PyTorch 2.5.1 / HIP 6.3.25405 / DTK 25.04.2 retained. The project venv reuses system packages. Missing ensurepip was worked around using host pip's --python option; only qwen-vl-utils 0.0.14 installed with --no-deps. No torch/torchvision replacement, quantization, or flash-attn installation.

BF16 queried as supported; actual BF16 128x128 matrix multiply produced finite results on K100_AI. Full model inference validation recorded separately.

Attention is eager for initial correctness. Pixel budget is 1003520 (1280 visual tokens at 28x28), with a 768-token output cap. Truncated answers are flagged; this budget affects both latency and visible detail.

Commands from the project directory:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/infer_image.py --image evaluation/images/tongue/SAMPLE.jpg --prompt '请客观描述舌象。'
.venv/bin/python scripts/infer_image.py --prompt '我最近容易疲劳，请给出需追问的问题。'
.venv/bin/python scripts/batch_evaluate.py
.venv/bin/python scripts/benchmark.py
```

All credentials stay out of project artifacts. The project has no remote-code execution dependency.

Processor uses use_fast=False to retain the slow image processor stored by the checkpoint; installed Transformers 4.57.1 otherwise changes this default and warns of possible output differences.

The checkpoint's generation_config.json contains repetition_penalty=1.05; this default is retained. do_sample=False overrides the checkpoint's sampling default, so temperature/top_p/top_k are inactive, and Transformers emits a warning about ignoring them. The checkpoint also stores an attention field in generation config; actual model attention is explicitly eager at model construction and checked separately in loaded_backend.json.

Initial ordinary-image smoke output used a 128-token cap and reached it. This was a runtime smoke test, not a complete semantic answer. Formal evaluation uses 768 tokens and records cap hits explicitly.
