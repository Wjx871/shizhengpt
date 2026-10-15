---
license: apache-2.0
datasets:
- FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT
- FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT
language:
- zh
base_model:
- Qwen/Qwen2.5-7B
pipeline_tag: image-text-to-text
tags:
- Traditional Chinese Medicin
- Multimodal LLM
- multimodal
---

<div align="center">
<h1>
  ShizhenGPT-7B-VL
</h1>
</div>

<div align="center">
<a href="https://github.com/FreedomIntelligence/ShizhenGPT" target="_blank">GitHub</a> | <a href="https://arxiv.org/abs/2508.14706" target="_blank">Paper</a>
</div>


**ShizhenGPT** is the first multimodal LLM for Traditional Chinese Medicine (TCM).
It not only possesses strong expertise in TCM, but also supports TCM multimodal diagnostic capabilities, which involve looking (望), listening/smelling (闻), questioning (问), and pulse-taking (切).

👉 More details on GitHub: [ShizhenGPT](https://github.com/FreedomIntelligence/ShizhenGPT)




# <span>Model Info</span>

> **ShizhenGPT-7B-VL** is a variant derived from ShizhenGPT-7B-Omni that includes only the LLM and vision encoder. It is recommended if your use case involves text or vision tasks exclusively. For broader multimodal needs, please select one of the versions below.

|                        | Parameters | Supported Modalities          | Link                                                                  |
| ---------------------- | ---------- | ----------------------------- | --------------------------------------------------------------------- |
| **ShizhenGPT-7B-LLM**  | 7B         | Text                          | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-LLM) |
| **ShizhenGPT-7B-VL**   | 7B         | Text, Image Understanding     | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-VL) |
| **ShizhenGPT-7B-Omni** | 7B         | Text, Four Diagnostics (望闻问切) | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-Omni) |
| **ShizhenGPT-32B-LLM**  | 32B        | Text                          | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-32B-LLM) |
| **ShizhenGPT-32B-VL**   | 32B        | Text, Image Understanding     | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-32B-VL) |
| **ShizhenGPT-32B-Omni** | 32B        | Text, Four Diagnostics (望闻问切) | Available soon                                                          |

*Note: The LLM and VL models are parameter-split variants of ShizhenGPT-7B-Omni. Since their architectures align with Qwen2.5 and Qwen2.5-VL, they are easier to adapt to different environments. In contrast, ShizhenGPT-7B-Omni requires `transformers==4.51.0`.*


# <span>Usage</span>
You can use ShizhenGPT-7B-VL in the same way as [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). You can deploy it with tools like [vllm](https://github.com/vllm-project/vllm) or [Sglang](https://github.com/sgl-project/sglang),  or perform direct inference:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


processor = AutoProcessor.from_pretrained("FreedomIntelligence/ShizhenGPT-7B-VL")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("FreedomIntelligence/ShizhenGPT-7B-VL", torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/path/to/your/image.png",
            },
            {"type": "text", "text": "请从中医角度解读这张舌苔。"},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```


# <span>📖 Citation</span>
```
@misc{chen2025shizhengptmultimodalllmstraditional,
      title={ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine}, 
      author={Junying Chen and Zhenyang Cai and Zhiheng Liu and Yunjin Yang and Rongsheng Wang and Qingying Xiao and Xiangyi Feng and Zhan Su and Jing Guo and Xiang Wan and Guangjun Yu and Haizhou Li and Benyou Wang},
      year={2025},
      eprint={2508.14706},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14706},
}
```