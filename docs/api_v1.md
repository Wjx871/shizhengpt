# ShizhenGPT 图文问答接口 v1

已在现有曙光 DCU 模型服务中加入 HTTP JSON 接口，复用同一份已加载模型。自定义 REST 接口，**不是 OpenAI Chat Completions 协议**。

## 当前部署版本

- 模型：`ShizhenGPT-7B-VL`，原始基础权重，`base-bf16`，未加载实验性 LoRA。
- 运行环境：现有曙光 DCU 实例，BF16、eager attention。
- `/v1/chat` 默认 `mode: "raw"`，直接返回原始模型回答，不经过额外质控和回答筛选。
- `guarded` 为谨慎回答 v2，`guarded_v1` 为旧谨慎版。两者仍使用同一基础权重，仅外层流程不同。
- API 版本：`1.0.0`；可通过 `/healthz` 和 `/v1/models` 查询实际模式、运行类型及是否加载 LoRA。

## 地址与访问方式

模型服务器内部地址：`http://127.0.0.1:7860`。服务绑定服务器回环地址，目前通过 SSH 隧道访问，未将 7860 直接开放为公网接口。

在调用方机器建立转发（保留该终端，使用自己的服务器登录凭据）：

```bash
ssh -N -L 7860:127.0.0.1:7860 -p 50025 root@<模型服务器地址>
```

之后调用方使用 `http://127.0.0.1:7860`；若本机 7860 被占用，可改为 `-L 17860:127.0.0.1:7860`，客户端地址随之改成 `http://127.0.0.1:17860`。统一部署平台可由其后端或网关通过服务器内部网络/隧道接入。当前接口没有账号鉴权，不应直接暴露到不受信任网络。

## 接口列表

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 模型就绪、API 与流程版本 |
| GET | `/v1/models` | 模型、运行类型、LoRA 状态、输入限制 |
| GET | `/openapi.json` | OpenAPI 3.0.3 描述，可导入 Apifox/Postman 等工具 |
| POST | `/v1/chat` | 文字、图片、图文联合、多轮会话 |
| POST | `/v1/sessions/reset` | 清理会话及临时图片 |

旧网页的 `/api/chat`、`/api/reset` 保留兼容；其缺省模式仍是 `guarded`。新接入请使用 `/v1/chat`，不要把旧路径的默认行为与新路径混淆。网页目前会显式选择 `raw`。

离线导入文件：[openapi_v1.json](openapi_v1.json)。导入后把工具中的服务地址设置为实际隧道或网关入口。

## 请求格式

`Content-Type: application/json`，字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 最多 6000 字；与图片至少提供一项 |
| `image_base64` | string | 单张 JPEG/PNG/WebP 的纯 Base64，或 `data:image/png;base64,...` |
| `session_id` | string | 首次省略；续聊传上一次返回值 |
| `mode` | string | `raw`（默认）、`guarded`、`guarded_v1`；同一会话每次必须使用相同模式 |
| `max_new_tokens` | integer | 仅 raw，1–1024，默认 768；是输出 token 上限，不是汉字数 |

只支持 Base64，不接收本地文件路径或远程图片 URL。图片解码后最多 8 MB、2400 万像素，请求体最多 12 MB。图片会纠正 EXIF 朝向、转为 RGB、最长边缩至 1600 并保存为 JPEG，再进入原推理流程。

## 最小调用

服务器或建立隧道后的机器执行：

```bash
curl http://127.0.0.1:7860/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"我最近轻微疲劳，请先追问。","mode":"raw"}'
```

图文联合的 JSON 结构（将占位内容替换为实际 Base64）：

```json
{
  "text": "请描述这张图片中可以观察到的内容。",
  "image_base64": "data:image/jpeg;base64,<图片Base64>",
  "mode": "raw",
  "max_new_tokens": 512
}
```

只提供 `image_base64` 也可以，服务会使用一个通用图片描述问题。推荐使用下方 Python 客户端自动读取图片并编码。

## 返回格式

以下为格式示例，不是诊断或性能实测：

```json
{
  "request_id": "服务生成的请求编号",
  "api_version": "1.0.0",
  "model": "ShizhenGPT-7B-VL",
  "weights_version": "base-bf16",
  "session_id": "服务生成的会话编号",
  "mode": "raw",
  "workflow_version": "raw_base",
  "answer": "模型原始回答……",
  "latency_sec": 8.2,
  "finish_reason": "stop",
  "truncated": false,
  "guard_status": "raw_unchecked",
  "usage": {"input_tokens": 100, "output_tokens": 80},
  "model_invocations": 1
}
```

`answer` 保留模型原始文字，可能包含 Markdown，调用方可自行渲染。达到输出上限时 `truncated=true`、`finish_reason="length"`。谨慎模式的 `workflow_version` 为其实际版本、`guard_status` 为检查/弃答状态，`usage=null`，`model_invocations` 统计实际阶段与重试，不会编造 token 用量。

## Python 客户端

无需第三方 SDK：

```bash
python scripts/api_client.py --text "你好，请简单介绍自己。"
python scripts/api_client.py --image /path/to/photo.jpg --text "请描述这张图片。"
python scripts/api_client.py --image /path/to/photo.jpg --max-new-tokens 512
```

从其他 Python 系统调用（把 `scripts/api_client.py` 放入可导入目录）：

```python
from api_client import ShizhenClient

client = ShizhenClient("http://127.0.0.1:7860", timeout=180)
first = client.chat(text="请描述图片。", image="photo.jpg", mode="raw")
print(first["answer"])

second = client.chat(
    text="请简短概括上一条回答。",
    session_id=first["session_id"],
    mode="raw",
)
print(second["answer"])
client.reset(first["session_id"])
```

会话最多 15 次回复，raw 输入上下文最多 12000 tokens，空闲两小时后过期，重启后失效。旧会话过期会返回错误，不会静默丢弃历史。切换模式请新建会话。客户端不会自动重试推理，避免连接超时后重复提交同一轮。

## 错误与并发

```json
{"request_id":"请求编号","error":{"code":"invalid_request","message":"错误说明"}}
```

| HTTP 状态 | code | 含义 |
|---|---|---|
| 400 | `invalid_request` / `invalid_content_type` | 参数、图片或 JSON 不合法；会话失效/模式不匹配 |
| 404 | `not_found` | 路径不存在 |
| 413 | `request_too_large` | 请求超过 12 MB |
| 503 | `model_busy` | 单个模型正在处理请求，附 `Retry-After: 2` |
| 500 | `inference_failed` | 模型内部失败，保留 request_id 供维护者查日志 |

当前同一时间只处理一个模型请求，返回完整回答，不做流式输出。调用超时建议 180 秒；多应用同时使用时，由统一后端安排排队。

## 验证

`scripts/check_chat_api.py` 检查无模型的接口契约；`scripts/check_chat_api_live.py` 在真实服务检查文字、图片、图文、多轮、清理、截断、参数错误及并发繁忙响应。测试记录保存在 `evaluation/chat_api_v1_checks/results.json`，不评价医学诊断正确率。

本次部署已通过 6 项契约测试及 22 项真实 HTTP 检查；Python 命令行客户端也已实测。运行时查询确认 `device_name=K100_AI`、`dtype=torch.bfloat16`、`lora_loaded=false`。
