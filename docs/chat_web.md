# 简约模型聊天页面

本地访问：<http://127.0.0.1:7860>

2026-10-13 更新：网页默认使用“谨慎回答”，会进行质控、依据核对和信息不足时弃答；可切换“原始模型（未核对）”做对照。具体能力范围见 [谨慎回答说明](cautious_chat_usage.md)。以下原始模型回复说明仅适用于原始模式。

支持文字、单张图片及图文联合输入，多轮上下文、新对话、上传预览和原始模型回复。Enter 发送，Shift+Enter 换行。图片可使用 JPG、PNG、WebP，最多 8 MB；输入图片会纠正 EXIF 朝向并缩放至最长边 1600 像素，模型仍使用原推理配置的像素上限。回答一次完整显示，不做流式输出或额外诊断包装。

默认使用 ShizhenGPT-7B-VL 基础权重，不启用实验性 LoRA。界面中的耗时为服务器处理时间。回答达到 768 token 上限时会显示提示。每个对话最多 15 次回复，输入上下文上限 12000 token；达到限制时需要新建对话。

## 重新启动

服务器上执行（若服务已经运行，无需重复启动）：

```bash
cd /root/private_data/shizhengpt-evaluation
mkdir -p evaluation/web
nohup .venv/bin/python scripts/chat_server.py > evaluation/web/server.log 2>&1 &
```

本机 PowerShell 开启 SSH 转发，按提示输入服务器密码，保持窗口打开：

```powershell
ssh -N -L 7860:127.0.0.1:7860 -p 50025 -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=10 root@172.19.128.219
```

浏览器打开 <http://127.0.0.1:7860>。若 SSH 连接中断，重新执行转发命令并刷新页面即可。服务绑定服务器回环地址，通过 SSH 使用，不直接公开到互联网。

## 对话与数据

对话保存在服务内存中，图片在服务器临时目录中。新对话会清理旧会话；超过两小时未使用的对话会在下一次请求时清理。服务重启或刷新页面后不会恢复会话。没有账号系统和持久聊天记录。模型同一时间处理一个请求，繁忙时页面提示重试。

## 文件

- `web/index.html`：无构建依赖的 HTML/CSS/JavaScript 页面。
- `scripts/chat_server.py`：静态页面及聊天接口，复用原始推理引擎。
- `scripts/check_chat_web.cjs`：连接真实模型的浏览器端验证，检查文本、历史记忆、图片、联合输入和移动端宽度；需要 Playwright。
- `evaluation/web/`：浏览器验证输出和页面截图。
