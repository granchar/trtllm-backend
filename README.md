# TensorRT-LLM Web Backend

基于 TensorRT-LLM v0.9.0 的 Web 聊天后端，支持模型管理、GPU 实时监控。

## 功能

- 💬 聊天对话界面（TensorRT-LLM 推理）
- 📊 GPU 实时状态监控（显存、利用率、温度、功耗）
- 📥 模型下载（从 HuggingFace 镜像）
- 🔄 模型切换与加载状态展示

## 依赖

- Python 3.10+
- TensorRT-LLM v0.9.0
- FastAPI + uvicorn
- huggingface_hub

## 快速启动

```bash
pip install fastapi uvicorn huggingface_hub tensorrt-llm==0.9.0 --extra-index-url https://pypi.nvidia.com
python server.py
```

服务启动在 `http://0.0.0.0:8080`

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/status` | GET | 当前状态 + GPU 信息 |
| `/api/gpu` | GET | GPU 实时状态 |
| `/api/models` | GET | 可用模型列表 |
| `/api/download` | POST | 下载模型 |
| `/api/load` | POST | 加载模型 |
| `/api/chat` | POST | 聊天推理 |

## 配置

在 `server.py` 中修改以下配置：

- `MODELS_DIR`: 模型下载目录（默认 `/hy-tmp/trtllm-web/models`）
- `TMPDIR`: TensorRT 引擎构建临时目录（默认 `/hy-tmp/trtllm-web/tmp`）
- `PRESET_MODELS`: 预置模型列表
- `HF_ENDPOINT`: HuggingFace 镜像地址（默认 `https://hf-mirror.com`）
