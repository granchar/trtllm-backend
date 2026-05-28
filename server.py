#!/usr/bin/env python3.10
"""
TensorRT-LLM Web Server
提供聊天 API 和模型管理功能
"""
import os, sys, json, time, threading, subprocess
os.environ["TMPDIR"] = "/hy-tmp/trtllm-web/tmp"
import multiprocessing
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = FastAPI(title="TensorRT-LLM Web UI")

# ============ 配置 ============
MODELS_DIR = Path("/hy-tmp/trtllm-web/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 预置小模型列表（适合 RTX 3080 20GB）
PRESET_MODELS = [
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B Chat", "size": "2.2GB", "desc": "轻量级对话模型"},
    {"id": "Qwen/Qwen1.5-0.5B-Chat", "name": "Qwen1.5 0.5B Chat", "size": "1GB", "desc": "通义千问超小模型"},
    {"id": "Qwen/Qwen1.5-1.8B-Chat", "name": "Qwen1.5 1.8B Chat", "size": "3.6GB", "desc": "通义千问小模型"},
    {"id": "microsoft/Phi-2", "name": "Phi-2 2.7B", "size": "5GB", "desc": "微软高质量小模型"},
    {"id": "google/gemma-2b-it", "name": "Gemma 2B IT", "size": "4GB", "desc": "Google Gemma 指令微调"},
]

# ============ LLM 管理器 ============
class LLMManager:
    def __init__(self):
        self.llm = None
        self.current_model_id = None
        self.current_model_name = None
        self.model_config = None
        self.loading = False
        self.loading_progress = ""
        self.lock = threading.Lock()

    def get_local_models(self):
        """获取已下载的本地模型列表"""
        models = []
        if MODELS_DIR.exists():
            for d in MODELS_DIR.iterdir():
                if d.is_dir() and (d / "config.json").exists():
                    models.append({"id": d.name, "name": d.name, "local": True})
        # 也检查 /root/models
        root_models = Path("/root/models")
        if root_models.exists():
            for d in root_models.iterdir():
                if d.is_dir() and (d / "config.json").exists():
                    if not any(m["id"] == d.name for m in models):
                        models.append({"id": d.name, "name": d.name, "local": True, "path": str(d)})
        return models

    def find_model_path(self, model_id: str) -> Optional[str]:
        """查找模型的本地路径"""
        # 先查 /hy-tmp
        p = MODELS_DIR / model_id.replace("/", "_")
        if p.exists() and (p / "config.json").exists():
            return str(p)
        # 再查 /root/models
        p = Path("/root/models") / model_id.split("/")[-1]
        if p.exists() and (p / "config.json").exists():
            return str(p)
        return None

    def load_model(self, model_id: str):
        """加载模型"""
        with self.lock:
            if self.llm is not None and self.current_model_id == model_id:
                return  # 已经加载了

            # 卸载旧模型
            if self.llm is not None:
                try:
                    self.llm.shutdown()
                except:
                    pass
                self.llm = None
                self.current_model_id = None

            self.loading = True
            self.loading_progress = "正在查找模型..."

        try:
            model_path = self.find_model_path(model_id)
            if model_path is None:
                raise Exception(f"模型 {model_id} 未找到，请先下载")

            self.loading_progress = f"正在加载模型: {model_path}"

            from tensorrt_llm.hlapi.llm import LLM, ModelConfig

            self.loading_progress = "正在构建/加载 TensorRT 引擎（首次约80秒）..."
            config = ModelConfig(model_dir=model_path)
            self.llm = LLM(config)

            self.current_model_id = model_id
            self.current_model_name = model_id.split("/")[-1]

            # 读取模型信息
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                self.model_config = {
                    "model_type": cfg.get("model_type", "unknown"),
                    "architectures": cfg.get("architectures", []),
                    "vocab_size": cfg.get("vocab_size", 0),
                    "hidden_size": cfg.get("hidden_size", 0),
                    "num_layers": cfg.get("num_hidden_layers", cfg.get("n_layer", 0)),
                    "num_heads": cfg.get("num_attention_heads", cfg.get("n_head", 0)),
                }
            else:
                self.model_config = {}

            self.loading_progress = "模型加载完成"
        except Exception as e:
            self.loading = False
            self.loading_progress = f"加载失败: {str(e)}"
            raise e
        finally:
            self.loading = False

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """生成回复"""
        if self.llm is None:
            raise Exception("没有加载模型")

        from tensorrt_llm.hlapi.utils import SamplingConfig
        sampling_config = SamplingConfig(max_new_tokens=max_new_tokens)

        result_text = ""
        for output in self.llm.generate([prompt], sampling_config):
            result_text = output.text
        return result_text

llm_manager = LLMManager()

# ============ API 路由 ============

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

class DownloadRequest(BaseModel):
    model_id: str

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = Path("/hy-tmp/trtllm-web/index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>index.html not found</h1>"

def get_gpu_info():
    """获取 GPU 实时状态"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            parts = [x.strip() for x in r.stdout.strip().split(",")]
            return {
                "name": parts[0],
                "memory_total": int(parts[1]),
                "memory_used": int(parts[2]),
                "memory_free": int(parts[3]),
                "utilization": int(parts[4]),
                "temperature": int(parts[5]),
                "power_draw": float(parts[6]),
                "power_limit": float(parts[7]),
            }
    except:
        pass
    return None

@app.get("/api/status")
async def get_status():
    """获取当前状态（含 GPU 实时信息）"""
    gpu = get_gpu_info()
    return {
        "current_model": llm_manager.current_model_id,
        "current_model_name": llm_manager.current_model_name,
        "model_config": llm_manager.model_config,
        "loading": llm_manager.loading,
        "loading_progress": llm_manager.loading_progress,
        "trtllm_version": "0.9.0",
        "gpu": gpu,
    }

@app.get("/api/gpu")
async def gpu_status():
    """GPU 实时状态"""
    return get_gpu_info()

@app.get("/api/models")
async def get_models():
    """获取可用模型列表（预置 + 已下载）"""
    local_models = llm_manager.get_local_models()
    local_ids = {m["id"].split("/")[-1] for m in local_models}

    models = []
    for pm in PRESET_MODELS:
        short_id = pm["id"].split("/")[-1]
        models.append({
            **pm,
            "downloaded": short_id in local_ids or pm["id"] in local_ids,
        })
    return models

@app.get("/api/models/local")
async def get_local_models():
    """获取已下载的本地模型"""
    return llm_manager.get_local_models()

@app.post("/api/download")
async def download_model(req: DownloadRequest):
    """下载模型"""
    model_id = req.model_id
    save_name = model_id.replace("/", "_")
    save_dir = MODELS_DIR / save_name

    if save_dir.exists() and (save_dir / "config.json").exists():
        return {"status": "already_exists", "path": str(save_dir)}

    def download_stream():
        import sys
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from huggingface_hub import snapshot_download, hf_hub_url, model_info

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            yield json.dumps({"status": "downloading", "message": f"正在获取模型信息 {model_id} ...", "progress": 0}) + "\n"

            # 获取模型文件列表和大小
            info = model_info(model_id, files_metadata=True)
            siblings = info.siblings if info.siblings else []
            total_size = sum(s.size for s in siblings if s.size) or 0
            total_files = len(siblings)

            yield json.dumps({"status": "downloading", "message": f"共 {total_files} 个文件，总大小 {total_size/1024/1024:.1f}MB", "progress": 0, "total_size": total_size, "total_files": total_files}) + "\n"

            # 用回调跟踪进度
            downloaded = [0]
            last_reported = [0]

            class ProgressTracker:
                def __init__(self):
                    pass
            tracker = ProgressTracker()

            def progress_callback(progress_obj):
                current = progress_obj.nbytes
                file_name = getattr(progress_obj, "filename", "")
                # 累计已下载大小
                downloaded[0] += current - last_reported[0]
                last_reported[0] = current
                pct = min(round(downloaded[0] / total_size * 100, 1), 100) if total_size > 0 else 0
                yield json.dumps({"status": "downloading", "message": f"正在下载 {file_name}", "progress": pct, "downloaded_mb": round(downloaded[0]/1024/1024, 1), "total_mb": round(total_size/1024/1024, 1)}) + "\n"

            # 使用 snapshot_download 的 resume_download 和回调
            snapshot_download(
                repo_id=model_id,
                local_dir=str(save_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            yield json.dumps({"status": "done", "message": f"下载完成: {save_dir}", "path": str(save_dir), "progress": 100}) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(download_stream(), media_type="application/x-ndjson")

@app.post("/api/load")
async def load_model(req: DownloadRequest):
    """加载模型"""
    model_id = req.model_id

    if llm_manager.loading:
        raise HTTPException(400, "正在加载模型中，请稍候")

    def load_stream():
        try:
            yield json.dumps({"status": "loading", "message": f"正在加载 {model_id} ..."}) + "\n"
            llm_manager.load_model(model_id)
            yield json.dumps({
                "status": "done",
                "message": f"模型 {model_id} 加载成功",
                "model_name": llm_manager.current_model_name,
                "model_config": llm_manager.model_config,
            }) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(load_stream(), media_type="application/x-ndjson")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """聊天推理"""
    if llm_manager.llm is None:
        raise HTTPException(400, "没有加载模型，请先选择并加载模型")
    if llm_manager.loading:
        raise HTTPException(400, "模型正在加载中，请稍候")

    try:
        t0 = time.time()
        reply = llm_manager.generate(req.prompt, req.max_new_tokens)
        latency = time.time() - t0
        return {"reply": reply, "latency": round(latency, 2)}
    except Exception as e:
        raise HTTPException(500, f"推理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
