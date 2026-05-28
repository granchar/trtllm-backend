#!/usr/bin/env python3.10
"""
TensorRT-LLM Web Server
提供聊天 API 和模型管理功能
支持 TRT-LLM 和 HuggingFace Transformers 双引擎
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

# TRT-LLM 支持的模型架构（用于自动判断用哪个引擎）
TRTLLM_SUPPORTED_ARCHS = {
    "llama", "mistral", "mixtral", "falcon", "gptj", "gpt_neox", "gpt_bigcode",
    "qwen2", "qwen", "gemma", "gemma2", "phi", "phi3", "baichuan", "chatglm",
    "internlm2", "internlm", "bloom", "mpt", "opt", "starcoder", "santacoder",
    "dbrx", "arctic", "minitron", "nemotron", "exaone", "grok", "granite",
    "recurrentgemma", "jais", "persimmon", "stablelm", "smaug",
}

# 预置小模型列表（适合 RTX 3080 20GB）
PRESET_MODELS = [
    # DeepSeek 蒸馏版（推荐！）
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "name": "DeepSeek R1 1.5B", "size": "3GB", "desc": "DeepSeek推理模型·超轻量", "engine": "transformers"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek R1 7B", "size": "15GB", "desc": "DeepSeek推理模型·推荐", "engine": "transformers"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "name": "DeepSeek R1 Llama 8B", "size": "16GB", "desc": "DeepSeek推理模型·Llama架构", "engine": "transformers"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "name": "DeepSeek R1 14B", "size": "29GB", "desc": "DeepSeek推理模型·需量化", "engine": "transformers"},
    # 原有模型
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B Chat", "size": "2.2GB", "desc": "轻量级对话模型"},
    {"id": "Qwen/Qwen1.5-0.5B-Chat", "name": "Qwen1.5 0.5B Chat", "size": "1GB", "desc": "通义千问超小模型"},
    {"id": "Qwen/Qwen1.5-1.8B-Chat", "name": "Qwen1.5 1.8B Chat", "size": "3.6GB", "desc": "通义千问小模型"},
    {"id": "microsoft/Phi-2", "name": "Phi-2 2.7B", "size": "5GB", "desc": "微软高质量小模型"},
    {"id": "google/gemma-2b-it", "name": "Gemma 2B IT", "size": "4GB", "desc": "Google Gemma 指令微调"},
]

# ============ LLM 管理器 ============
class LLMManager:
    def __init__(self):
        self.llm = None          # TRT-LLM 引擎实例
        self.hf_model = None     # HuggingFace Transformers 引擎实例
        self.hf_tokenizer = None # HuggingFace tokenizer
        self.current_model_id = None
        self.current_model_name = None
        self.model_config = None
        self.loading = False
        self.loading_progress = ""
        self.engine_type = None  # "trtllm" or "transformers"
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

    def detect_arch(self, model_path: str) -> str:
        """检测模型架构类型"""
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", [])
            if archs:
                arch_str = archs[0].lower()
                # DeepSeek 蒸馏版基于 Qwen 或 Llama
                for supported in TRTLLM_SUPPORTED_ARCHS:
                    if supported in arch_str:
                        return supported
                # deepseek 前缀的架构（如 DeepseekV3ForCausalLM）不归 TRT-LLM 管
                if "deepseek" in arch_str:
                    return "deepseek"
                return arch_str
        return "unknown"

    def should_use_trtllm(self, model_path: str, model_id: str = "") -> bool:
        """判断是否应该用 TRT-LLM 引擎"""
        # DeepSeek 系列全部走 Transformers（TRT-LLM v0.9.0 不支持）
        if "deepseek" in model_id.lower() or "deepseek" in model_path.lower():
            return False
        arch = self.detect_arch(model_path)
        if arch == "deepseek":
            return False
        # Qwen2 架构在 TRT-LLM v0.9.0 中也不稳定，走 Transformers
        if arch == "qwen2":
            return False
        if arch in TRTLLM_SUPPORTED_ARCHS:
            return True
        return False

    def load_model(self, model_id: str):
        """加载模型（自动选择引擎）"""
        with self.lock:
            if (self.llm is not None or self.hf_model is not None) and self.current_model_id == model_id:
                return  # 已经加载了

            # 卸载旧模型
            self._unload_current()

            self.loading = True
            self.loading_progress = "正在查找模型..."

        try:
            model_path = self.find_model_path(model_id)
            if model_path is None:
                raise Exception(f"模型 {model_id} 未找到，请先下载")

            self.loading_progress = f"正在加载模型: {model_path}"

            # 自动选择引擎
            if self.should_use_trtllm(model_path, model_id):
                self._load_with_trtllm(model_path, model_id)
            else:
                self._load_with_transformers(model_path, model_id)

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
                    "engine": self.engine_type,
                }
            else:
                self.model_config = {"engine": self.engine_type}

            self.loading_progress = f"模型加载完成（引擎: {self.engine_type}）"
        except Exception as e:
            self.loading = False
            self.loading_progress = f"加载失败: {str(e)}"
            raise e
        finally:
            self.loading = False

    def _unload_current(self):
        """卸载当前模型"""
        if self.llm is not None:
            try:
                self.llm.shutdown()
            except:
                pass
            self.llm = None
        if self.hf_model is not None:
            try:
                del self.hf_model
            except:
                pass
            self.hf_model = None
        if self.hf_tokenizer is not None:
            try:
                del self.hf_tokenizer
            except:
                pass
            self.hf_tokenizer = None
        self.current_model_id = None
        self.current_model_name = None
        self.engine_type = None
        # 清理 GPU 缓存
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    def _load_with_trtllm(self, model_path: str, model_id: str):
        """使用 TRT-LLM 引擎加载"""
        self.loading_progress = "正在构建/加载 TensorRT 引擎（首次约80秒）..."
        self.engine_type = "trtllm"
        from tensorrt_llm.hlapi.llm import LLM, ModelConfig
        config = ModelConfig(model_dir=model_path)
        self.llm = LLM(config)

    def _load_with_transformers(self, model_path: str, model_id: str):
        """使用 HuggingFace Transformers 引擎加载"""
        self.loading_progress = "正在加载 HuggingFace 模型到 GPU..."
        self.engine_type = "transformers"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 检测模型类型，决定加载方式
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)

        archs = cfg.get("architectures", [])
        model_type = cfg.get("model_type", "")

        # DeepSeek 蒸馏版基于 Qwen 或 Llama，需要用对应的类加载
        # 但 transformers 4.38 的 AutoModelForCausalLM 可以自动识别
        self.loading_progress = "正在加载 Tokenizer..."

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.loading_progress = "正在加载模型权重到 GPU（可能需要1-3分钟）..."

        # 根据模型大小选择精度
        num_params = cfg.get("num_hidden_layers", 24) * cfg.get("hidden_size", 2048) * cfg.get("num_attention_heads", 32) * 128
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        self.hf_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        self.loading_progress = "模型加载完成，正在预热..."

        # 简单预热
        try:
            inputs = self.hf_tokenizer("你好", return_tensors="pt").to(self.hf_model.device)
            with torch.no_grad():
                _ = self.hf_model.generate(**inputs, max_new_tokens=2)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[预热警告] {e}")

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """生成回复"""
        if self.llm is not None:
            return self._generate_trtllm(prompt, max_new_tokens)
        elif self.hf_model is not None:
            return self._generate_transformers(prompt, max_new_tokens)
        else:
            raise Exception("没有加载模型")

    def _generate_trtllm(self, prompt: str, max_new_tokens: int) -> str:
        """TRT-LLM 引擎生成"""
        from tensorrt_llm.hlapi.utils import SamplingConfig
        sampling_config = SamplingConfig(max_new_tokens=max_new_tokens)
        result_text = ""
        for output in self.llm.generate([prompt], sampling_config):
            result_text = output.text
        return result_text

    def _generate_transformers(self, prompt: str, max_new_tokens: int) -> str:
        """HuggingFace Transformers 引擎生成"""
        import torch
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.hf_tokenizer.pad_token_id or self.hf_tokenizer.eos_token_id,
            )
        # 只取新生成的部分
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        reply = self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
        return reply

llm_manager = LLMManager()

# ============ 会话存储 ============
chat_history = []  # [{"role": "user"|"assistant", "content": "...", "latency": 0}]

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
    """获取当前状态（含 GPU 实时信息 + 会话历史）"""
    gpu = get_gpu_info()
    return {
        "current_model": llm_manager.current_model_id,
        "current_model_name": llm_manager.current_model_name,
        "model_config": llm_manager.model_config,
        "loading": llm_manager.loading,
        "loading_progress": llm_manager.loading_progress,
        "engine_type": llm_manager.engine_type,
        "trtllm_version": "0.9.0",
        "gpu": gpu,
        "chat_history": chat_history,
    }

@app.get("/api/history")
async def get_history():
    """获取会话历史"""
    return chat_history

@app.post("/api/history/clear")
async def clear_history():
    """清空会话历史"""
    global chat_history
    chat_history = []
    return {"status": "ok"}

@app.get("/api/gpu")
async def gpu_status():
    """GPU 实时状态"""
    return get_gpu_info()

@app.get("/api/models")
async def get_models():
    """获取可用模型列表（预置 + 已下载）"""
    local_models = llm_manager.get_local_models()
    local_ids = set()
    for m in local_models:
        local_ids.add(m["id"].split("/")[-1])  # short name
        local_ids.add(m["id"])  # full id (e.g. "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B")
        local_ids.add(m["id"].replace("_", "/"))  # try to reconstruct original id

    models = []
    for pm in PRESET_MODELS:
        short_id = pm["id"].split("/")[-1]
        is_downloaded = short_id in local_ids or pm["id"] in local_ids
        is_loaded = llm_manager.current_model_id == pm["id"]
        models.append({
            **pm,
            "downloaded": is_downloaded,
            "loaded": is_loaded,
            "engine": pm.get("engine", ""),
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
        from huggingface_hub import model_info, hf_hub_download

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            yield json.dumps({"status": "downloading", "message": "Getting model info...", "progress": 0}) + "\n"

            # Get file list
            info = model_info(model_id, files_metadata=True)
            siblings = [s for s in (info.siblings or []) if not s.rfilename.endswith(".incomplete")]
            total_size = sum(s.size for s in siblings if s.size) or 1
            total_files = len(siblings)

            yield json.dumps({
                "status": "downloading",
                "message": f"Preparing: {total_files} files, {total_size/1024/1024:.1f} MB",
                "progress": 0, "total_mb": round(total_size/1024/1024, 1), "total_files": total_files
            }) + "\n"

            # Download file by file
            downloaded_total = 0
            for i, sibling in enumerate(siblings):
                filename = sibling.rfilename
                file_size = sibling.size or 0

                yield json.dumps({
                    "status": "downloading",
                    "message": f"[{i+1}/{total_files}] {filename}",
                    "progress": round(downloaded_total / total_size * 100, 1),
                    "downloaded_mb": round(downloaded_total/1024/1024, 1),
                    "total_mb": round(total_size/1024/1024, 1),
                    "file": filename,
                    "file_progress": 0
                }) + "\n"

                # Download single file with progress callback
                last_size = [0]
                def file_cb(progress):
                    chunk = progress.nbytes - last_size[0]
                    if chunk > 0:
                        nonlocal downloaded_total
                        downloaded_total += chunk
                    last_size[0] = progress.nbytes

                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        local_dir=str(save_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                except Exception as fe:
                    yield json.dumps({"status": "downloading", "message": f"Skip {filename}: {str(fe)}", "progress": round(downloaded_total/total_size*100,1)}) + "\n"

            yield json.dumps({"status": "done", "message": f"Download complete: {save_dir}", "path": str(save_dir), "progress": 100}) + "\n"
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
                "engine_type": llm_manager.engine_type,
            }) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(load_stream(), media_type="application/x-ndjson")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """聊天推理"""
    if llm_manager.llm is None and llm_manager.hf_model is None:
        raise HTTPException(400, "没有加载模型，请先选择并加载模型")
    if llm_manager.loading:
        raise HTTPException(400, "模型正在加载中，请稍候")

    try:
        t0 = time.time()
        reply = llm_manager.generate(req.prompt, req.max_new_tokens)
        latency = time.time() - t0
        # 保存到会话历史
        chat_history.append({"role": "user", "content": req.prompt})
        chat_history.append({"role": "assistant", "content": reply, "latency": round(latency, 2)})
        return {"reply": reply, "latency": round(latency, 2), "engine": llm_manager.engine_type}
    except Exception as e:
        raise HTTPException(500, f"推理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
