"""
TinyLlama LoRA Ticket Parser API
CPU-only with Memory Management and Error Handling
"""

import os
import torch
import logging
import psutil
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ============================================================================
# CONFIG
# ============================================================================

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "tinyllama-ticket-parser-lora"   # folder you pasted in this project

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODEL
# ============================================================================

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None

# ============================================================================
# MEMORY MONITOR
# ============================================================================

class MemoryMonitor:
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    @staticmethod
    def log_memory(label: str = ""):
        mem = MemoryMonitor.get_memory_info()
        logger.info(
            f"Memory [{label}] - RSS: {mem['rss_mb']:.2f}MB | "
            f"VMS: {mem['vms_mb']:.2f}MB | Usage: {mem['percent']:.2f}%"
        )

    @staticmethod
    def check_memory_limit(threshold_percent: float = 85.0) -> bool:
        mem = MemoryMonitor.get_memory_info()
        if mem["percent"] > threshold_percent:
            logger.warning(f"⚠️ Memory critical: {mem['percent']:.2f}%")
            return False
        return True

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TicketRequest(BaseModel):
    text: str
    max_length: int = 200

class TicketResponse(BaseModel):
    result: Optional[str] = None
    error: Optional[str] = None
    memory_used_mb: float
    confidence: float = 0.85

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_usage: Dict[str, float]

# ============================================================================
# MODEL LOADING (BASE + LORA)
# ============================================================================

async def load_model_at_startup():
    """Load base TinyLlama + LoRA adapter once at startup (simplified)."""
    global _model, _tokenizer

    logger.info("=" * 70)
    logger.info("🚀 STARTING MODEL + LoRA LOADING (simple)")
    logger.info("=" * 70)

    MemoryMonitor.log_memory("Before loading")

    try:
        if not os.path.isdir(ADAPTER_PATH):
            raise RuntimeError(f"Adapter folder not found at: {ADAPTER_PATH}")

        logger.info(f"📦 Base model: {BASE_MODEL_NAME}")
        logger.info(f"📦 LoRA adapter folder: {ADAPTER_PATH}")

        # 1) Tokenizer
        logger.info("⏳ Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            model_max_length=512,
        )

        # 2) Base model (CPU only)
        logger.info("⏳ Loading base model (CPU-only, optimized)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # 3) Attach LoRA adapter (no PeftConfig)
        logger.info("⏳ Loading LoRA adapter (simple)...")
        _model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            device_map="cpu",
        )

        _model.eval()
        torch.set_grad_enabled(False)

        logger.info("✅ Base + LoRA model loaded successfully (simple)")
        MemoryMonitor.log_memory("After loading")
        return True

    except Exception as e:
        logger.error(f"❌ FATAL during model load: {e}", exc_info=True)
        _model = None
        _tokenizer = None
        return False


async def shutdown_model():
    """Cleanup on shutdown."""
    global _model, _tokenizer
    logger.info("🛑 Shutting down...")
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("✅ Cleanup complete")

# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await load_model_at_startup()
        logger.info("✅ Ready to serve")
    except Exception as e:
        logger.critical(f"Failed to start: {e}")
        # let FastAPI start but with model_loaded = False
    yield
    await shutdown_model()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="TinyLlama LoRA Ticket Parser API",
    description="CPU-only TinyLlama with LoRA adapter and memory monitoring",
    version="2.0.0",
    lifespan=lifespan,
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    global _model, _tokenizer

    mem = MemoryMonitor.get_memory_info()

    if _model is None or _tokenizer is None:
        logger.warning("❌ Health check - model not loaded")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "memory_usage": mem,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "memory_usage": mem,
    }

@app.post("/parse-ticket", response_model=TicketResponse, tags=["Inference"])
async def parse_ticket(request: TicketRequest):
    global _model, _tokenizer

    MemoryMonitor.log_memory("Start")

    if _model is None or _tokenizer is None:
        logger.error("❌ Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    if not MemoryMonitor.check_memory_limit(85.0):
        raise HTTPException(status_code=503, detail="Server memory high")

    try:
        logger.info(f"Processing ticket: {request.text[:80]}...")

        inputs = _tokenizer(
            request.text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_length=min(request.max_length, 200),
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id,
            )

        result_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        mem = MemoryMonitor.get_memory_info()

        logger.info("✅ Generation success")

        return TicketResponse(
            result=result_text,
            error=None,
            memory_used_mb=mem["rss_mb"],
            confidence=0.85,
        )

    except Exception as e:
        logger.error(f"❌ Error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory", tags=["Monitoring"])
async def memory_status():
    mem = MemoryMonitor.get_memory_info()
    sys_mem = psutil.virtual_memory()
    return {
        "process_memory_mb": mem,
        "system_memory": {
            "total_gb": sys_mem.total / (1024 ** 3),
            "available_gb": sys_mem.available / (1024 ** 3),
            "percent": sys_mem.percent,
        },
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting TinyLlama LoRA API (CPU-only)")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False,
        log_level="info",
    )
