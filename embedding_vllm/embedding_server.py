# server.py (vLLM backend, keep DeepWiki-style API)
import base64
import io
import os
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from vllm import LLM

APP_TITLE = "Qwen3-VL-Embedding Service (vLLM pooling)"
DEFAULT_INSTRUCTION = os.environ.get("DEFAULT_INSTRUCTION", "Represent the user's input")
MODEL_PATH = os.environ.get("MODEL_PATH", "/model")

# 8B 默认 4096（你原逻辑保留）
DEFAULT_DIMENSIONS = int(os.environ.get("DEFAULT_DIMENSIONS", "4096"))
MIN_DIMENSIONS = int(os.environ.get("MIN_DIMENSIONS", "64"))
MAX_DIMENSIONS = int(os.environ.get("MAX_DIMENSIONS", "4096"))

# vLLM 省显存相关（可通过环境变量调）
VLLM_TP_SIZE = int(os.environ.get("VLLM_TP_SIZE", "1"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "8"))
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "bfloat16")  # "float16" / "bfloat16"

# vLLM 多模态占位符（DeepWiki要求手动插入）
VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _decode_base64_image(image_base64: str) -> Image.Image:
    try:
        raw = image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")


def _validate_dimensions(dim: Optional[int]) -> int:
    if dim is None:
        return DEFAULT_DIMENSIONS
    if not (MIN_DIMENSIONS <= dim <= MAX_DIMENSIONS):
        raise HTTPException(status_code=422, detail=f"dimensions must be in [{MIN_DIMENSIONS}, {MAX_DIMENSIONS}]")
    return int(dim)


def _postprocess(emb: torch.Tensor, dimensions: int, normalize: bool) -> torch.Tensor:
    # emb: [D]
    if emb.dim() != 1:
        raise HTTPException(status_code=500, detail=f"Unexpected embedding shape: {tuple(emb.shape)}")
    if emb.numel() < dimensions:
        raise HTTPException(status_code=500, detail=f"Model dim {emb.numel()} < requested {dimensions}")
    emb = emb[:dimensions]
    if normalize:
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


# =========================
# FastAPI models (keep your schema)
# =========================
class EmbeddingItem(BaseModel):
    text: Optional[str] = None
    image: Optional[Union[str, Any]] = None  # keep field, but we mainly use image_base64 in service
    video: Optional[Any] = None              # vLLM视频这里先不做（需要额外适配）

    instruction: Optional[str] = None

    image_base64: Optional[str] = None
    dimensions: Optional[int] = Field(default=None, description="MRL output dims, e.g. 1024/2048/4096")
    normalize: bool = True

    def to_vllm_input(self) -> Dict[str, Any]:
        """
        vLLM interface requires:
          {"prompt": "...", "multi_modal_data": {"image": PIL.Image}}
        and prompt must include the vision placeholder when image is present.
        """
        if self.video is not None:
            raise HTTPException(status_code=422, detail="video is not supported in this vLLM server yet")

        instruction = (self.instruction or DEFAULT_INSTRUCTION).strip()

        # Decide if we have an image
        pil_img = None
        if self.image_base64:
            pil_img = _decode_base64_image(self.image_base64)
        elif self.image is not None:
            # 如果你想支持 image=本地路径/URL，需要你在这里自己实现加载；
            # 先明确报错避免默默失败
            raise HTTPException(
                status_code=422,
                detail="For vLLM server, please use image_base64 (PIL only in multi_modal_data)."
            )

        # Decide if we have text
        text = (self.text or "").strip()

        if pil_img is None and not text:
            raise HTTPException(status_code=422, detail="Each item must include at least one of: text/image/video")

        if pil_img is None:
            prompt = f"{instruction}\n{text}"
            return {"prompt": prompt}

        # image-only or image+text: must insert placeholder
        if text:
            prompt = f"{VISION_PLACEHOLDER}\n{instruction}\n{text}"
        else:
            prompt = f"{VISION_PLACEHOLDER}\n{instruction}"

        return {"prompt": prompt, "multi_modal_data": {"image": pil_img}}


class EmbeddingRequest(BaseModel):
    model: str = "Qwen3-VL-Embedding-8B"
    input: Union[str, List[str], EmbeddingItem, List[EmbeddingItem]]


# =========================
# App init
# =========================
app = FastAPI(title=APP_TITLE, version="1.0.0")

# Initialize vLLM pooling runner
# DeepWiki: vLLM version >= 0.14.0, use LLM(..., runner="pooling") and .embed() :contentReference[oaicite:2]{index=2}
llm = LLM(
    model=MODEL_PATH,
    runner="pooling",
    tensor_parallel_size=VLLM_TP_SIZE,
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    max_num_seqs=VLLM_MAX_NUM_SEQS,
    dtype=VLLM_DTYPE,
    max_model_len=2048,
    trust_remote_code=True
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "default_dimensions": DEFAULT_DIMENSIONS,
        "vllm": {
            "tensor_parallel_size": VLLM_TP_SIZE,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "dtype": VLLM_DTYPE,
        },
    }


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    # Normalize input to list[EmbeddingItem]
    items: List[EmbeddingItem] = []
    if isinstance(req.input, str):
        items = [EmbeddingItem(text=req.input)]
    elif isinstance(req.input, list) and all(isinstance(x, str) for x in req.input):
        items = [EmbeddingItem(text=x) for x in req.input]
    elif isinstance(req.input, EmbeddingItem):
        items = [req.input]
    elif isinstance(req.input, list) and all(isinstance(x, EmbeddingItem) for x in req.input):
        items = req.input
    else:
        raise HTTPException(status_code=422, detail="Unsupported input format")

    vllm_inputs: List[Dict[str, Any]] = []
    dims: List[int] = []
    norms: List[bool] = []

    for it in items:
        vllm_inputs.append(it.to_vllm_input())
        dims.append(_validate_dimensions(it.dimensions))
        norms.append(bool(it.normalize))

    # vLLM embed
    try:
        outputs = llm.embed(vllm_inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM embed failed: {e}")

    # Each output: EmbeddingRequestOutput with outputs.embedding :contentReference[oaicite:3]{index=3}
    data = []
    for i, out in enumerate(outputs):
        emb_list = out.outputs.embedding
        emb = torch.tensor(emb_list, dtype=torch.float32)  # postprocess in fp32
        emb = _postprocess(emb, dims[i], norms[i])

        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": emb.cpu().tolist(),
            }
        )

    return {"object": "list", "model": req.model, "data": data}
