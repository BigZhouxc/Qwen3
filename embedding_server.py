# server.py
import base64
import io
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

# DeepWiki / README 推荐的 embedder
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

APP_TITLE = "Qwen3-VL-Embedding Service (DeepWiki style)"
DEFAULT_INSTRUCTION = os.environ.get("DEFAULT_INSTRUCTION", "Represent the user's input")
MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 8B 默认 4096；2B 默认 2048（DeepWiki 写了规格） :contentReference[oaicite:6]{index=6}
DEFAULT_DIMENSIONS = int(os.environ.get("DEFAULT_DIMENSIONS", "4096"))
MIN_DIMENSIONS = int(os.environ.get("MIN_DIMENSIONS", "64"))
MAX_DIMENSIONS = int(os.environ.get("MAX_DIMENSIONS", "4096"))

# Optional: flash attention
ATN_IMPL = os.environ.get("ATN_IMPL", "")  # e.g. "flash_attention_2"
TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "")  # e.g. "bfloat16", "float16"


def _parse_dtype(s: str):
    if not s:
        return None
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported TORCH_DTYPE: {s}")


# =========================
# FastAPI models
# =========================
class EmbeddingItem(BaseModel):
    # 与 DeepWiki 输入 keys 保持一致：text/image/video/instruction :contentReference[oaicite:7]{index=7}
    text: Optional[str] = None
    image: Optional[Union[str, Any]] = None  # str(path/url) or PIL.Image
    video: Optional[Any] = None

    instruction: Optional[str] = None

    # 扩展（服务层能力，不影响 DeepWiki 原生结构）
    image_base64: Optional[str] = None  # 如果提供，会解码成 PIL 并写入 image
    dimensions: Optional[int] = Field(default=None, description="MRL output dims, e.g. 1024/2048/4096")
    normalize: bool = True

    def to_embedder_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.text:
            d["text"] = self.text
        if self.video is not None:
            d["video"] = self.video
        if self.instruction:
            d["instruction"] = self.instruction
        else:
            d["instruction"] = DEFAULT_INSTRUCTION

        # image priority: image_base64 > image
        if self.image_base64:
            try:
                raw = self.image_base64
                if raw.startswith("data:"):
                    raw = raw.split(",", 1)[1]
                img_bytes = base64.b64decode(raw)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                d["image"] = img
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")
        elif self.image is not None:
            d["image"] = self.image

        if not any(k in d for k in ("text", "image", "video")):
            raise HTTPException(status_code=422, detail="Each item must include at least one of: text/image/video")
        return d


class EmbeddingRequest(BaseModel):
    model: str = "Qwen3-VL-Embedding-8B"
    # 兼容：OpenAI embeddings 风格的 input
    input: Union[str, List[str], EmbeddingItem, List[EmbeddingItem]]


# =========================
# App init
# =========================
app = FastAPI(title=APP_TITLE, version="1.0.0")

# 用 DeepWiki 的 Qwen3VLEmbedder 初始化与参数（max_length/min_pixels/max_pixels 等都由类支持） :contentReference[oaicite:8]{index=8}
_embedder_kwargs: Dict[str, Any] = {
    "model_name_or_path": MODEL_PATH,
}
try:
    dt = _parse_dtype(TORCH_DTYPE)
    if dt is not None:
        _embedder_kwargs["dtype"] = dt
except Exception as e:
    raise RuntimeError(str(e))

if ATN_IMPL:
    _embedder_kwargs["attn_implementation"] = ATN_IMPL

# 全局单例（避免每次请求都加载）
embedder = Qwen3VLEmbedder(**_embedder_kwargs)


def _validate_dimensions(dim: Optional[int]) -> int:
    if dim is None:
        return DEFAULT_DIMENSIONS
    if not (MIN_DIMENSIONS <= dim <= MAX_DIMENSIONS):
        raise HTTPException(status_code=422, detail=f"dimensions must be in [{MIN_DIMENSIONS}, {MAX_DIMENSIONS}]")
    return int(dim)


def _postprocess(emb: torch.Tensor, dimensions: int, normalize: bool) -> torch.Tensor:
    # emb: [B, D]
    if emb.dim() != 2:
        raise HTTPException(status_code=500, detail=f"Unexpected embedding shape: {tuple(emb.shape)}")
    if emb.size(-1) < dimensions:
        raise HTTPException(status_code=500, detail=f"Model dim {emb.size(-1)} < requested {dimensions}")
    emb = emb[:, :dimensions]
    if normalize:
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "default_dimensions": DEFAULT_DIMENSIONS,
        "attn_impl": ATN_IMPL or None,
        "torch_dtype": TORCH_DTYPE or None,
    }


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    # 统一转成 DeepWiki 要求的 inputs: List[Dict] :contentReference[oaicite:9]{index=9}
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

    embedder_inputs: List[Dict[str, Any]] = []
    dims: List[int] = []
    norms: List[bool] = []
    for it in items:
        embedder_inputs.append(it.to_embedder_dict())
        dims.append(_validate_dimensions(it.dimensions))
        norms.append(bool(it.normalize))

    # DeepWiki：model.process(inputs) -> tensor [B, D] :contentReference[oaicite:10]{index=10}
    with torch.no_grad():
        emb = embedder.process(embedder_inputs)

    # 按 item 的 dimensions/normalize 做后处理（允许每条不一样）
    data = []
    for i in range(emb.size(0)):
        e = _postprocess(emb[i : i + 1], dims[i], norms[i])[0]
        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": e.detach().cpu().float().tolist(),
            }
        )

    return {"object": "list", "model": req.model, "data": data}