import base64
import io
import os
from typing import List, Optional, Union

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoProcessor, Qwen3VLModel

# =========================
# 基础配置
# =========================
MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 加载模型 & Processor
# =========================
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = Qwen3VLModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

# =========================
# FastAPI
# =========================
app = FastAPI(title="Qwen3-VL-Embedding Service")

# =========================
# 请求模型
# =========================
class EmbeddingInput(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str], EmbeddingInput]


# =========================
# 工具函数
# =========================
def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
    """
    标准 embedding mean pooling
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted


@torch.no_grad()
def compute_embedding(
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> np.ndarray:

    if text is None and image is None:
        raise HTTPException(status_code=422, detail="text or image required")

    # ===== 关键：构造 conversation =====
    if image is not None:
        content = [
            {"type": "image"},
            {"type": "text", "text": text or ""}
        ]
    else:
        content = [
            {"type": "text", "text": text}
        ]

    conversation = [
        {
            "role": "user",
            "content": content
        }
    ]

    # ===== 关键：用 chat template =====
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )

    # ===== processor 编码 =====
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    emb = mean_pooling(last_hidden, attention_mask)
    emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb[0].cpu().numpy()


# =========================
# API
# =========================
@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    results = []

    # ---------- 文本批量 ----------
    if isinstance(req.input, list) and all(isinstance(x, str) for x in req.input):
        for idx, text in enumerate(req.input):
            emb = compute_embedding(text=text)
            results.append({
                "object": "embedding",
                "index": idx,
                "embedding": emb.tolist()
            })

    # ---------- 单文本 ----------
    elif isinstance(req.input, str):
        emb = compute_embedding(text=req.input)
        results.append({
            "object": "embedding",
            "index": 0,
            "embedding": emb.tolist()
        })

    # ---------- 图文 ----------
    elif isinstance(req.input, EmbeddingInput):
        image = None
        if req.input.image_base64:
            image = decode_base64_image(req.input.image_base64)

        emb = compute_embedding(
            text=req.input.text,
            image=image
        )

        results.append({
            "object": "embedding",
            "index": 0,
            "embedding": emb.tolist()
        })

    else:
        raise HTTPException(status_code=422, detail="Unsupported input format")

    return {
        "object": "list",
        "model": req.model,
        "data": results
    }


@app.get("/health")
def health():
    return {"status": "ok"}
