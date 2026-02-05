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
    dtype=DTYPE,
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
    input: Union[
        str,
        List[str],                 # ✅文本批量（OpenAI）
        EmbeddingInput,            # ✅单图文
        List[EmbeddingInput]       # ✅图文批量
    ]


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


@torch.no_grad()
def compute_embedding_batch(
    texts: List[str],
    images: Optional[List[Image.Image]] = None
) -> np.ndarray:
    """
    一次 forward 支持 batch（图文混合）
    return: (batch, hidden_dim)
    """

    batch_size = len(texts)

    # ===== 构造 conversations =====
    conversations = []
    for i in range(batch_size):
        if images and images[i] is not None:
            content = [
                {"type": "image"},
                {"type": "text", "text": texts[i]}
            ]
        else:
            content = [{"type": "text", "text": texts[i]}]

        conversations.append({"role": "user", "content": content})

    # ===== apply_chat_template 批量 prompt =====
    prompts = [
        processor.apply_chat_template(
            [conv],
            tokenize=False,
            add_generation_prompt=False
        )
        for conv in conversations
    ]

    # ===== processor batch encode =====
    inputs = processor(
        text=prompts,
        images=images if images is not None else None,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # ===== 一次 forward =====
    outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    # ===== mean pooling =====
    emb = mean_pooling(last_hidden, attention_mask)

    # normalize
    emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb.cpu().numpy()


# =========================
# API
# =========================
@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):

    # ---------- Case 1 ----------
    if isinstance(req.input, str):
        embs = compute_embedding_batch([req.input])

    # ---------- Case 2 ----------
    elif isinstance(req.input, list) and all(isinstance(x, str) for x in req.input):
        embs = compute_embedding_batch(req.input)

    # ---------- Case 3 ----------
    elif isinstance(req.input, EmbeddingInput):
        img = decode_base64_image(req.input.image_base64) if req.input.image_base64 else None
        embs = compute_embedding_batch([req.input.text or ""], [img] if img else None)

    # ---------- Case 4 ✅混合批量 ----------
    elif isinstance(req.input, list) and all(isinstance(x, EmbeddingInput) for x in req.input):

        text_texts, text_indices = [], []
        img_texts, img_images, img_indices = [], [], []

        for idx, item in enumerate(req.input):
            if item.image_base64:
                img_indices.append(idx)
                img_texts.append(item.text or "")
                img_images.append(decode_base64_image(item.image_base64))
            else:
                text_indices.append(idx)
                text_texts.append(item.text or "")

        embeddings = [None] * len(req.input)

        if text_texts:
            text_embs = compute_embedding_batch(text_texts)
            for i, j in enumerate(text_indices):
                embeddings[j] = text_embs[i]

        if img_texts:
            img_embs = compute_embedding_batch(img_texts, img_images)
            for i, j in enumerate(img_indices):
                embeddings[j] = img_embs[i]

        embs = embeddings

    else:
        raise HTTPException(status_code=422, detail="Unsupported input format")

    # ---------- OpenAI response ----------
    data = []
    for idx, emb in enumerate(embs):
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": emb.tolist()
        })

    return {
        "object": "list",
        "model": req.model,
        "data": data
    }


@app.get("/health")
def health():
    return {"status": "ok"}
