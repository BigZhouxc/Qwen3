#server.py
import os
import torch
import base64, io
import httpx

from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
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


MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))  # 20MB
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(200 * 1024 * 1024)))  # 200MB
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "10.0"))



def _is_http_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False


def _pil_decode(image_bytes: bytes) -> Image.Image:
    # 3) 确保 PIL 能够解码（并且文件未损坏）
    # verify() 会“验证但不解码成像素”，并且会消耗/关闭流，所以需要 reopen
    try:
        bio = io.BytesIO(image_bytes)
        img = Image.open(bio)
        img.verify()

        bio2 = io.BytesIO(image_bytes)
        img2 = Image.open(bio2).convert("RGB")
        return img2
    except Exception as e:
        raise ValueError(f"PIL decode/verify failed: {e}")


def _load_image_from_url(url: str) -> Image.Image:
    # 2) 检查远程 URL 可访问性 + 下载 bytes
    with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT) as client:
        # 先 HEAD（有些服务器不支持 HEAD，会 405）
        try:
            h = client.head(url)
            if h.status_code >= 400:
                # 有的站点 HEAD 会拒绝，继续走 GET
                pass
            else:
                ct = h.headers.get("content-type", "")
                if ct and ("image" not in ct.lower()):
                    raise ValueError(f"URL content-type not image: {ct}")
                cl = h.headers.get("content-length")
                if cl and int(cl) > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image too large: {cl} bytes > {MAX_IMAGE_BYTES}")
        except Exception:
            # HEAD 失败不直接判死，继续 GET
            pass

        r = client.get(url)
        r.raise_for_status()

        ct = r.headers.get("content-type", "")
        if ct and ("image" not in ct.lower()):
            raise ValueError(f"URL content-type not image: {ct}")

        content = r.content
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image too large: {len(content)} bytes > {MAX_IMAGE_BYTES}")

    return _pil_decode(content)


def _load_image_from_path(path_str: str) -> Image.Image:
    # 1) 验证路径存在且可读 + PIL 解码
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Local image path not found: {path_str}")
    if not p.is_file():
        raise ValueError(f"Local image path is not a file: {path_str}")
    try:
        data = p.read_bytes()
    except Exception as e:
        raise PermissionError(f"Local image path not readable: {path_str}, err={e}")

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {len(data)} bytes > {MAX_IMAGE_BYTES}")

    return _pil_decode(data)


def validate_and_load_image(image_field):
    """
    支持：
      - str(url 或 本地路径)
      - PIL.Image.Image
      - list[str|PIL.Image.Image]
    返回：
      - PIL.Image.Image 或 list[PIL.Image.Image]
    """
    if isinstance(image_field, Image.Image):
        # 已经是 PIL：也做一次“可解码”校验（最稳是转 bytes 再 verify）
        try:
            buf = io.BytesIO()
            image_field.save(buf, format="PNG")  # 用 PNG 做中转，不依赖原始编码
            return _pil_decode(buf.getvalue())
        except Exception as e:
            raise ValueError(f"Provided PIL image not decodable: {e}")

    if isinstance(image_field, str):
        if _is_http_url(image_field):
            return _load_image_from_url(image_field)
        return _load_image_from_path(image_field)

    if isinstance(image_field, list):
        out = []
        for x in image_field:
            out.append(validate_and_load_image(x))
        return out

    raise TypeError(f"Unsupported image type: {type(image_field)}")


def _check_remote_url_accessible(url: str, kind: str) -> None:
    """
    kind: 'image' or 'video'（这里只用 video，但保留通用性）
    做到：URL 可连通、HTTP 状态码 OK、(可选) content-type 合理、(可选) content-length 不超限
    """
    with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT) as client:
        # 先 HEAD（可能 405 / 不支持）
        head_ok = False
        try:
            h = client.head(url)
            if 200 <= h.status_code < 400:
                head_ok = True
                ct = (h.headers.get("content-type") or "").lower()
                if ct:
                    if kind == "video" and ("video" not in ct and "octet-stream" not in ct):
                        # 很多站会给 application/octet-stream，放行
                        raise ValueError(f"URL content-type not video: {ct}")
                cl = h.headers.get("content-length")
                if cl and int(cl) > MAX_VIDEO_BYTES:
                    raise ValueError(f"Video too large: {cl} bytes > {MAX_VIDEO_BYTES}")
        except Exception:
            # HEAD 不可靠，不直接判死，继续 GET 轻量探测
            pass

        # GET 轻量探测：只拉一小段（Range），避免把整视频下载下来
        # 有些服务器不支持 Range，会返回 200 全量；我们仍然只做“可访问性”判断
        headers = {"Range": "bytes=0-1023"}  # 只取前 1KB
        r = client.get(url, headers=headers)

        # 可接受 200（无 Range）或 206（Partial Content）
        if r.status_code not in (200, 206):
            r.raise_for_status()

        ct = (r.headers.get("content-type") or "").lower()
        if ct:
            if kind == "video" and ("video" not in ct and "octet-stream" not in ct):
                raise ValueError(f"URL content-type not video: {ct}")

        cl = r.headers.get("content-length")
        # 对 206 来说 content-length 可能只是片段长度；对 200 可能是全量
        if cl and r.status_code == 200:
            if int(cl) > MAX_VIDEO_BYTES:
                raise ValueError(f"Video too large: {cl} bytes > {MAX_VIDEO_BYTES}")


def _check_local_path_readable(path_str: str, kind: str) -> None:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Local {kind} path not found: {path_str}")
    if not p.is_file():
        raise ValueError(f"Local {kind} path is not a file: {path_str}")
    try:
        # 只验证可读，不读全量（视频可能很大）
        with p.open("rb") as f:
            f.read(1)
    except Exception as e:
        raise PermissionError(f"Local {kind} path not readable: {path_str}, err={e}")

    # 可选：用 stat 做个大小上限
    try:
        size = p.stat().st_size
        if kind == "video" and size > MAX_VIDEO_BYTES:
            raise ValueError(f"Video too large: {size} bytes > {MAX_VIDEO_BYTES}")
    except Exception:
        pass


def validate_video_reference(video_field):
    """
    只做“引用可用性”校验，不做解码。
    支持：
      - str(url 或 本地路径)
      - list[frames]（服务端内部调用才可能出现；HTTP JSON 传不过来 PIL）
    返回原值（embedder 仍按它自己支持的方式读取）。
    """
    if isinstance(video_field, str):
        if _is_http_url(video_field):
            _check_remote_url_accessible(video_field, kind="video")
            return video_field
        _check_local_path_readable(video_field, kind="video")
        return video_field

    if isinstance(video_field, list):
        # 这是“预抽帧”场景：一般是 list[PIL.Image]。
        # HTTP JSON 传不过来，但服务端内部调用可用；这里不做更多校验
        if len(video_field) == 0:
            raise ValueError("video frame list is empty")
        return video_field

    raise TypeError(f"Unsupported video type: {type(video_field)}")


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

class EmbeddingItem(BaseModel):
    # 允许 PIL（但注意：HTTP JSON 调用传不过来）
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    text: Optional[str] = None

    # DeepWiki: image 可以是 URL / 本地路径 / PIL / 多图 list :contentReference[oaicite:2]{index=2}
    image: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None

    # DeepWiki: video 可以是 URL / 本地路径 / 或 frame list（frame 通常是 PIL） :contentReference[oaicite:3]{index=3}
    video: Optional[Any] = None
    fps: Optional[float] = None
    max_frames: Optional[int] = None

    instruction: Optional[str] = None

    # 你的扩展
    image_base64: Optional[str] = None
    dimensions: Optional[int] = Field(default=None)
    normalize: bool = True

    def to_embedder_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        if self.text:
            d["text"] = self.text

        # instruction
        d["instruction"] = self.instruction or DEFAULT_INSTRUCTION

        # video + sampling params
        if self.video is not None:
            try:
                d["video"] = validate_video_reference(self.video)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid video: {e}")

            if self.fps is not None:
                d["fps"] = float(self.fps)
            if self.max_frames is not None:
                d["max_frames"] = int(self.max_frames)
            if self.fps is not None:
                d["fps"] = float(self.fps)
            if self.max_frames is not None:
                d["max_frames"] = int(self.max_frames)

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
            try:
                d["image"] = validate_and_load_image(self.image)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid image: {e}")

        if not any(k in d for k in ("text", "image", "video")):
            raise HTTPException(status_code=422, detail="Each item must include at least one of: text/image/video")

        return d


class EmbeddingRequest(BaseModel):
    model: str = "Qwen3-VL-Embedding-8B"

    # 关键：别用 Union 去抢类型，直接 Any，然后你自己 normalize
    input: Any


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
    items: List[EmbeddingItem]

    inp = req.input

    # 1) input 是 str
    if isinstance(inp, str):
        items = [EmbeddingItem(text=inp)]

    # 2) input 是 list
    elif isinstance(inp, list):
        # 2.1 list[str]
        if all(isinstance(x, str) for x in inp):
            items = [EmbeddingItem(text=x) for x in inp]
        else:
            # 2.2 list[dict] or list[mixed]
            try:
                items = [
                    x if isinstance(x, EmbeddingItem) else EmbeddingItem.model_validate(x)
                    for x in inp
                ]
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid input list items: {e}")

    # 3) input 是 dict（单条）
    elif isinstance(inp, dict):
        try:
            items = [EmbeddingItem.model_validate(inp)]
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid input object: {e}")

    # 4) input 已经是 EmbeddingItem（很少见，通常只会在服务端内部调用）
    elif isinstance(inp, EmbeddingItem):
        items = [inp]

    else:
        raise HTTPException(status_code=422, detail="Unsupported input format")

    # ---- 后面你的逻辑保持不变 ----
    embedder_inputs: List[Dict[str, Any]] = []
    dims: List[int] = []
    norms: List[bool] = []
    for idx, it in enumerate(items):
        try:
            embedder_inputs.append(it.to_embedder_dict())
        except HTTPException as e:
            # 把失败的 item 下标带回给客户端
            raise HTTPException(status_code=e.status_code, detail=f"input[{idx}] {e.detail}")

        # 只有成功生成 embedder dict 才去收集 dims/normalize
        dims.append(_validate_dimensions(it.dimensions))
        norms.append(bool(it.normalize))

    with torch.no_grad():
        emb = embedder.process(embedder_inputs)

    data = []
    for i in range(emb.size(0)):
        e = _postprocess(emb[i:i+1], dims[i], norms[i])[0]
        data.append({"object": "embedding", "index": i, "embedding": e.detach().cpu().float().tolist()})

    return {"object": "list", "model": req.model, "data": data}
