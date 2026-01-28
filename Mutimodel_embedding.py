import base64
from pathlib import Path
import httpx

BASE_URL = "http://xxxxx:38049"

def img_to_base64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")

def post_embeddings(payload: dict, timeout: float = 300.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{BASE_URL}/v1/embeddings", json=payload)
        r.raise_for_status()
        return r.json()

def print_dims(resp: dict):
    dims = [len(x["embedding"]) for x in resp["data"]]
    print("dims:", dims)

#========单文本==============
payload = {
    "model": "Qwen3-VL-Embedding-8B",
    "input": "北京是中国的首都。"
}
resp = post_embeddings(payload)
print_dims(resp)
print("first 5:", resp["data"][0]["embedding"][:5])

#==========多文本=============
payload = {
    "model": "Qwen3-VL-Embedding-8B",
    "input": [
        "我喜欢机器学习。",
        "Embedding 常用于语义检索。"
    ]
}
resp = post_embeddings(payload)
print_dims(resp)

#==========单图片==============
img_b64 = img_to_base64("./test.jpg")

payload = {
    "model": "Qwen3-VL-Embedding-8B",
    "input": {
        "image_base64": img_b64,
        "instruction": "Represent the image",
        "dimensions": 2048,     # 可选：1024/2048/4096...
        "normalize": True
    }
}
resp = post_embeddings(payload)
print_dims(resp)

#==========多图片==============
img1 = img_to_base64("./a.jpg")
img2 = img_to_base64("./b.jpg")

payload = {
    "model": "Qwen3-VL-Embedding-8B",
    "input": [
        {"image_base64": img1, "instruction": "Represent the image", "dimensions": 1024},
        {"image_base64": img2, "instruction": "Represent the image", "dimensions": 1024},
    ]
}
resp = post_embeddings(payload)
print_dims(resp)

#==========单条多模态==============
img_b64 = img_to_base64("./test.jpg")

payload = {
    "model": "Qwen3-VL-Embedding-8B",
    "input": {
        "text": "这张图的主要内容是什么？",
        "image_base64": img_b64,
        "instruction": "Represent the user's input",
        "dimensions": 2048,
        "normalize": True
    }
}
resp = post_embeddings(payload)
print_dims(resp)

#===========多条多模态==============
img_b64 = img_to_base64("./test.jpg")

payload = {
    "model": "Qwen3-VL-Embedding-8B",
    "input": [
        {"text": "纯文本样例", "dimensions": 2048, "normalize": True},
        {"image_base64": img_b64, "instruction": "Represent the image", "dimensions": 2048},
        {"text": "图文一起", "image_base64": img_b64, "instruction": "Represent the user's input", "dimensions": 2048}
    ]
}
resp = post_embeddings(payload)
print_dims(resp)


import numpy as np
img1 = img_to_base64("./a.jpg")
img2 = img_to_base64("./a.jpg")
# 两段文本 embedding
payload = {"model": "Qwen3-VL-Embedding-2B",      "input": [
       {"image_base64": img1},
       {"image_base64": img2},
    ]}
resp = post_embeddings(payload)

e1 = np.array(resp["data"][0]["embedding"], dtype=np.float32)
e2 = np.array(resp["data"][1]["embedding"], dtype=np.float32)

cos = float(np.dot(e1, e2))  # normalize=True 时等价于 cosine
print("cosine:", cos)
