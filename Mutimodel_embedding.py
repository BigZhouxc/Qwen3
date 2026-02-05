import base64
import json
from pathlib import Path

from PIL import Image

import httpx

BASE_URL = "http://123.181.192.132:38049"

def img_to_base64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")

def post_embeddings(payload: dict, timeout: float = 300.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{BASE_URL}/v1/embeddings", json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 关键：把服务端的错误 body 打出来
            print("Status:", e.response.status_code)
            ct = e.response.headers.get("content-type", "")
            if "application/json" in ct:
                print("Error JSON:", json.dumps(e.response.json(), ensure_ascii=False, indent=2))
            else:
                print("Error text:", e.response.text)
            raise
        return r.json()

def print_dims(resp: dict):
    dims = [len(x["embedding"]) for x in resp["data"]]
    print("dims:", dims)

# #========单文本==============
# payload = {
#     "model": "Qwen3-VL-Embedding-8B",
#     "input": "北京是中国的首都。"
# }
# resp = post_embeddings(payload)
# print_dims(resp)
# print("first 5:", resp["data"][0]["embedding"][:5])
#
# #==========多文本=============
# payload = {
#     "model": "Qwen3-VL-Embedding-8B",
#     "input": [
#         "我喜欢机器学习。",
#         "Embedding 常用于语义检索。"
#     ]
# }
# resp = post_embeddings(payload)
# print_dims(resp)
#
# #==========单图片==============
# img_b64 = img_to_base64("./test.jpg")
#
# payload = {
#     "model": "Qwen3-VL-Embedding-2B",
#     "input": {
#         "image_base64": img_b64,
#         "instruction": "Represent the image",
#         "dimensions": 2048,     # 可选：1024/2048/4096...
#         "normalize": True
#     }
# }
# resp = post_embeddings(payload)
# print_dims(resp)
#
# #==========多图片==============
img1 = img_to_base64("./a.jpg")
img2 = img_to_base64("./b.jpg")

payload = {
    "model": "Qwen3-VL-Embedding-2B",
    "input": [
        {"text":"i love you"},
        {"text":"i heat you"},
        {"image_base64": img1, "instruction": "Represent the image", "dimensions": 1024},
        {"image_base64": img2, "instruction": "Represent the image", "dimensions": 1024},
        {"image": "http://172.17.0.1:38044/water/static/20260127/21a021fb920747c9a7a6432e8ccc4ea9.jpg"},
        {"image": "./d.png"},
        # {"image": Image.open("./3.jpg")},#AttributeError: module 'PIL' has no attribute 'Image'
        {"image": ["./2.jpg", "./3.jpg"]}  # Multiple images
    ]
}
resp = post_embeddings(payload)
print_dims(resp)
#
# #==========单条多模态==============
# img_b64 = img_to_base64("./test.jpg")
#
# payload = {
#     "model": "Qwen3-VL-Embedding-8B",
#     "input": {
#         "text": "这张图的主要内容是什么？",
#         "image_base64": img_b64,
#         "instruction": "Represent the user's input",
#         "dimensions": 2048,
#         "normalize": True
#     }
# }
# resp = post_embeddings(payload)
# print_dims(resp)
#
# #===========多条多模态==============
# img_b64 = img_to_base64("./test.jpg")
#
# payload = {
#     "model": "Qwen3-VL-Embedding-8B",
#     "input": [
#         {"text": "纯文本样例", "dimensions": 2048, "normalize": True},
#         {"image_base64": img_b64, "instruction": "Represent the image", "dimensions": 2048},
#         {"text": "图文一起", "image_base64": img_b64, "instruction": "Represent the user's input", "dimensions": 2048}
#     ]
# }
# resp = post_embeddings(payload)
# print_dims(resp)

# ============视频embedding===============
# payload = {
#     "model": "Qwen3-VL-Embedding-2B",
#     "input": [
#         {"text":"i love you"},
#         {"text":"i heat you"},
#         {
#             "video": "http://172.17.0.1:38044/videos/0395712d1c9b45f7a1147dcb25b4c858_cap.mp4",
#             "fps": 2.0,  # Sample 2 frames per second
#             "max_frames": 32  # Maximum 32 frames total
#         },
#         # {
#         #     "video": r"./free-vedios.mp4"  # Uses default fps=1.0, max_frames=64
#         # },
#     ]
# }
# resp = post_embeddings(payload)
# print_dims(resp)



# import json
# import base64
# from pathlib import Path
# import httpx
# import numpy as np
#
# OUT_JSONL = "embeddings_single.jsonl"
# MODEL = "Qwen3-VL-Embedding-2B"
#
# def post_embeddings_one(item: dict, timeout: float = 300.0) -> dict:
#     """
#     单条请求：payload['input'] 只包含一个元素
#     item 例子：{"text":"荷花"} 或 {"image_base64": "..."}
#     """
#     payload = {"model": MODEL, "input": [item]}
#     with httpx.Client(timeout=timeout) as client:
#         r = client.post(f"{BASE_URL}/v1/embeddings", json=payload)
#         r.raise_for_status()
#         return r.json()
#
# def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     n = np.linalg.norm(x)
#     return x / (n + eps)
#
# def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#     a = l2_normalize(a.astype(np.float32))
#     b = l2_normalize(b.astype(np.float32))
#     return float(np.dot(a, b))
#
# # ========= 准备数据 =========
# img1 = img_to_base64("./1.png")
# img2 = img_to_base64("./2.jpg")
# img3 = img_to_base64("./3.jpg")
# img4 = img_to_base64("./4.jpg")
# img5 = img_to_base64("./5.jpg")
# img6 = img_to_base64("./6.jpg")
#
# with open("hehua.txt", "r", encoding="utf-8") as f:
#     content = f.read().strip()
# img_hehua_64 = content.split(",", 1)[1] if "," in content else content
#
# items = [
#     ("e1_text_hehua", {"text": "荷花"}),
#     ("e2_text_shouju", {"text": "收据"}),
#     ("e3_img1", {"image_base64": img1}),
#     ("e4_img2", {"image_base64": img2}),
#     ("e5_img3", {"image_base64": img3}),
#     ("e6_img4", {"image_base64": img4}),
#     ("e7_img5", {"image_base64": img5}),
#     ("e8_img6", {"image_base64": img6}),
#     ("e9_img_hehua", {"image_base64": img_hehua_64}),
# ]
#
# # ========= 1) 循环单条请求并保存 =========
# embeddings = {}  # name -> np.ndarray
# for name, item in items:
#     resp = post_embeddings_one(item)
#     emb = np.array(resp["data"][0]["embedding"], dtype=np.float32)
#     embeddings[name] = emb
#
# with open(OUT_JSONL, "w", encoding="utf-8") as f:
#     for name, item in items:
#         resp = post_embeddings_one(item)
#         emb = np.array(resp["data"][0]["embedding"], dtype=np.float32)
#         embeddings[name] = emb
#
#         record = {
#             "name": name,
#             "input": item,
#             "dim": int(len(emb)),
#             "embedding": emb.tolist(),
#         }
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")
#
# print(f"Saved {len(embeddings)} embeddings to {OUT_JSONL}")
# print("dims:", {k: len(v) for k, v in embeddings.items()})
#
# # ========= 2) 循环计算相似度（cosine） =========
# e1 = embeddings["e1_text_hehua"]
#
# e2 = embeddings["e6_img4"]
# e3 = embeddings["e9_img_hehua"]
# print("e1==e2:", cosine_sim(e1, e2))
# print("e1==e3:", cosine_sim(e1, e3))
#
# img_keys = ["e3_img1", "e4_img2", "e5_img3", "e6_img4", "e7_img5", "e8_img6"]


