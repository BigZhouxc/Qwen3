import base64
import requests
from PIL import Image
import io
from minio import Minio
from pathlib import Path
import uuid
import httpx
# ======================
# 基本配置
# ======================
CHAT_BASE_URL = "http://xxxx:38043"
# EMBED_BASE_URL = "http://xxxx:38042"
EMBED_BASE_URL = "http://123.181.192.132:38042"

CHAT_MODEL = "Qwen3-VL-8B-Instruct"
EMBED_MODEL = "Qwen3-VL-Embedding-8B"

MINIO_ENDPOINT = "xxxxx:38044"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "videos"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)
# ======================
# 工具函数
# ======================
def upload_video(video_path: str) -> str:
    video_path = Path(video_path)
    assert video_path.exists(), f"{video_path} not exists"

    object_name = f"{uuid.uuid4().hex}_{video_path.name}"

    client.fput_object(
        BUCKET,
        object_name,
        str(video_path),
        content_type="video/mp4",
    )

    return f"http://{MINIO_ENDPOINT}/{BUCKET}/{object_name}"
    # return f"http://xxxx:38044/{BUCKET}/{object_name}"

def image_to_base64(image_path: str) -> str:
    """读取图片并转成 base64"""
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ======================
# 1. 测试 Chat（纯文本）
# ======================
def test_chat_text():
    print("\n=== Chat 文本测试 ===")

    url = f"{CHAT_BASE_URL}/v1/chat/completions"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个专业的 AI 助手"},
            {"role": "user", "content": "用一句话介绍一下 NVIDIA A100"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    print("模型回复：")
    print(data["choices"][0]["message"]["content"])


# ======================
# 2. 测试 Chat（图文）
# ======================
def test_chat_image(image_path: str):
    print("\n=== Chat 图文测试 ===")

    image_base64 = image_to_base64(image_path)

    url = f"{CHAT_BASE_URL}/v1/chat/completions"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这个图片的内容"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043"
                                   "/keepme/image/receipt.png "
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    print("模型回复：")
    print(data["choices"][0]["message"]["content"])

# ======================
# 3. 测试 Chat (视频)
# ======================
def chat_with_video(video_url: str):
    url = f"{CHAT_BASE_URL}/v1/chat/completions"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这个视频的内容"},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url}
                    }
                ]
            }
        ],
        "max_tokens": 1024,
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    print("模型回复：")
    print(data["choices"][0]["message"]["content"])

# ======================
# 4. 测试 Embedding（文本）
# ======================
def test_embedding_text():
    print("\n=== Embedding 文本测试 ===")

    url = f"{EMBED_BASE_URL}/v1/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "input": [
            "北京是中国的首都",
            "苹果是一种水果"
        ]
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    for item in data["data"]:
        emb = item["embedding"]
        print(f"文本 {item['index']} 向量维度：{len(emb)}")
        print(f"前 5 个值：{emb[:5]}")



# ======================
# 5. 测试 Embedding（图文）
# ======================
import numpy as np
def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec  # 避免除零错误

def test_embedding_image(image_path: str):

    print("\n=== Embedding 图文测试 ===")
    with open("hehua.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
    img_hehua_64 = content.split(",", 1)[1] if "," in content else content

    image_base64 = image_to_base64(image_path)
    img1 = image_to_base64("./1.png")
    img2 = image_to_base64("./2.jpg")
    img3 = image_to_base64("./3.jpg")
    img4 = image_to_base64("./4.jpg")
    img5 = image_to_base64("./5.jpg")
    img6 = image_to_base64("./6.jpg")

    payload = {
        "model": EMBED_MODEL,
        "input": [
            {"text": "荷花"},
            {"text": "收据"},
        ]
    }
    payload_image = {
        "model": EMBED_MODEL,
        "input": [
            {"image_base64": img1},
            {"image_base64": img2},
            {"image_base64": img3},
            {"image_base64": img4},
            {"image_base64": img5},
            {"image_base64": img6},
            {"image_base64": img_hehua_64},
        ]
    }


    resp = requests.post(
        f"{EMBED_BASE_URL}/v1/embeddings",
        json=payload,
        timeout=300
    )
    resp.raise_for_status()

    resp_image = requests.post(
        f"{EMBED_BASE_URL}/v1/embeddings",
        json=payload_image,
        timeout=300
    )
    resp_image.raise_for_status()

    data = resp.json()
    data_image = resp_image.json()

    e1 = data["data"][0]["embedding"][:2048]
    e1 = l2_normalize(e1)
    print("文本向量维度：", len(e1))
    e2 = data["data"][1]["embedding"][:2048]
    e2 = l2_normalize(e2)
    e3 = data_image["data"][0]["embedding"][:2048]
    e4 = data_image["data"][1]["embedding"][:2048]
    e5 = data_image["data"][2]["embedding"][:2048]
    e6 = data_image["data"][3]["embedding"][:2048]
    e7 = data_image["data"][4]["embedding"][:2048]
    e8 = data_image["data"][5]["embedding"][:2048]
    e8 = l2_normalize(e8)
    e9 = data_image["data"][6]["embedding"][:2048]
    e9 = l2_normalize(e9)
    #将e1-e7embedding都存入文本文件中，在前面加上标识
    # with open("embedding_8b_add.txt", "w") as f:
    #     f.write("=======8B 2048=========")
    #     f.write("\n")
    #     f.write("e1:")
    #     f.write(str(e1))
    #     f.write("\n")
    #     f.write("e2:")
    #     f.write(str(e2))
    #     # f.write("\n")
    #     # f.write("e3:")
    #     # f.write(str(e3))
    #     # f.write("\n")
    #     # f.write("e4:")
    #     # f.write(str(e4))
    #     # f.write("\n")
    #     # f.write("e5:")
    #     # f.write(str(e5))
    #     # f.write("\n")
    #     # f.write("e6:")
    #     # f.write(str(e6))
    #     # f.write("\n")
    #     # f.write("e7:")
    #     # f.write(str(e7))
    #     f.write("\n")
    #     f.write("e8:")
    #     f.write(str(e8))


    #
    # 将e1，e2分别和e3，e4，e5进行计算相似度
    # cos1 = float(np.dot(e1, e3))
    # print("e1==e3:", cos1)
    # cos2 = float(np.dot(e1, e4))
    # print("e1==e4:", cos2)
    # cos3 = float(np.dot(e1, e5))
    # print("e1==e5:", cos3)
    # cos4 = float(np.dot(e1, e6))
    # print("e1==e6:", cos4)
    # cos5 = float(np.dot(e1, e7))
    # print("e1==e7:", cos5)
    # cos11 = float(np.dot(e1, e8))
    # print("e1==e8:", cos11)
    #
    # cos6 = float(np.dot(e2, e3))  # normalize=True 时等价于 cosine
    # print("e2==e3:", cos6)
    # cos7 = float(np.dot(e2, e4))
    # print("e2==e4:", cos7)
    # cos8 = float(np.dot(e2, e5))
    # print("e2==e5:", cos8)
    # cos9 = float(np.dot(e2, e6))
    # print("e2==e6:", cos9)
    # cos10 = float(np.dot(e2, e7))
    # print("e2==e7:", cos10)
    # cos12 = float(np.dot(e2, e8))
    # print("e2==e8:", cos12)
    cos13 = float(np.dot(e1, e6))
    print("e1==e9:", cos13)


# ======================
# 主入口
# ======================
if __name__ == "__main__":
    # 1. Chat 文本
    # test_chat_text()

    # 2. Chat 图文
    # test_chat_image(r"F:\workspace_mine\dify-data\u.jpg")

    # 3. chat 视频
    # video_url = upload_video(r"F:\workspace_mine\dify-data\free-videos.mp4")
    # print(video_url)
    # chat_with_video(video_url)

    # 4. Embedding 文本
    # test_embedding_text()

    # 5. Embedding 图文
    test_embedding_image(r"F:\workspace_mine\dify-data\u.jpg")




