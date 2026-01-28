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
EMBED_BASE_URL = "http://xxxx:38042"

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
def test_embedding_image(image_path: str):
    print("\n=== Embedding 图文测试 ===")

    image_base64 = image_to_base64(image_path)

    payload = {
        "model": EMBED_MODEL,
        "input": [{
            "text": "这张图片里有什么？",
            "image_base64": image_base64
        },
        {
            "text": "这张图片里有什么？",
            "image_base64": image_base64
        },
        ]
    }

    resp = requests.post(
        f"{EMBED_BASE_URL}/v1/embeddings",
        json=payload,
        timeout=300
    )
    resp.raise_for_status()

    data = resp.json()
    emb = data["data"][0]["embedding"]

    print("embedding 维度:", len(emb))
    print("前 5 个值:", emb[:5])




# ======================
# 主入口
# ======================
if __name__ == "__main__":
    # 1. Chat 文本
    # test_chat_text()

    # 2. Chat 图文
    test_chat_image(r"F:\workspace_mine\dify-data\u.jpg")

    # 3. chat 视频
    # video_url = upload_video(r"F:\workspace_mine\dify-data\free-videos.mp4")
    # print(video_url)
    # chat_with_video(video_url)

    # 4. Embedding 文本
    # test_embedding_text()

    # 5. Embedding 图文
    # test_embedding_image(r"F:\workspace_mine\dify-data\u.jpg")




