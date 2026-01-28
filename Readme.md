## 代码结构解释
1. 最外层的docker-compose及Dockerfile文件都是能够正常运行的，其中chat使用的是vllm框架，
embedding使用的是transfors框架
2. embedding_vllm文件夹内的服务使用vllm框架,但是要求vllm框架>=0.14,导致环境不匹配没有启动成功
3. embedding_test文件夹内的服务使用的是多模态和单模态分开处理的方案，能够正常启动

## Embedding API

```text
http://XXXXX:38049
```

## `GET /health`

**说明**
 用于检查服务是否存活、模型配置是否正确。

**Response**

```json
{
  "status": "ok",
  "model_path": "/model",
  "device": "cuda",
  "default_dimensions": 2048,
  "attn_impl": null,
  "torch_dtype": "float16"
}
```

## `POST /v1/embeddings`

> 兼容 **OpenAI Embeddings API**
> 支持 **文本 / 图片 / 多模态 / 批量**
> 支持 **MRL 动态维度裁剪**

---

**Request** Schema

```json
{
  "model": "Qwen3-VL-Embedding-2B | Qwen3-VL-Embedding-8B",
  "input": string | string[] | EmbeddingItem | EmbeddingItem[]
}
```

**EmbeddingItem** 结构

```json
{
  "text": "可选，文本输入",
  "image_base64": "可选，base64 编码图片",
  "video": "可选（预留）",
  "instruction": "可选，默认：Represent the user's input",
  "dimensions": 1024 | 2048 | 4096,
  "normalize": true
}
```

**字段说明**

| 字段           | 类型   | 说明                                    |
| -------------- | ------ | --------------------------------------- |
| `text`         | string | 文本输入                                |
| `image_base64` | string | Base64 图片（支持 data:image/... 前缀） |
| `instruction`  | string | embedding 指令                          |
| `dimensions`   | int    | 输出维度（MRL 裁剪）                    |
| `normalize`    | bool   | 是否 L2 normalize（默认 true）          |

⚠️ **每个 EmbeddingItem 至少包含一个：`text / image / video`**

**Response Schema**

```json
{
  "object": "list",
  "model": "Qwen3-VL-Embedding-2B",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.4567, ...]
    }
  ]
}
```

------

##  使用示例

1️⃣ 单文本

```json
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": "北京是中国的首都。"
}
```

2️⃣ 多文本

```json
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": [
    "我喜欢机器学习。",
    "Embedding 常用于语义检索。"
  ]
}
```

3️⃣ 单图片（Base64）

```json
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": {
    "image_base64": "<base64>",
    "instruction": "Represent the image",
    "dimensions": 2048,
    "normalize": true
  }
}
```

4️⃣ 多图片

```json
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": [
    {
      "image_base64": "<img1>",
      "dimensions": 1024
    },
    {
      "image_base64": "<img2>",
      "dimensions": 1024
    }
  ]
}
```

5️⃣ 单条多模态（图 + 文）

```json
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": {
    "text": "这张图的主要内容是什么？",
    "image_base64": "<base64>",
    "instruction": "Represent the user's input",
    "dimensions": 2048
  }
}
```

6️⃣ 多条多模态（混合）

```json
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": [
    {
      "text": "纯文本样例",
      "dimensions": 2048
    },
    {
      "image_base64": "<img>",
      "instruction": "Represent the image",
      "dimensions": 2048
    },
    {
      "text": "图文一起",
      "image_base64": "<img>",
      "dimensions": 2048
    }
  ]
}
```

----
## 2. Chat Completion API（对话接口）

**2.1 接口说明**

```http
POST /v1/chat/completions
```

用于多模态对话、图像理解、文本生成。

**2.2 请求参数**

```json
{
  "model": "Qwen3-VL-8B-Instruct",
  "messages": [
    {
      "role": "system",
      "content": "你是一个专业的 AI 助手"
    },
    {
      "role": "user",
      "content": "介绍一下 NVIDIA A100"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**参数说明**

| 字段        | 类型   | 必填 | 说明                        |
| ----------- | ------ | ---- | --------------------------- |
| model       | string | 是   | 固定为 Qwen3-VL-8B-Instruct |
| messages    | array  | 是   | 对话消息列表                |
| temperature | number | 否   | 采样温度                    |
| max_tokens  | number | 否   | 最大生成 token 数           |

图片上传数量限制：1

图片格式：官方没有白名单，目前只测试了.mp4

**2.3 图文\视频对话示例**

```json
{
  "model": "Qwen3-VL-8B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "请描述这张图片/视频" },
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,xxxx"}
        },
        {
         "type": "video_url",
         "video_url": {"url": video_url}
        }
      ]
    }
  ],
  "max_tokens": 512
}
```

**2.4 返回示例**

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "这张图片中可以看到……"
      }
    }
  ]
}
```
    