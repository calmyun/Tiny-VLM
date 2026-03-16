# 模型文件下载

本目录存放 Tiny-VLM 所需的预训练模型文件。

## 模型列表

| 模型 | 用途 | ModelScope ID |
|------|------|---------------|
| Qwen2-7B | 语言模型 | qwen/Qwen2-7B |
| InternViT-300M-448px | 视觉编码器 | OpenGVLab/InternViT-300M-448px |


### 方式：手动下载

#### 1. 安装 ModelScope

```bash
pip install modelscope
```

#### 2. 下载 Qwen2-7B

```bash
modelscope download --model qwen/Qwen2-7B --local_dir ./llm/qwen2-7b
```

#### 3. 下载 InternViT-300M-448px

```bash
modelscope download --model OpenGVLab/InternViT-300M-448px --local_dir ./vision_model/internvit-300m
```

### 方式二：使用 Python 代码下载

```python
from modelscope import snapshot_download

# 下载 Qwen2-7B
snapshot_download(
    'qwen/Qwen2-7B',
    cache_dir='./llm/qwen2-7b',
    revision='master'
)

# 下载 InternViT-300M-448px
snapshot_download(
    'OpenGVLab/InternViT-300M-448px',
    cache_dir='./vision_model/internvit-300m',
    revision='master'
)
```