# Tiny-VLM
一个轻量级的多模态预训练模型
=======
# Tiny-VLM: 多模态视觉语言模型

## 项目概述

Tiny-VLM 是一个基于 InternViT-300M 和 Qwen2-7B 的多模态视觉语言模型（Vision Language Model, VLM）。本项目参考 MiniMind-V 的架构设计，采用更强大的视觉编码器和语言模型，实现更优秀的图像理解和对话能力。

### 核心特性

- **强大的视觉编码器**: 使用 InternViT-300M-448px，具备更细粒度的图像特征提取能力
- **优秀的语言模型**: 基于 Qwen2-7B，支持中英文对话，具备强大的语言理解和生成能力
- **简洁的架构设计**: 采用线性投影层实现视觉特征到语言模型的对齐
- **完整的训练流程**: 包含预训练（Pretrain）和监督微调（SFT）全过程

### 模型架构

```
Image → InternViT-300M → Vision Projector → Qwen2-7B → Decoder
```

| 组件 | 模型 | 参数量 | 输出维度 |
|------|------|--------|----------|
| Vision Encoder | InternViT-300M-448px | 300M | 1024 |
| Vision Projector | 2-Layer MLP | ~14M | 3584 |
| Language Model | Qwen2-7B | 7B | 3584 |

## 模型架构详解

### 1. Vision Encoder (InternViT-300M)

InternViT-300M 是 OpenGVLab 开发的视觉编码器，具有以下特点：

- **输入尺寸**: 448×448 像素
- **Patch大小**: 14×14
- **输出特征**: 1024维向量
- **优势**: 相比 CLIP-ViT-Base，InternViT 具有更强的细粒度图像理解能力

### 2. Vision Projector

Vision Projector 负责将视觉特征对齐到语言模型的嵌入空间：

```python
class VisionProjector(nn.Module):
    def __init__(self):
        self.projector = nn.Sequential(
            nn.Linear(1024, 3584),  # InternViT → Qwen2
            nn.GELU(),
            nn.Linear(3584, 3584),
        )
```

### 3. Language Model (Qwen2-7B)

Qwen2-7B 是阿里云开源的大语言模型：

- **隐藏层维度**: 3584
- **注意力头数**: 28
- **层数**: 28
- **词表大小**: 152064
- **支持语言**: 中文、英文

### 4. 多模态融合

图像特征通过 Vision Projector 投影后，替换文本序列中的 `<image>` 占位符，实现多模态融合：

```
输入: "描述这张图片：<image>"
处理: 将 <image> 替换为投影后的图像特征向量
输出: 图像描述文本
```

## 数据集介绍

### 数据来源

本项目使用与 MiniMind-V 相同的数据集格式：

- **预训练数据**: 来自 CC-3M 和 COCO 2014，约 57 万张图像
- **SFT数据**: 来自 llava-en-zh-300k，约 30 万条指令微调数据

### 数据格式

数据集使用 Parquet 格式存储，包含以下字段：

```python
{
    "conversations": [
        {"role": "user", "content": "描述这张图片。<image>"},
        {"role": "assistant", "content": "这是一张..."}
    ],
    "image_bytes": <图像二进制数据>
}
```

### 数据预处理

1. **图像预处理**: 使用 InternViT 的图像处理器，将图像调整为 448×448
2. **文本处理**: 使用 Qwen2 分词器，支持 ChatML 格式的对话模板
3. **标签生成**: 仅计算 assistant 回复部分的损失

### 核心文件说明

| 文件 | 功能 |
|------|------|
| `model/model_tiny_vlm.py` | TinyVLM 主模型，包含 Vision Projector 和多模态融合逻辑 |
| `model/model_vision.py` | InternViT 视觉编码器封装 |
| `dataset/lm_dataset.py` | Parquet 数据集加载和处理 |
| `trainer/train_pretrain_vlm.py` | 预训练脚本，冻结 LLM 主体参数 |
| `trainer/train_sft_vlm.py` | SFT 脚本，全参数微调 |
| `trainer/trainer_utils.py` | 训练工具函数（学习率调度、检查点等） |
| `eval_vlm.py` | 模型推理和评估 |

### 创新点

1. **更强的视觉编码器**: InternViT-300M 具有更细粒度的图像特征提取能力
2. **更强大的语言模型**: Qwen2-7B 提供优秀的语言理解和生成能力
3. **更高的图像分辨率**: 448×448 输入，捕捉更多图像细节
4. **完整的中文支持**: Qwen2 原生支持中英文对话

## 安装步骤

### 1. 克隆项目

```bash
cd /root/autodl-tmp/minimind-v/Tiny-VLM
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 下载预训练模型

```bash
python download_models.py
```

这将从 ModelScope 下载：
- InternViT-300M-448px
- Qwen2-7B

### 4. 准备数据集

将数据集文件放入 `dataset/` 目录：
- `pretrain_i2t.parquet` - 预训练数据
- `sft_i2t.parquet` - SFT 数据

## 运行指南

### 预训练

```bash
# 单卡训练
python trainer/train_pretrain_vlm.py --epochs 4 --batch_size 4 --learning_rate 2e-5

# 多卡训练
torchrun --nproc_per_node N trainer/train_pretrain_vlm.py --epochs 4 --batch_size 4
```

预训练阶段冻结 InternViT 和 Qwen2-7B 主体参数，仅训练 Vision Projector 和 LLM 最后几层。

### 监督微调

```bash
# 单卡训练
python trainer/train_sft_vlm.py --epochs 2 --batch_size 2 --learning_rate 1e-6

# 多卡训练
torchrun --nproc_per_node N trainer/train_sft_vlm.py --epochs 2 --batch_size 2
```

SFT 阶段解冻全部参数进行全参数微调。

### 模型评估

```bash
python eval_vlm.py --weight sft_vlm --image_dir ./dataset/eval_images/
```

### 断点续训

```bash
python trainer/train_pretrain_vlm.py --from_resume 1
```

## 训练参数说明

### 预训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 4 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--learning_rate` | 2e-5 | 初始学习率 |
| `--max_seq_len` | 2048 | 最大序列长度 |
| `--freeze_llm` | 1 | 是否冻结LLM参数 |
| `--from_weight` | None | 基础权重名称 |

### SFT参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 |
| `--learning_rate` | 1e-6 | 初始学习率 |
| `--max_seq_len` | 4096 | 最大序列长度 |
| `--from_weight` | pretrain_vlm | 基础权重名称 |

## License

Apache-2.0 License
