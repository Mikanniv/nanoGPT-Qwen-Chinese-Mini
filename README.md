# Chinese-Mini-LLM-Pipeline: 从零开始预训练和微调中文GPT

本项目基于 [nanoGPT](https://github.com/karpathy/nanogpt) 架构，构建了一个完整的**中文迷你语言模型 (Mini-LLM)** 训练管线，涵盖了从**预训练 (Pre-Training)** 到**指令微调 (SFT)** 的全流程。

我们使用 **Qwen 的分词器**在中文维基百科数据上进行训练，并针对中文语境优化了超参数和训练策略，旨在为 LLM 实践者提供一个稳定、可复现的入门级项目。

## 核心特性

| 特性 | 脚本/文件 | 描述 |
| :--- | :--- | :--- |
| **GPT 核心实现** | `nanoGPT.py` | 实现了经典的 $\text{GPT}$ 模型架构，包含 **Pre-Normalization Block** 和 **Tied Weights**。 |
| **稳定预训练** | `pretrain.py` | 采用 **Warmup + Cosine Decay** 学习率调度，**AdamW** (**weight_decay=0.1**)，解决了训练初期损失震荡的问题。 |
| **中文支持** | `pretrain.py`, `SFT.py` | 集成 **Qwen/Qwen2-0.5B** 的 **Tokenizer**，词汇表大小为 **151643**。 |
| **指令微调** | `SFT.py` | 基于 `BelleGroup/multiturn_chat_0.8M` 数据集，实现多轮对话指令微调。 |
| **推理部署** | `generate.py` | 提供加载检查点和进行文本生成的简洁示例。 |
| **混合精度** | `pretrain.py`, `SFT.py` | 全面支持 **AMP** (自动混合精度) 训练。 |

## 模型与训练配置

本项目使用的 **Mini-GPT** 模型结构参数如下：

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `block_size` | **256** | 上下文长度 |
| `n_layer` | **12** | Transformer 层数 |
| `n_head` | **12** | 注意力头数 |
| `n_embd` | **768** | 嵌入和隐藏层维度 |
| `vocab_size` | **151643** | 词汇表大小 |
| **Effective Batch Size** | **256** (`8 * 32`) | 预训练阶段的有效批次大小 |
| **初始 LR** | **1e-5** | 预训练阶段的初始学习率 |

## 代码结构
- .
- ├── nanoGPT.py        # GPT模型核心架构
- ├── pretrain.py       # 预训练脚本 (WikiDataset, LR调度)
- ├── SFT.py            # 指令微调脚本 (SFTDataset)
- ├── generate.py               # 模型推理和生成示例
- ├── requirements.txt      # 环境依赖
- └── README.md             # 本文件

## 训练指南

### 1. 环境与依赖

安装项目所需的依赖包：

```bash
# 建议通过环境导出或手动创建 requirements.txt
pip install torch transformers datasets matplotlib
```
### 2. 预训练 (Pre-Training)

使用中文维基百科数据进行语言基础学习。

数据准备: 确保您的中文维基百科数据（例如 wiki_zh_2019）已放置在项目根目录，并按 AA/, AB/ 文件夹结构组织。[wiki数据](https://github.com/brightmart/nlp_chinese_corpus)

启动:

```Bash
python pretrain.py
重要说明： pretrain.py 中已经集成了 Warmup 和 Cosine Decay 调度，训练应该更加稳定。
```

### 3. 指令微调 (SFT)

加载预训练权重，在指令数据集上进行微调，赋予模型指令跟随能力。

准备权重: 在 SFT.py 脚本中，将 resume_path 设置为预训练阶段保存的最新 .pt 文件路径。

启动:

```Bash
python SFT.py
```

### 4. 模型推理

使用 调用.py 脚本验证您的模型效果。

配置路径: 在 调用.py 中，将 path 变量指向您 SFT 阶段保存的最终模型检查点。

运行生成:

```Bash
python 调用.py
示例 Prompt: '数学是'
预期输出: （模型生成的文本，如：数学是研究数量、结构、变化以及空间等概念的一门学科。）
```
