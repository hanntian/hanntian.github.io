---
title: 'Transformer Series (2) - Architecture'
date: 2024-08-19
permalink: /posts/2024/08/blog-post-lm/
excerpt: |
  🐿️ **TL;DR**

  1. Build Transformer Language Model: Embedding, Transformer blocks, Linear
  2. Transformer blocks: RMSNorm, Causal Self Attention,SWiGLU FFN
  3. Transformer Q&A
tags:
  - Transformer Blocks
  - Transformer
toc: true
---

<div class="notice--info" markdown="1">
🐿️ **TL;DR**

1. Build Transformer Language Model: Embedding, Transformer blocks, Linear
2. Transformer blocks: RMSNorm, Causal Self Attention,SWiGLU FFN
3. Transformer Q&A

</div>

# 1 Transformer LM

Through BPE, now we have a sequence of token IDs from input text(i.e., torch.Tensor of shape (batch_size, sequence_length)). 

Next, the Transformer language model uses an input embedding to convert token IDs to dense vectors, passes the embedded tokens through num_layers Transformer blocks, and then applies a learned linear projection (the “output embedding” or “LM head”) to produce the predicted next-token logits (i.e., a PyTorch Tensor of shape (batch_size, sequence_length, vocab_size)).

<figure class="align-center" style="width: 360px">
  <img src="/images/lm-arch.png" alt="Figure 1 An overview of decoder-only Transformer language model">
  <figcaption style="text-align: center;">Figure 1 An overview of decoder-only Transformer language model.</figcaption>
</figure>

## 1.1 Embedding layer

Each token embedding layer(the red block of Figure 1) takes in a tensor of integers of shape
(batch_size, sequence_length) and produces a sequence of vectors of shape <span style="color: #e07b39"> (batch_size,
sequence_length, d_model)</span>. 

## 1.2 Pre-norm Transformer Block

After embedding layer, the activations are processed by num_layers identical layers (commonly
called Transformer “blocks”). Each Transformer block takes in an input of shape (batch_size,
sequence_length, d_model) and returns an output of shape <span style="color: #e07b39"> (batch_size, sequence_length, d_model) </span>.

<figure class="align-center" style="width: 320px">
  <img src="/images/lm-transformer-block.png" alt="Figure 2 A pre-norm Transformer block">
  <figcaption style="text-align: center;">Figure 2 A pre-norm Transformer block</figcaption>
</figure>

Each Transformer block has two sub-layers: a **multi-head self-attention** mechanism and a **position-wise feed-forward** network.

The “pre-norm” blocks requires the use of layer normalization (yellow blocks in Figure 1 as well as Figure 2) after the final Transformer block **to ensure its outputs are properly scaled**.

## 1.3  Linear

After num_layers Transformer blocks, we will take the final activations and turn them into a distribution over the vocabulary, which finally returns<span style="color: #e07b39"> (batch_size, seq_len, V) </span>.

# 2 Building Blocks

## 2.1  Parameter Initialization

Training neural networks effectively often requires careful initialization of the model parameters. Here are some approximate initializations that should work well for most cases. For now, use:

- Linear weights: $\mathcal{N}\left(\mu = 0,\ \sigma^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$truncated at $[-3\sigma, \, 3\sigma]$
- Embedding: $\mathcal{N}\left(\mu = 0,\ \sigma^2 = 1\right)$ truncated at $[-3, \, 3]$
- RMSNorm: 1

Use `torch.nn.init.trunc_normal_` to initialize the truncated normal weight.

## 2.2 Embedding Module

Actually look up embedding matrix using token IDs.

Embedding matrix’s size is ($vocab\_size$, $d_{model}$).

Code: Please see [here](https://github.com/hanntian/assignment1-basics/blob/06add0d164c797ca1e1b7a84e1a1f54cea669fd2/cs336_basics/embedding.py#L3).

In previous Transformer implementations, the input embedding is formed by TokenEmbedding(above) + PositionEmbedding, where the position embedding can be learned or sinusoidal.

Nowadays, many newer LLMs no longer use additive absolute position embeddings at the input layer. Instead, they inject positional information inside the attention mechanism. For example, LLaMA uses RoPE ( refer to [2.3.4](https://www.notion.so/Transformer-Language-Model-Architecture-35f30cd0ce5980c793a2c1df5d8691a7?pvs=21)).

So the main difference is:
Additive position embedding: position information is added once to the input embedding.
RoPE: position information is injected inside the attention mechanism by rotating the Q and K vectors, allowing the attention scores to capture positional relationships.

## 2.3 Pre-Norm Transformer Block

### 2.3.1 Post-Norm vs. Pre-Norm

In the original Transformer paper, the model uses a residual connection around each of the two sublayers, followed by layer normalization. This architecture is commonly known as the “**post-norm**” Transformer, since layer normalization is applied to the sub-layer output.

```python
 y = LayerNorm(x + SubLayer(x))
```

👉 sublayer **+ residual → LayerNorm → this is called post-norm**

```python
y = x + SubLayer(LayerNorm(x))
```

👉  **LayerNorm →**  sublayer **+ residual → this is called pre-norm**

<figure class="align-center" style="width: 520px">
  <img src="/images/lm-prenorm-postnorm.png" alt="Figure 3 Post-norm (left) versus Pre-norm (right)">
  <figcaption style="text-align: center;">Figure 3 Post-norm (left) versus Pre-norm (right)</figcaption>
</figure>

**Why use pre-norm?** A variety of work has found that moving layer normalization from the output of each sub-layer to the input of each sublayer (with an additional layer normalization after the final Transformer block) improves Transformer **training stability**.

An **intuition for pre-norm** is that there is a clean “residual stream” without any normalization going from the input embeddings to the final output of the Transformer, which is purported to improve gradient flow. 

This pre-norm Transformer is now the standard used in language models today (e.g., GPT-3, LLaMA, PaLM, etc.), so we will implement this variant. We will walk through each of the components of a pre-norm Transformer block, implementing them in sequence

### 2.3.2 Root Mean Square Normalization(RMSNorm)

The original Transformer implementation uses layer normalization to normalize activations. 

Here we will use root mean square layer normalization for layer normalization.

Given a vector $\mathbf{a} \in \mathbb{R}^{d_{\text{model}}}$ model of activations, RMSNorm will rescale each activation $a_i$
as follows:

$$
\mathrm{RMSNorm}(a_i)=\frac{a_i}{\mathrm{RMS}(\mathbf{a})}g_i
$$

where $\mathrm{RMSNorm}(a)=\sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} a_i^2 + \varepsilon}$
 . Here, $g_i$ is a **learnable** “gain” parameter (there are $d_model$ such parameters total), and $\varepsilon$ is a hyper-parameter that is often fixed at 1e-5.

Don’t forget to upcast activation to torch.float32 to prevent overflow when we square it.

### 2.3.3  Causal Multi-Head Self-Attention

#### 2.3.3.1 Scaled dot Attention

First, the attention operation mathematically is defined as follows:

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

where $Q=XW_Q, K = XW_K, V=XW_V$.$Q \in \mathbb{R}^{n \times d_{\text{k}}}$, $K \in \mathbb{R}^{m \times d_{\text{k}}}$, $K \in \mathbb{R}^{m \times d_{\text{k}}}$.

In self attention (Encoder-only model), $n = m$. In cross attention(Encoder-Decoder model, Query from Decoder, Key/Value from Encoder), $n \neq m$.

#### 2.3.3.2 Multi-head Self Attention

Then, multi-head self attention is defined as follows:

$$
\begin{aligned}
\mathrm{MultiHead}(Q,K,V)
&=
\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O \\
\mathrm{head}_i
&=
\mathrm{Attention}(Q_i,K_i,V_i) \\
\mathrm{MultiHeadSelfAttention}(x)
&=
W_O\,
\mathrm{MultiHead}
\left(
W_Q x,\,
W_K x,\,
W_V x
\right)
\end{aligned}
$$

with $Q_i, K_i, V_i$  being slice number $i \in \{1,\ldots,h\}$ of size $d_k$ or  $d_v$ of the embedding dimension for $Q,K,$ and $V$respectively. The learnable parameters are $W_Q \in \mathbb{R}^{h d_k \times d_{\mathrm{model}}}$, $W_K \in \mathbb{R}^{h d_k \times d_{\mathrm{model}}}$, $W_V \in \mathbb{R}^{h d_v \times d_{\mathrm{model}}}$,$W_O \in \mathbb{R}^{d_{\mathrm{model}} \times h d_v}$.

**We can see**, instead of maintaining separate projection matrices $W_i^Q, W_i^K,$ and $W_i^V$, for each attention head $i$, modern Transformer implementations use a single large projection matrix $W^Q, W^K,$ and $W^V$ to compute all heads simultaneously. For example: $Q = W_Qx$, we can get shape like $(hd_k​,d_{model}​)\times(d_{model}​,N)→(hd_k​,N)$ which can be reshaped $(h,N,d_k​)$.

```python
batch_size, seq_len, _ = x.shape
Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # view在不改变内存中数据的情况下，重新调整张量的形状（Reshape): d_model -> (num_heads, d_k)
K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # view 的特性：它绝对不会复制内存。它只是创建了一个新的张量头部（Tensor Header），改变了对底层同一块连续内存的“解读方式”。如果内存不连续（比如刚做完 transpose），view 会直接报错。O(1)开销。
V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2) # 而reshape呢：如果内存连续，reshape 和 view 的行为是一样的；如果内存不连续，reshape 会在后台默默地调用 .clone().view(...)，帮你复制一份内存，让它变连续。是情况而定的，可能是 O(1) 也可能是 O(n) 的开销。
```

#### 2.3.3.3 Causal Masking

Finally, let’s see another important part while implementing MHA, that’s masking.

Causal masking means preventing the model from attending to future tokens in the sequence：

We use lower triangular mask matrices for computation by taking the pre-softmax values ( $𝑄𝐾^⊤/√𝑑_𝑘$) and adding a −∞ to any entry of the mask matrix that is False.

```python
 scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # compute pre-softmax
 causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        ).view(1, 1, seq_len, seq_len)
 scores = scores.masked_fill(~causal_mask, float('-inf')) 
 attn_weights = softmax(scores, dim=-1)
 output = torch.matmul(attn_weights, V)
```

### 2.3.4 Rotary Position Embeddings

To inject positional information into the model, we will implement Rotary Position Embeddings often called RoPE.

**RoPE should be applied to the query and key vectors, but not the value vectors**. Also, the head dimension should be handled as a batch dimension, because in multi-head attention, attention is being applied independently for each head. This means that precisely the same RoPE rotation should be applied to the query and key vectors **for each head**.

Let's take an example to show how RoPE calculates.

假设 $q^{(i)}$ 是位置 $i$ 处某个注意力头的 Query 向量，其维度为 $d$。我们将其展开为：

$$
[q_1,q_2,q_3,q_4...q_d]^T
$$

我们可以构造一个 $d \times d$ 的块对角矩阵 $𝑅_𝑖$ ，将整个  $d$维空间切分成 $\frac{d}{2}$ 个二维子空间，每个子空间独立进行旋转：

$$
R_i = \begin{pmatrix} R_1^i &  &  &  \\  & R_2^i &  &  \\  &  & \ddots &  \\  &  &  & R_{\frac{d}{2}}^i \end{pmatrix}
$$

其中，每个对角分块 $R_k^i \in \mathbb{R}^{2 \times 2}$ （ $k \in \{1, \dots, \frac{d}{2}\}$）定义为：

$$
R_k^i = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}
$$

这里的旋转角度 $\theta_{i,k}$ 取决于当前 token 的绝对位置 $i$ 以及对应的维度序号 $k$:  $\theta_{i,k} = \frac{i}{10000^{\frac{2k}{d}}}$。

拿pair $[q_1,q_2]$举例：

$$
[q_1^{'}, q_2^{'}] = [q_1cos\theta -q_2sin\theta, q_1sin\theta +q_2cos\theta]
$$

同样，对 $k^{(j)}$也是同样的计算。

那为什么这里能体现相对位置呢？我们明明注入的是绝对位置。

推理如下，也没有什么值钱的，就是三角函数公式。

> 
> 
> 
> ### 1. 写出 $q'$ 和 $k'$的旋转结果
> {: .no_toc}
> 
> 对于某一维度的二维子向量：
> 
> - **位置 $i$ 的 Query**：
>     
>     $$
>     q' = [q_1 \cos(i\theta) - q_2 \sin(i\theta), \quad q_1 \sin(i\theta) + q_2 \cos(i\theta)]
>     $$
>     
> - **位置 $j$的 Key**（同样进行 RoPE 旋转，只是把位置 $i$ 换成 $j$）：
>     
>     $$
>     k' = [k_1 \cos(j\theta) - k_2 \sin(j\theta), \quad k_1 \sin(j\theta) + k_2 \cos(j\theta)]
>     $$
>     
> 
> ### 2. 计算内积 $q' \cdot k'$
> {: .no_toc}
> 
> 向量内积就是对应位置相乘再相加：
> 
> $$
> \begin{aligned} q' \cdot k' = \;& \big(q_1 \cos(i\theta) - q_2 \sin(i\theta)\big)\big(k_1 \cos(j\theta) - k_2 \sin(j\theta)\big) \\ + \;& \big(q_1 \sin(i\theta) + q_2 \cos(i\theta)\big)\big(k_1 \sin(j\theta) + k_2 \cos(j\theta)\big) \end{aligned}
> $$
> 
> ### 3. 暴力展开并合并同类项
> {: .no_toc}
> 
> 我们把上面这个式子全部乘开：
> 
> - **第一部分乘开：**
>     
>     $$
>     q_1 k_1 \cos(i\theta)\cos(j\theta) - q_1 k_2 \cos(i\theta)\sin(j\theta) - q_2 k_1 \sin(i\theta)\cos(j\theta) + q_2 k_2 \sin(i\theta)\sin(j\theta)
>     $$
>     
> - **第二部分乘开：**
>     
>     $$
>     q_1 k_1 \sin(i\theta)\sin(j\theta) + q_1 k_2 \sin(i\theta)\cos(j\theta) + q_2 k_1 \cos(i\theta)\sin(j\theta) + q_2 k_2 \cos(i\theta)\cos(j\theta)
>     $$
>     
> 
> 现在，我们将带有 $q_1 k_1, q_2 k_2, q_1 k_2, q_2 k_1$ 的项分别归类提取：
> 
> $$
> \begin{aligned} q' \cdot k' = \;& q_1 k_1 \Big( \cos(i\theta)\cos(j\theta) + \sin(i\theta)\sin(j\theta) \Big) \\ + \;& q_2 k_2 \Big( \cos(i\theta)\cos(j\theta) + \sin(i\theta)\sin(j\theta) \Big) \\ + \;& q_1 k_2 \Big( \sin(i\theta)\cos(j\theta) - \cos(i\theta)\sin(j\theta) \Big) \\ + \;& q_2 k_1 \Big( \cos(i\theta)\sin(j\theta) - \sin(i\theta)\cos(j\theta) \Big) \end{aligned}
> $$
> 
> ### 4. 见证奇迹：套用三角函数公式
> {: .no_toc}
> 
> - $\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$
> - $\sin(\alpha - \beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta$
> 
> 把 $\alpha = i\theta, \beta = j\theta$ 代入上面的式子，化简得到最终结果：
> 
> $$
> q' \cdot k' = (q_1 k_1 + q_2 k_2) \cos\big((i-j)\theta\big) + (q_1 k_2 - q_2 k_1) \sin\big((i-j)\theta\big)
> $$
> 
> ### 结论
> {: .no_toc}
> 
> 你看，最后化简出来的式子里面，所有的 $i$ 和 $j$都变成了 $(i-j)$。
> 
> 这意味着，虽然 $q$和 $k$ 在计算时只知道自己的绝对位置（ $i$ or $j$），但当它们做点乘（计算 Attention Score）时，三角函数公式自动帮它们把绝对位置做差，转变成了**相对位置 $i-j$**！
> 

Code: Please see [here](https://github.com/during-gt/assignment1-basics/blob/8711f432ae5ce04d5b3e97789c0af2d620edc1d3/cs336_basics/RoPE.py#L10).

### 2.3.5 Position-Wise Feed-Forward Network

FFN in original Transformer paper is like:

<figure class="align-center" style="width: 600px">
  <img src="/images/lm-orig-ffn.png" alt="Figure 4 FFN in the original Transformer">
  <figcaption style="text-align: center;">Figure 4 FFN in the original Transformer</figcaption>
</figure>

We can see, the inner-layer expands the input dimension by 4×, allowing the model to learn richer nonlinear transformations. It then projects the features back to the original dimension so that residual connections can be applied.

Modern language models tend to incorporate two main changes compared to this [original
design](https://arxiv.org/pdf/1706.03762): use another activation function and employ a gating mechanism. 

Here we use SwiGLU = SiLU(Swish activation function) + GLU(Gated Linear Unit).

> 
> 
> 
> ## 从 SwiGLU 的提出看深度学习中的结构创新
> {: .no_toc}
> 
> 从 SwiGLU 的提出过程可以看出，很多深度学习方法并不是完全凭空发明出来的，而是从已有结构出发，不断提出变体，再通过实验验证效果。SwiGLU也是目前开源 SOTA 模型的趋势。
> 
> Transformer常见的激活函数包括 ReLU 和 GELU。
> 
> ```python
> x
> │
> Linear (d → 4d)
> │
> GELU
> │
> Linear (4d → d)
> │
> output
> ```
> 
> 研究者在此基础上会进一步思考：激活函数能不能更平滑、更连续，并且具有更强的表达能力？
> 
> 于是出现了 SiLU，也叫 Swish。它的直觉是保留 ReLU 的非线性能力，同时让梯度变化更加平滑，从而对模型训练更加友好。
> 
> $SiLU(x) = x\cdot \sigma(x)=\frac{x}{1+e^{-x}}$
> 
> 接着，研究者又引入了门控机制，也就是 GLU 的思想。GLU 最早来自语言模型，其形式为：
> 
> $\text{GLU}(x,W_1,W_2)=\sigma(W_1x))\odot W_2x$
> 
> 它的核心思想是将输入分成两部分：一部分表示内容，另一部分作为门，用来控制有多少信息可以通过。
> 
> SwiGLU 可以理解为 Swish 和 GLU 的组合。它把原本 GLU 中的 sigmoid gate 换成了表达能力更强的 Swish，从而让整个前馈网络结构更加 expressive。
> 
> $\text{SwiGLU}(x, W_1, W_2, W_3)= W_2 \left( SiLU(W_1x) \odot W_3 x \right)$
> 
> $x \in \mathbb{R}^{d_{\text{model}}}$, $W_1,W_3 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$, $W_2\in \mathbb{R}^{d_{\text{model}}\times d_{ff}}$, in LLaMA $d_{ff} = \frac {8}{3} d_{model}$。
> 
> ```python
>                  ┌─ Linear_1 (d → h) ─ SiLU ─┐
> x ───────────────┤                            ×
>                  └─ Linear_3 (d → h) ────────┘
>                               │
>                        Linear_2 (h → d)
>                               │
>                            output
> ```
> 
> 因此，SwiGLU 的效果更好并不是因为随机尝试，而是来自一个典型的深度学习研发流程：
> 
> 首先根据经验提出直觉，例如平滑激活是否更好，门控结构是否更强，非线性组合是否能带来更强的表达能力。 然后基于已有结构设计变体，例如从 ReLU 到 GELU，再到 SiLU；从普通 FFN 到 GLU，再到 SwiGLU。接着在大规模任务上进行实验验证，比如比较 loss、perplexity 和收敛速度。最后，再尝试对实验结果进行解释。
> 

## 2.4  Linear Module

Note that we do not include a bias term, following most modern LLMs.

Linear tranformation: $y = W x$

Code: Please see [here](https://github.com/hanntian/assignment1-basics/blob/93b569088171c90235ff9891bd470af677ae501b/cs336_basics/linear.py#L9).

# 3 Q&A

## 3.1 Question: What does ‘B’ stand for?

业界常说的 **7B** (7 Billion, 70亿)、**14B** 等数字，指的是大模型的大约**总参数量 (Total Parameters)**。

它是一个“全家桶”总和，包含以下所有部分：

- **词汇嵌入层 (Embedding)**
- **所有隐藏层 (Attention + FFN)**
- **输出头 (LM Head)**

具体的参数计算参考[3.2](https://www.notion.so/Transformer-Language-Model-Architecture-35f30cd0ce5980c793a2c1df5d8691a7?pvs=21) 

### 3.1.1 Example: Mistral **8x7B**

在 MoE 架构中，总参数量 $\neq 8 \times 7\text{B}$。如果只走一条最基础的单路径（共享参数 + 仅激活 1 个专家），其规模恰好等于原本的 Mistral 7B 单体模型。

 **参数拆解应该是这样的：**
• **共享参数**（Attention、Embedding 等）：约 **2B**
• **单个专家的 FFN 参数**：约 **5.6B** 
把准确的数字代入公式，一切就对得上了：
• **总参数量**： $2B + 8 \times 5.6B = 2B + 44.8B = 46.8B \approx \textbf{47B}$
• **单次激活参数量（Top-2）**： $2B + 2 \times 5.6B = 2B + 11.2B = 13.2B \approx \textbf{13B}$

### 3.1.2 Version

|  | Base Model  | Instruct Model | Chat Model |
| --- | --- | --- | --- |
| Core Behavior | Standard next-token prediction | Understands and executes specific instructions | Optimized for multi-turn conversations and safe persona alignment |
| Training Stage | Unsupervised pre-training on massive text corpora. | Fine-tuned via **Instruction Tuning** | Fine-tuned via SFT followed by **RLHF** |
| Example | LLaMA 7B | LLaMA 7B Instruct | LLaMA 7B Chat |

## 3.2 Question: The parameter estimation of Transformer LM

The total parameters of Transformer LM means trainable parameters updated via back propagation. Estimating the parameter count means calculating the size of each matrix.

$$
P_{total} \approx P_{embedding} + L \times P_{transformer\_block} + P_{lm\_head}
$$

Take LLaMA as an example:

**Attention Matrix** ： $W_q, W_k, W_v, W_o$  ，the parameters here are  $4 \times d_{model} \times d_{model}$

**FN Matrix**： $W_g, W_u, W_d$ ，the parameters here are $3 \times d_{model} \times d_{ffn} \approx 8 d_{model}^2$

So, the total parameters are

$$
P_{embedding} = P_{lm\_head} = V \times d_{model} \\
P_{transformer\_block} = L(4d_{model}^2 + 3d_{model}d_{ff})

$$

$$
P_{total} \approx 2Vd_{model} + L(4d_{model}^2 + 3d_{model}d_{ff})
$$

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ffn: int, num_layers: int):
        super().__init__()
        
        # --- 1. GLOBAL PARAMETERS ---
        # 1. Embedding matrix (vocab_size -> dense vectors)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 5. Final lm_head (dense vectors -> vocab_size logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # --- LAYER PARAMETERS (Repeated `num_layers` times) ---
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ffn) for _ in range(num_layers)
        ])

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        
        # 2. RMSNorm: Scale factor gain (Pre-Attention & Pre-FFN)
        self.attn_norm_gain = nn.Parameter(torch.ones(d_model))
        self.ffn_norm_gain = nn.Parameter(torch.ones(d_model))
        
        # 3. Attention: W_Q, W_K, W_V, W_O (Bias is typically False in modern LLMs)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # 4. FFN: Three gating linear matrices (SwiGLU architecture)
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)  # Up-projection 1
        self.w_up   = nn.Linear(d_model, d_ffn, bias=False)  # Up-projection 2
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)  # Down-projection
```

## 3.3 Layer Norm vs Batch Norm

**LayerNorm**：按Token切。针对单个样本的**每个 Token 的 $d_{model}$向量**独立计算均值方差，彻底杜绝了句子间和位置间的噪声污染，完美适配变长文本。

> **历史上的 LayerNorm (2016)**：最初为全连接层（MLP）设计，指将**单个样本在某一层（Layer）的所有神经元激活值**拉出来统一算均值和方差。**Transformer 中的 LayerNorm (2017+)**：大模型的输入三维张量为 `[B, T, C]`（ $Batch, Seq\_len, d_{model}$）。为了防止文本前后语义污染，业界将归一化维度锚定在最后一个维度 $d_{model}$。**本质**：它并没有对“整层”或“整句”做 Norm，而是**盯着每一个独立的 Token，对其自身的 $d_{model}$ 维度单独进行归一化**。因此，在 LLM 语境下，它最本质、最贴切的名字应该叫 **Token Norm**（或 Token-wise Norm。
> 

**BatchNorm**：按**列**切。针对整个 Batch 内部**所有 Token 的某一个具体 Channel 维度**跨样本、跨时间步计算均值方差，强依赖 Batch 大小，适合物理通道含义明确的图像数据。