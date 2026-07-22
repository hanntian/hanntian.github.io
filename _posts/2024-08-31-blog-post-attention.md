---
title: 'Transformer Series (3) - Attention & Alternatives'
date: 2024-08-31
permalink: /posts/2024/08/blog-post-attention/
tags:
  - Attention
  - Transformer
  - LLM
excerpt: |
  🐿️ **TL;DR**

  1. Attention Complexity
  2. Way to Speedup Attention 
toc: true
---

<div class="notice--info" markdown="1">
🐿️ **TL;DR**

1. Attention Complexity
2. Way to Speedup Attention 

</div>



# 1. Complexity of Multi Head Attention

假设输入序列的长度（通常是pad后的）为 $N$, 每个 Token 的Embedding Dimension 为 $d_{model}$。

Batch 维度独立并行,以下公式省略。

首先，生成 $Q,K,V$矩阵。

将输入矩阵$X \in \mathbb{R}^{N \times d_{model}}$ 分别乘以三个权重矩阵。$\quad W_Q, W_K \in \mathbb{R}^{  d_{model}\times hd_k}, W_V\in \mathbb{R}^{ d_{model}\times hd_v }$

$$
Q = XW_Q,\quad K =  X W_K,\quad V =XW_V
$$

- 复杂度: $2(N⋅d_{model}⋅hd_k)+(N⋅d_{model}⋅hd_v)=O(N⋅d_{model}^2)$

接着，计算注意力分数($QK^T$)。

矩阵 $Q \in \mathbb{R}^{N \times hd_k}$ 与 $\quad K^\top \in \mathbb{R}^{hd_k \times N}$ 相乘，得到一个 $Q K^\top \in \mathbb{R}^{N \times N}$ 的相似度矩阵。

- 复杂度: $O(N^2 hd_k)$

然后，Softmax 与权重聚合($Softmax(..)V$)。

将$N \times N$ 的分数矩阵与$V \in \mathbb{R}^{N \times hd_v}$  相乘，得到最终输出。

- 复杂度: $O(N^2 hd_v)$

最后输出投影，$𝑊_𝑂Attention(Q,K,V)$。其中  $W_O \in \mathbb{R}^{d_{model} \times hd_v}$。

- 复杂度: $O(N⋅hd_v⋅d_{model})$

所以，

短上下文时，Attention的计算瓶颈为 $O(N⋅d_{model}^2)$项。

长上下文时( $N >>d_{model}$)时， $O(N^2⋅d_{model})$为主导项。

# 2. Attention Alternatives

下面我通过判断输出结果是不是和标准 attention 完全一样，得到如下分类。

<style>
.attn-table { border-collapse: collapse; width: 100%; font-size: 0.9em; line-height: 1.5; }
.attn-table th, .attn-table td { border: 1px solid #d0d0d0; padding: 8px 10px; vertical-align: top; text-align: left; }
.attn-table th { background: #f2f2f2; font-weight: 600; }
</style>

<table class="attn-table">
<thead>
<tr>
<th>Attention Optimization</th>
<th>Representatives</th>
<th>Core Mechanism</th>
<th>Cost</th>
</tr>
</thead>
<tbody>
<tr style="background:#e6f4ea;">
<td rowspan="2"><strong>近似优化</strong><br>(牺牲精度换复杂度)</td>
<td>Linear Attention</td>
<td>$(QK^T)V \rightarrow Q(K^TV)$，复杂度从 $O(N^2 d)$ 降到 $O(Nd^2)$</td>
<td>损失部分表达能力</td>
</tr>
<tr style="background:#e6f4ea;">
<td>Sparse Attention</td>
<td>仅计算部分关键 Token 之间的权重，将密集计算转化为<strong>稀疏计算</strong>，降低计算量</td>
<td></td>
</tr>
<tr style="background:#fdecea;">
<td><strong>精确优化</strong><br>(结果完全等价)</td>
<td>Flash Attention (v1/2/3)</td>
<td>把 $O(N^2)$ 的中间矩阵不实例化到 HBM，只在 SRAM 里算。降低 memory IO，<strong>训练和推理都用，结果完全等价于标准 attention</strong>。</td>
<td>不存 forward 的中间结果 → backward 需要 <strong>recompute</strong>；需要更复杂的 kernel</td>
</tr>
<tr style="background:#fce8f0;">
<td rowspan="2"><strong>架构替代</strong><br>(干脆不用 attention)</td>
<td>Mamba</td>
<td>基于状态空间模型（State Space Models），通过时变选择性机制实现<strong>具备线性复杂度的长文本上下文记忆</strong>。</td>
<td></td>
</tr>
<tr style="background:#fce8f0;">
<td>RWKV、Gated DeltaNet</td>
<td>线性注意力的 RNN 递归形式，维持恒定隐状态 $S_t$。训练可并行，推理 O(1) <strong>隐状态</strong>更新。</td>
<td></td>
</tr>
<tr style="background:#e3f0fc;">
<td rowspan="3"><strong>推理专属优化</strong></td>
<td><strong>KV Cache</strong>：MQA、GQA、MLA(DeepSeek)</td>
<td>减少推理时 Key 和 Value 的显存占用。MQA：全 head 共享 1 组 KV；GQA：$g$ 组共享；MLA：低秩联合压缩，仅缓存潜向量 $c_t = W^{DKV}x_t$，而且<strong>不用把潜向量显化回完整的 K、V</strong>，靠的是<strong>矩阵吸收(absorption)</strong>。<strong>KV cache 里只存一份 $c_s^{KV}$（加一小段共享的 RoPE key）</strong>。</td>
<td>MQA 质量略降</td>
</tr>
<tr style="background:#e3f0fc;">
<td><strong>PagedAttention</strong> (vLLM)</td>
<td>借鉴操作系统虚拟内存思想，将 KV Cache 离散存储在不连续的物理内存页中。显存碎片 ≈ 0，显存利用率↑ → batch↑、吞吐↑</td>
<td>需自定义 paged kernel</td>
</tr>
<tr style="background:#e3f0fc;">
<td><strong>量化</strong>(KV cache 量化、权重量化)</td>
<td>$\hat w = s\cdot\mathrm{round}(w/s)$，FP16/BF16 → INT8/INT4</td>
<td>低 bit 有精度损失，需校准 / 异常值处理</td>
</tr>
</tbody>
</table>

这四类的"省"的对象其实不同:近似优化和架构替代省的是**计算复杂度**( $N^2\!\to\!N$), Flash 省的是

**显存 IO** (FLOPs 没变), 推理专属那一组省的是**推理时 KV 显存**。

## 2.1 Linear Attention

### 2.1.1.Associative Law of Matrix Multiplication

Consider the usual attention operation:

$$
Atten(Q,K,V) = 𝜌 (QK^T)V
$$

This is quadratic (see 1). Can we do better (when 𝜌 is the identity)? 

$$
(QK^T)V = Q(K^TV)
$$

Very simple, and we get from $N^2hd_k +N^2hd_v$  to $2Nhd_khd_v$ which from quadratic to linear.

That’s linear attention.

### 2.2.2 Recurrent form  of Linear Attention

Let’s set  $S =  K^TV$, with $K^T \in \mathbb{R}^{d\times N}$, $V \in \mathbb{R}^{N \times d}$. then $S \in \mathbb{R}^{d \times d}$. 

Then, expand $K^TV$,

$$
K = \begin{bmatrix}
k_1^\top \\
k_2^\top \\
\vdots \\
k_N^\top
\end{bmatrix}
$$

$$
V = \begin{bmatrix}
v_1^\top \\
v_2^\top \\
\vdots \\
v_N^\top
\end{bmatrix}
$$

$$
K^\top V = \sum_{i=1}^N k_i v_i^\top
$$

Please note $k_i \in \mathbb{R}^d$, $v_i^\top \in \mathbb{R}^{1 \times d}$. So $k_i v_i^\top$ is a $d \times d$ outer product. So, $S_N = \sum_{i=1}^N k_i v_i^\top$.

Finally, we can write the above result into recurrent form

$$
S_t = \sum_{i=1}^t k_i v_i^\top
$$

Apparently, 

$$
S_t = S_{t-1} + k_t v_t^\top
$$

For the t-th token, the attention output can be computed as:

$$
y_t=q_t^⊤S_t
$$

为什么快？逐步更新一个“压缩后的 KV 状态”（S），每个 token 用自己的 q 去读取这个状态

为什么表达力变弱？把“每个 query 自己选 token” → 变成“所有 token 先混在一起”

### 2.2.3 General Format of Linear Attention

In 2.2.1, we suppose 𝜌 is the identity. In general, 𝜌 is a feature map $\phi(\cdot)$ that can make

$$
\exp(q^{\top}k) \approx \phi(q)^{\top}\phi(k)
$$

Then,

$$
\boldsymbol{A}_{t} = \frac{\phi(\boldsymbol{q}_t)^{\top}\phi(\boldsymbol{k}_j)}{\sum_l \phi(\boldsymbol{q}_t)^{\top}\phi(\boldsymbol{k}_l)}
$$

$$
\boldsymbol{y}_t = \frac{\sum_j \phi(\boldsymbol{q}_t)^{\top}\phi(\boldsymbol{k}_j)\boldsymbol{v}_j}{\sum_j \phi(\boldsymbol{q}_t)^{\top}\phi(\boldsymbol{k}_j)}
$$

Then, using the associative law:

$$
\boldsymbol{y}_t = \frac{\phi(\boldsymbol{q}_t)^{\top} \left( \sum_j \phi(\boldsymbol{k}_j)\boldsymbol{v}_j^{\top} \right)}{\phi(\boldsymbol{q}_t)^{\top} \left( \sum_j \phi(\boldsymbol{k}_j) \right)}
$$

At this point, two recurrent states emerge:

$$
\boldsymbol{S}_t = \sum_{j=1}^t \phi(\boldsymbol{k}_j)\boldsymbol{v}_j^{\top}
$$

$$
\boldsymbol{Z}_t = \sum_{j=1}^t \phi(\boldsymbol{k}_j)
$$

Therefore:

$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \phi(\boldsymbol{k}_t)\boldsymbol{v}_t^{\top}
$$

$$
\boldsymbol{Z}_t = \boldsymbol{Z}_{t-1} + \phi(\boldsymbol{k}_t)
$$

$$
\boldsymbol{y}_t = \frac{\phi(\boldsymbol{q}_t)^{\top} \boldsymbol{S}_t}{\phi(\boldsymbol{q}_t)^{\top} \boldsymbol{Z}_t}
$$

### 2.2.4 Example: Lightning Attention

Let's take a look at lightning attention, which ever used in MiniMax-M1 allowing M1 to natively support an ultra-long context of up to 1 million tokens, and drastically reduced inference compute (FLOPs). 

I highlighted the key details of the **forward pass** here since the **tiling** strategy here is ingenious.

It splits the input along the sequence dimension into multiple small tiles (each with a length of `BLOCK`, such as 64).

**Intra-block parallel:** Once a full $64 \times 64 QK^T$  tile is loaded into SRAM, all tokens within this block execute matrix multiplication simultaneously and in parallel. This fully squeezes out  the parallel computing power of the GPU Tensor Cores.

**Inter-block recurrent:**Since there is a causal temporal relationship between blocks, it adopts a recurrent approach when transitioning from one block to the next: after computing the first tile, the extracted historical state (hidden $KV$ state) is passed to the second tile; the second tile then incorporates this state to complete its computation, updates it, and passes it on to the third.

当前的 $Q$只需要直接和这个 $KV$做一次乘法，就顺理成章地吸收了“过去所有的历史信息”。

```python
# compute
qk = tl.dot(q, k_trans) * diag_decay
o_intra = tl.dot(qk, v) # in block standard attention
o_inter = tl.dot(q, kv) * q_decay # inter block linear attention
o = o_intra + o_inter
kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
```

### 2.2.5 Mamba-2

Let’s generalize linear attention a little bit and add per-position weights.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image.png" alt="">
</figure>

There is a lot more words to justify this (go read the mamba 2 paper) but the mechanics is that we can make linear attention **more expressive via gating** (gating is good!

### 2.2.6 Gated delta net (and friends)

Let’s generalize things further – gate the input and selectively erase the state.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%201.png" alt="Figure 1 Gated Delta Net (GDN) update Equation">
  <figcaption style="text-align: center;">Figure 1 Gated Delta Net (GDN) update Equation</figcaption>
</figure>

$k_tk_t^T$是一个秩 1 的投影矩阵( $k_t$已归一化)，作用是把信息投影到当前 $key$的方向。那么 $1-\beta_t k_tk_t^T$意思就是把旧memory中“沿着当前 $key$方向”的一部分按强度 $\beta_t$抹掉。 $\beta_t$=1 时完全擦除,$\beta_t$=0 时不动。

抹掉之后再写入新的 $\beta_t k_tv_t^\top$,合起来就是上面公式: **在当前 key 上,先删掉旧 value、再写入新 value。**

因此, $\gamma$和 $\beta$ 一个控制的是整体memory强度，一个控制的某个语义方向。

## 2.2 Sparse Attention

Instead of attending to every token, do sparse attention (DSA).

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%202.png" alt="Figure 2 Sparse Attention Process">
  <figcaption style="text-align: center;">Figure 2 Sparse Attention Process</figcaption>
</figure>

## 2.3 Flash Attention

Flash Attention属于精确优化，意思是不改变原有Attention的计算结果。而且随插即用，可以直接套用在任何使用Attention的模型上。

背景: GPU 在工作时，数据是先从**片外的 HBM** 大批大批地搬运到片内的 SRAM（缓存）中，再由 GPU 核心直接从 SRAM 中读取进行计算。

| **Feature** | **SRAM (Cache)** | **HBM (VRAM)** |
| --- | --- | --- |
| **Physical Location** | **On-Die** (Inside the GPU core) | **Off-Die** (Outside the GPU core, within the same package) |
| **Capacity Level** | MB level (Extremely precious) | GB level  |
| **Main Task** | Stores immediate data currently being computed | Stores the entire AI model, weights, and large datasets |
| **Latency (Speed)** | Extremely low (Available to the compute cores instantly) | Relatively higher (Data needs to travel across the interposer) |
| **Bandwidth (Throughput)** | Extremely high | Extremely high (Parallel transmission via thousands of pins) |

### 2.3.1 How to calculate standard attention in GPU?

分块将HBM中的值搬到SRAM中, 依次求 $a_i$,  $a_{max}$,分子 $a_i':e^{a_i -a_{max}}$, 分母 $\sum_{i=1}^{L}e^{a_i -a_{max}}$，分子分母相除得到 attention weight: $\hat a_i$, 最后 $\hat a_i * v_i$输出 $o$。

以单头query和key为例, HBM为仓库(灰色), SRAM为工作台(绿色)。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%203.png" alt="Step1: 计算a。其中N取决于不同GPU SRAM大小">
  <figcaption style="text-align: center;">Step1: 计算a。其中N取决于不同GPU SRAM大小</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%204.png" alt="从a算出attention score a^, 实际中我们需要先算出所有a_i中的最大值a_max,否则会overflow">
  <figcaption style="text-align: center;">从a算出attention score a^, 实际中我们需要先算出所有a_i中的最大值a_max,否则会overflow</figcaption>
</figure>

那如何算出$a_{max}$呢？

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%205.png" alt="把所有的a_i搬到SRAM快速算出a_max是不可行的">
  <figcaption style="text-align: center;">把所有的a_i搬到SRAM快速算出a_max是不可行的</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%206.png" alt="Step2: 计算a_max。把L/N个chunks依次找局部最大,最后的d_B就是全局最大的,也就是我们需要的a_max">
  <figcaption style="text-align: center;">Step2: 计算a_max。把L/N个chunks依次找局部最大,最后的d_B就是全局最大的,也就是我们需要的a_max</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%207.png" alt="Step3: 计算分子a_i’ 和分母sum(a_i’)。">
  <figcaption style="text-align: center;">Step3: 计算分子a_i’ 和分母sum(a_i’)。</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%208.png" alt="Step4: 分子a_i’和分母相除最终得到了attention score a^。">
  <figcaption style="text-align: center;">Step4: 分子a_i’和分母相除最终得到了attention score a^。</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%209.png" alt="Step5: 最后attention score a^和v乘积就得到了某个位置的output了">
  <figcaption style="text-align: center;">Step5: 最后attention score a^和v乘积就得到了某个位置的output了</figcaption>
</figure>

### 2.3.2 Flash Attention

我们可以从[图中](https://app.notion.com/p/Attention-Alternatives-37130cd0ce59802ba59ec3b0398a66f1?pvs=21)看出 $a$到 $\hat a$,需要多次搬运chunks从HBM到SRAM。那真的需要这么多次吗？！

Flash attention 的做法是不需要真的计算出attention weight $\hat a$ ，直接一步到位从 $a$到 $o$, 从而减少了搬运次数。具体做法就是分块从 $a_i$直接算出 $o$, 当前错误先将错就错，下一个chunk再修补。下面将展示具体如何修补错误。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2010.png" alt="Chunk1: 截至当前chunk的最大值存到d里(目前是d1),然后在当前chunk算分子，分母，相除,乘以v得到截止目前chunk的输出o1。但我们知道，这是不对的。目前的d不一定是全局最大的d，输出也不是最终的o。">
  <figcaption style="text-align: center;">Chunk1: 截至当前chunk的最大值存到d里(目前是d1),然后在当前chunk算分子，分母，相除,乘以v得到截止目前chunk的输出o1。但我们知道，这是不对的。目前的d不一定是全局最大的d，输出也不是最终的o。</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2011.png" alt="Chunk2: 修补错误。截至当前chunk的最大值存到d里(目前是d2),然后在当前chunk算分子，修补分母，相除,乘以v，修补输出o2。">
  <figcaption style="text-align: center;">Chunk2: 修补错误。截至当前chunk的最大值存到d里(目前是d2),然后在当前chunk算分子，修补分母，相除,乘以v，修补输出o2。</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2012.png" alt="Chunk B: 以此类推。">
  <figcaption style="text-align: center;">Chunk B: 以此类推。</figcaption>
</figure>

## 2.4 KV cache

LLM 推理（Inference）通常分为两个阶段：

- **Prefill（Prompt Processing）**：将用户输入的全部 Prompt token 一次性并行送入模型计算。此阶段除了预测出第一个输出 token 外，更关键的是计算并缓存所有输入 token 的 KV Cache。
    - 由于一次性处理大量 token，Prefill 是**计算密集型（compute-bound）**，GPU 算力是主要瓶颈。
- **Decode（Token Generation）**：逐字生成的自回归过程。每步仅将上一步新生成的 token 作为输入，计算其Q向量及新的 K、V 向量，append 到历史 KV Cache，再经注意力机制预测下一个 token，循环往复直至结束。
    - 由于每步只算一个 token 却要反复读取整份 KV Cache，Decode 是**访存密集型（memory-bound）**，显存带宽是主要瓶颈。这也是 KV Cache 压缩、量化优化的核心动因。
    
    > 大模型推理时，显存主要被两样东西吃掉：
    **模型权重（Weights）：**比如一个 7B（70亿参数）的模型，用 fp16 精度存，死死占用 **14 GB**显存。这个数字是固定的。
    **KV Cache：**
    这是动态的。随着你的上下文越来越长、对话轮数越来越多，KV Cache 会无休止地疯狂长大。一个 7B 的模型，当长文本达到 32k 甚至 128k 时，光是**KV Cache 占用的显存就能达到几十个 GB**，甚至远远超过模型本身权重的体积！
    为了不用恶心的“全量 Prefill”重算方式，工业界标配了**KV Cache**。
    标配了 KV Cache 之后，发现它在长文本时简直是**显存吞噬者**。
    为了让显存多撑久一点，科学家们才绞尽脑汁卷出了**MQA（单头）**和**GQA（分组头）——通过在空间上砍掉 KV 的头数，来拼命压缩 KV Cache 的体积。**
    > 

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2013.png" alt="An illustration of the key-value caching mechanism (From Nvidia’s blog)">
  <figcaption style="text-align: center;">An illustration of the key-value caching mechanism (From Nvidia’s blog)</figcaption>
</figure>

### 2.4.1 Multi-query attention(MQA) and Grouped -query attention(GQA)

MHA(2017) → MQA(2019) → GQA(2023) 的演进逻辑：

**#query_head 始终不变，只调整 K/V head 的数量：**

- **MHA**「每头一份」：效果最好，但 KV Cache 最大；
- **MQA**「全局一份」：KV Cache 压到最小，但效果掉得明显；
- **GQA**「分组共享」：折中方案，用少量分组找回大部分效果，兼顾显存与质量。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2014.png" style="width:auto; max-height:460px; max-width:100%;" alt="标准Attention: # query_head = #key_head = # value_head">
  <figcaption style="text-align: center;">标准Attention: # query_head = #key_head = # value_head</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2015.png" style="width:auto; max-height:460px; max-width:100%;" alt="MQA: #key_head = #value_head = 1; GQA: #key_head = #value_head = #groups(&gt;1 and &lt; #query_head)">
  <figcaption style="text-align: center;">MQA: #key_head = #value_head = 1; GQA: #key_head = #value_head = #groups(&gt;1 and &lt; #query_head)</figcaption>
</figure>

### 2.4.2 Multi-head Latent Attention(MLA,2024, DeepSeek系)

把输入 $x$ 压缩成一个低维潜向量 $c$，缓存时只存 $c$；借助**矩阵吸收**技巧， $K$和 $V$其实无需真正算出：把 $W_k$吸收进 Q 侧、 $W_v$ 吸收进输出侧，就能直接基于 $c$完成注意力计算，省掉了存储KV的开销。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2016.png" style="width:auto; max-height:460px; max-width:100%;" alt="">
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2017.png" style="width:auto; max-height:460px; max-width:100%;" alt="算attention weight时k无需真正算出，只要q进行转换然后直接和c点积即可">
  <figcaption style="text-align: center;">算attention weight时k无需真正算出，只要q进行转换然后直接和c点积即可</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2018.png" style="width:auto; max-height:460px; max-width:100%;" alt="算输出o时, V 也无需真正算出。先用 attention weight 对c加权求和,再转换即可">
  <figcaption style="text-align: center;">算输出o时, V 也无需真正算出。先用 attention weight 对c加权求和,再转换即可</figcaption>
</figure>

Please see complete formula as [original paper](https://arxiv.org/pdf/2405.04434):

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2019.png" style="width:auto; max-height:460px; max-width:100%;" alt="">
</figure>

### 2.4.3 Sliding Window Attention(2020)

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2020.png" alt="所有层都采用固定大小的滑动窗口注意力：每个 query 只关注自己及前 W 个 token（Mistral 窗口为 4096），窗口随位置滑动。单层感受野虽受窗口限制，但信息能逐层向上传递——底层 token 的信息通过堆叠的层不断扩散，使高层的有效感受野随深度近似线性增长（k 层后约覆盖 k×W 个 token）。图中箭头正是展示这种跨层的信息传播。">
  <figcaption style="text-align: center;">所有层都采用固定大小的滑动窗口注意力：每个 query 只关注自己及前 W 个 token（Mistral 窗口为 4096），窗口随位置滑动。单层感受野虽受窗口限制，但信息能逐层向上传递——底层 token 的信息通过堆叠的层不断扩散，使高层的有效感受野随深度近似线性增长（k 层后约覆盖 k×W 个 token）。图中箭头正是展示这种跨层的信息传播。</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2021.png" alt="采用滑动窗口注意力与全注意力交替使用：部分层只看一个很短的局部窗口（图中小框），部分层则关注全部先前 token">
  <figcaption style="text-align: center;">采用滑动窗口注意力与全注意力交替使用：部分层只看一个很短的局部窗口（图中小框），部分层则关注全部先前 token</figcaption>
</figure>

当关键信息距离超出窗口、且需要精确检索时（比如长文档里的"大海捞针"任务），sliding window attention表现会明显变差。

### 2.4.4 Streaming LLM(2023)

StreamingLLM 是推理时的技巧，核心发现是"注意力汇聚点"（attention sink）。要解决的问题是：当序列无限长、想用一个滚动窗口把旧 KV 直接丢掉时，**一旦最开头那几个 token 被踢出缓存，模型性能就会突然崩溃**。

**解法:保留少数几个起始 token（sink）+ 一个滚动的近期窗口。**

即缓存 = 最前面 4 个左右的 sink token 的 KV + 最近 W 个 token 的 KV。中间的全部丢弃。这样就能在固定大小缓存下，让模型稳定处理几百万 token 的流式输入，且**不需要重新训练**

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2022.png" alt="">
</figure>

### 2.4.5 Pruning KV cache(2023)

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2023.png" alt="可以看到两个现象:一是每一步真正获得注意力的只有一小部分 token,大片位置几乎不被关注;二是少数 token 会在不同位置反复吸走大量注意力(图中那些贯穿始终的深色竖线)。这意味着:既然大多数 token 的 KV 几乎用不上,就可以在推理时只保留这些被反复关注的关键 token 的 KV、丢弃其余,从而把 KV 缓存压到很小而几乎不掉性能。">
  <figcaption style="text-align: center;">可以看到两个现象:一是<strong>每一步真正获得注意力的只有一小部分 token</strong>,大片位置几乎不被关注;二是<strong>少数 token 会在不同位置反复吸走大量注意力</strong>(图中那些贯穿始终的深色竖线)。这意味着:既然大多数 token 的 KV 几乎用不上,就可以在推理时只保留这些被反复关注的关键 token 的 KV、丢弃其余,从而把 KV 缓存压到很小而几乎不掉性能。</figcaption>
</figure>

### 2.4.6 Cross-Conversation Prefix Caching

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2024.png" alt="尤其是agents">
  <figcaption style="text-align: center;">尤其是agents</figcaption>
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2025.png" alt="">
</figure>

### 2.4.7 Summary

- **结构上**:主流不再用原始 MHA,普遍是 GQA,DeepSeek 系是 MLA。
- **注意力范围上**:大多数仍是全局注意力,sliding window 只是部分模型部分层用。
- **FlashAttention**:它是内核优化,和 MLA/GQA 是叠加关系—— 几乎每个现代模型都在用它来算那份(无论什么结构的)注意力。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Attention%20Alternatives/image%2026.png" alt="X表示“不是”；O表示“是”；？表示可以选择是也可以选择不是（每篇paper不一样）">
  <figcaption style="text-align: center;">X表示“不是”；O表示“是”；？表示可以选择是也可以选择不是（每篇paper不一样）</figcaption>
</figure>