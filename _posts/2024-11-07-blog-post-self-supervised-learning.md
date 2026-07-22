---
title: 'Self Supervised Learning'
date:  2024-11-07
permalink: /posts/2024/09/blog-post-policy-gradient/
tags:
  - Self supervised Learning
  - Contrasive Learning
  - Self-distillation
excerpt: |
  🐝 **TL;DR**

  Self-supervised learning (SSL) allows models to automatically learn a "good" representation space using the data in a given dataset without the need for their labels. Specifically, if our dataset were a bunch of images, then self-supervised learning allows a model to learn and generate a "good" representation vector for images.

  **Timeline** : Pretext Tasks → Contrastive Learning → Self-Distillation → **Renaissance (MAE)**
toc: true
---
<div class="notice--info" markdown="1">
🐝 **TL;DR**

  Self-supervised learning (SSL) allows models to automatically learn a "good" representation space using the data in a given dataset without the need for their labels. Specifically, if our dataset were a bunch of images, then self-supervised learning allows a model to learn and generate a "good" representation vector for images.
  
  **Timeline** : Pretext Tasks → Contrastive Learning → Self-Distillation → **Renaissance (MAE)**
</div>

# 0 Intro

**From AlexNet to Self-Supervised Learning**: AlexNet (2012) → people found the penultimate *4096-dim vector* transfers well to new tasks (*DeCAF*, 2014; *CNN Features off-the-shelf*, 2014) → *the representation is the real prize, and it's reusable,* but this still relies on *labeled* ImageNet pretraining → so the question became: *can we drop the labels from pretraining too?* → *self-supervised learning* was born, aiming to(SSL’s final goal) **train an encoder** that rivals supervised ImageNet pretraining **without any labels**.

![                                        Figure 1: AlexNet as a feature extractor](Self%20supervised%20learning/image.png)

                                        Figure 1: AlexNet as a feature extractor

![Figure 2: The SSL paradigm — pretext pretraining on unlabeled data, then transfer to downstream tasks](Self%20supervised%20learning/image%201.png)

Figure 2: The SSL paradigm — pretext pretraining on unlabeled data, then transfer to downstream tasks

**How SSL evolved**: hand-crafted **pretext tasks** (Context Encoders, 2016; Colorization, 2016; Jigsaw, 2016; RotNet, 2018) → worked, but still trailed supervised pretraining → **contrastive learning** (CPC, 2018; MoCo, 2019; SimCLR, 2020) finally matched or beat supervised features → **self-distillation without negatives** (BYOL, 2020; SwAV, 2020; DINO, 2021) → **masked image modeling**, the "image BERT" idea (BEiT, 2021; MAE, 2021) → general-purpose **foundation encoders** (DINOv2, 2023; I-JEPA / V-JEPA, 2023–2024).

[Details-SSL](https://app.notion.com/p/Details-SSL-39230cd0ce598035b5e8f57b44e9ae76?pvs=21)

> 
> 
> 
> Pretext Task
> 
> - Define a task based on the data itself
> - No manual annotation
> - Could be considered an unsupervised task;
> - but we learn with supervised learning objectives, e.g., classification or regression.
> 
> Downstream Task
> 
> - The application you care about
> - You do not have large datasets
> - The dataset is labeled

LeCun has a famous **cake analogy** (which he repeated often, ~2016–2019): **self-supervised learning is the cake itself** (the bulk, where most of the information is), **supervised learning is just the icing** on top, and **reinforcement learning is the cherry** on the cake. The point: most of what an intelligent system learns should come from unlabeled data, not from labels or rewards.

![                            Figure 3: LeCun talk in **NIPS 2016**](Self%20supervised%20learning/image%202.png)

                            Figure 3: LeCun talk in **NIPS 2016**

# 1 Self-supervised pretext tasks

| Pretext Type | Pretext tasks | How it works | Input | Output | **Extension** |
| --- | --- | --- | --- | --- | --- |
| Classification | Relative patch locations(2015) | Take a center patch + one of its 8 neighbors; predict where the neighbor sits relative to the center | A pair of patches (center + one neighbor) | Which of the 8 relative positions (8-way) |  |
|  | Jigsaw puzzles(2016) | Cut image into 3×3 = 9 patches, shuffle them by a permutation from a predefined set. Each patch goes through its own CNN branch (9 branches, **shared conv weights**); features are concatenated, then classified | Shuffled patches | which index in the predefined permutation set |  |
|  |  Rotations prediction
(RotNet, 2018) | Rotate the image by one of 4 angles {0°, 90°, 180°, 270°}; the net predicts the applied angle | A rotated image | Which of the 4 rotations (4-way) |  |
| Generation | Inpainting
(Context Encoders, 2016) | Mask out a region of the image; the net reconstructs the missing content | Image with a masked region | The missing region |  **→ MAE** (Masked Autoencoder, He et al., 2021): same masked-prediction idea, but built on **ViT**, masks a very high ratio (**75%**) of patches, and uses an **asymmetric encoder-decoder** (encoder sees only visible patches). Simple pixel reconstruction, no GAN. **Self-supervised pretraining that finally surpassed supervised ImageNet pretraining** — scalable and SOTA. |
|  | Colorization(2016) | Convert image to Lab space; feed the L (lightness) channel, predict the missing ab color channels; combine L+ab to recover the color image | Grayscale image (L channel) | Predicted a/b color channels | → **video colorization** (Vondrick 2018): propagate color from a reference frame via cross-frame attention; the model implicitly learns **tracking** |
- More details about Context Encoders(Pathak et al., 2016).
    
    Context Encoder (2016) is one of the foundational **early works** in self-supervised vision. It **proved that the direction of "masked prediction can learn transferable features" is viable, but in absolute numbers it still fell well short of supervised pretraining.**
    
    ![                                    Figure 4: Context Encoder: encoder → bottleneck → decode](Self%20supervised%20learning/image%203.png)
    
                                        Figure 4: Context Encoder: encoder → bottleneck → decode
    
    **Input (far left):** An image with a white square hole cut out of the center — this is what the network receives.
    
    **Encoder (red trapezoid):** Encodes the masked image into Encoder Features (the bottleneck representation).
    
    **Channel-wise Fully Connected (middle horizontal lines):** The key intermediate layer. It applies fully-connected connections within each channel, propagating information across spatial locations so the decoder has enough context to infer what belongs in the hole.
    
    **Decoder (second red trapezoid):** Decodes the features to generate the content of the missing region. Once training is complete, the decoder will be **discarded**. 
    
    **The two patches in the bracket (far right):** This is where the loss $\mathcal{L}(\cdot,\cdot)$ is computed.
    
    - Left patch (dark, blurry) = the region **predicted** by the network.
    - Right patch (sharp, with player "7") = the **ground-truth** missing region.
    - $\mathcal{L}$ is the reconstruction loss comparing the two (masked L2 + adversarial).
    
    ---
    
    Next, let’s see the details of the $\mathcal{L}(\cdot,\cdot)$.
    
    ![                                            Figure 5: Loss in Context Encoder](Self%20supervised%20learning/image%204.png)
    
                                                Figure 5: Loss in Context Encoder
    
    Reconstruction loss(  $L₂$ ) ensures pixel alignment, but the generated regions tend to be blurry. To achieve sharp texture details, the authors introduce the concept of GANs.
    
    A discriminator $D$ is introduced to play a real-vs-fake adversarial game:
    
    - **$x$ (Real Image):** $D$ aims to output a high score (close to 1) for the perfect original image, maximizing $\log(D(x))$.
    - **$D(F(\dots))$ (Inpainted Image):** $D$ aims to spot the forgery and output a low score (close to 0), minimizing  $D(F(\dots))$ to maximize $\log(1 - D(F(\dots)))$.
    
    By acting as a strict critic, $D$ forces the generator $F$ to move beyond producing blurry, averaged pixels, compelling it to synthesize high-quality images with realistic details that can fool $D$.
    
- More details about Split-brain Autoencoder([Zhang et al., 2017](https://arxiv.org/pdf/1611.09842)).
    
    ![                                                       Figure 6: Color a/b Generation](Self%20supervised%20learning/image%205.png)
    
                                                           Figure 6: Color a/b Generation
    
    If the overall network $\mathcal{F}$ is a standard CNN (e.g., AlexNet), the subnetworks $\mathcal{F}_1$ and $\mathcal{F}_2$ are obtained by splitting each layer in half along the channel dimension. In implementation, this is equivalent to using grouped convolution with `groups=2`. 
    
    In aggregate, $\mathcal{F}_1$ and $\mathcal{F}_2$ are essentially convolutional encoder networks that each handle half of the channel dimension, called "cross-channel encoders".
    
    ![                                                          Figure 7: Split-brain Autoencoder (1)](Self%20supervised%20learning/b53c6893-edfd-48a5-b2cc-aa09f3b4eee2.png)
    
                                                              Figure 7: Split-brain Autoencoder (1)
    
    ![                                                            Figure 8: Split-brain Autoencoder (2)](Self%20supervised%20learning/image%206.png)
    
                                                                Figure 8: Split-brain Autoencoder (2)
    

# 2 Contrastive Representation Learning

Early self-supervised learning relied on handcrafted pretext tasks (rotation prediction, jigsaw, colorization, etc.). These had two problems: (1) designing pretext tasks one by one is tedious, and (2) the learned representations tend to be tied to the specific task and aren't general enough. Even later-born stronger reconstruction-based methods like MAE, while excellent at fine-tuning, **underperform at linear probing—because their objective is pixel reconstruction, the representations aren't explicitly pushed toward semantic linear separability.**

So can we find a more general pretext task? This led to **contrastive learning** (pulling positive samples together and pushing negatives apart to directly learn semantically linear-separable, transferable representations).

What we want:

$$
score(f(x),f(x^+)) >> score(f(x),f(x^-))
$$

with x: reference sample; x+ positive sample; x- negative sample.

Given a chosen score function, we **aim to learn an encoder function $f$** that yields high score for positive pairs $(x, x^+)$ and low scores for negative pairs $(x, x^-)$.

## 2.1 A formulation of contrastive learning

---

In this part, I'll introduce InfoNCE, which is widely used.

Again, InfoNCE is just one mathematical formalization of contrastive learning, not the only one.

**The differences between formulations usually lie in:** 

- **The choice of the score function**,
- **How to quantify " $\gg$"**(turn this vague objective into a differentiable and optimizable loss function,e.g., using a margin or a softmax function)
- **How negative samples are utilized** (e.g., using a single negative, N negatives, or implicit negatives).

### 2.1.1 InfoNCE v.s. Cross entropy loss

![                  Figure 9: InfoNCE loss is actually N-way cross entropy loss!](Self%20supervised%20learning/image%207.png)

                  Figure 9: InfoNCE loss is actually N-way cross entropy loss!

with $f(·)$ means encoder we want to learn which maps sample into feature embedding, and $s(a,b)$ means similarity score.  

Actullay, InfoNCE is cross entropy loss with N-way classifier :

If the ground-truth label is represented as a one-hot vector $\mathbf{y} = [y_1, y_2, \dots, y_N]$ (where $y_c = 1$ for the true class and $0$ otherwise):

$$
L = -\sum_{i=1}^{N} y_i \log(p_i)
$$

where $p_i$ is the predicted probability of the current sample belonging to class $i$, computed via the Softmax function:

$$
p_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}
$$

Since $\mathbf{y}$ is a one-hot vector, the loss simplifies to the negative log-probability of the true class $t$:

$$
L = -\log(p_c) = -\log\left( \frac{e^{z_c}}{\sum_{j=1}^{N} e^{z_j}} \right)
$$

In practice, we average the loss over all samples within a mini-batch of size $B$:

$$
\text{Total Loss} = -\frac{1}{B} \sum_{i=1}^{B} \log\left( \frac{e^{z_{i,c}}}{\sum_{j=1}^{N} e^{z_{i, j}}} \right)
$$

### 2.1.2 InfoNCE lower bound

![                      Figure 10: InfoNCE loss and its mutual information lower bound](Self%20supervised%20learning/image%208.png)

                      Figure 10: InfoNCE loss and its mutual information lower bound

- Detailed Derivation
    
    我们将正样本 $x^+$ 记为 $y_1$，将 $N-1$ 个负样本 $x_j^-$ 记为 $y_2, \dots, y_N$。
    此时的样本分布为：主体 $x \sim p(x)$，正样本 $y_1 \sim p(y\vert{}x)$，负样本 $y_{2:N} \sim p(y)$。
    
    1. **变形 $\log(N) - L$**
    首先，我们将$\log(N)$移到 InfoNCE 损失函数 $L$ 中的期望内部：
    
    $$
    \log(N) - L = \mathbb{E} \left[ \log(N) + \log \frac{e^{s(x, y_1)}}{\sum_{i=1}^N e^{s(x, y_i)}} \right] = \mathbb{E}_{p(x)p(y_1\vert{}x)\prod_{i=2}^N p(y_i)} \left[ \log \frac{e^{s(x, y_1)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \right]
    $$
    
    1. **作差并引入互信息 $I(x; y_1)$**
    根据互信息的数学定义， $I(x; y_1) = \mathbb{E} \left[ \log \frac{p(y_1\vert{}x)}{p(y_1)} \right]$。我们将两式相减: 
        
        $$
        \log(N) - L - I(x; y_1) = \mathbb{E}_{p(x)p(y_1\vert{}x)\prod_{i=2}^N p(y_i)} \left[ \log \left( \frac{e^{s(x, y_1)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \cdot \frac{p(y_1)}{p(y_1\vert{}x)} \right) \right]
        $$
        
    2. **应用詹森不等式 (Jensen's Inequality)**
    因为对数函数 $\log$ 是凹函数，根据詹森不等式 $\mathbb{E}[\log Z] \leq \log \mathbb{E}[Z]$，我们可以将期望提到 $\log$ 外面，从而将公式放大：
        
        $$
        \log(N) - L - I(x; y_1) \leq \log \left( \mathbb{E}_{p(x)p(y_1\vert{}x)\prod_{i=2}^N p(y_i)} \left[ \frac{e^{s(x, y_1)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \cdot \frac{p(y_1)}{p(y_1\vert{}x)} \right] \right)
        $$
        
    3. **核心魔术：概率测度变换 (Change of Measure)**
    注意看期望符号底下的真实积分分布是 $p(x)p(y_1\vert{}x)\prod_{i=2}^N p(y_i)$，而括号里面乘了一个权重 $\frac{p(y_1)}{p(y_1\vert{}x)}$。两者相乘，**刚好把条件概率 $p(y_1\vert{}x)$ 约掉了**：
        
        $$
        p(x)p(y_1\vert{}x)\prod_{i=2}^N p(y_i) \cdot \frac{p(y_1)}{p(y_1\vert{}x)} = p(x)p(y_1)\prod_{i=2}^N p(y_i) = p(x)\prod_{i=1}^N p(y_i)
        $$
        
        这意味着，在变换后的新分布下，**所有的 $y_1, y_2, \dots, y_N$ 之间没有任何区别，全部变成了独立同分布（i.i.d.）于边缘分布 $p(y)$ 的样本**。上面的积分项可以简化为：
        
        $$
        \mathbb{E}_{p(x)\prod_{i=1}^N p(y_i)} \left[ \frac{e^{s(x, y_1)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \right]
        $$
        
    4. **利用对称性 (Symmetry) 完型填空**
    既然在这个新分布下所有 $y_i$ 都是完全对等的，那么把其中的 $y_1$ 换成任何一个 $y_i$，其期望值都必然完全相同。因此，我们可以用全员的平均值来代替单项：
        
        $$
        \mathbb{E} \left[ \frac{e^{s(x, y_1)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \right] = \frac{1}{N} \sum_{i=1}^N \mathbb{E} \left[ \frac{e^{s(x, y_i)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \right] = \frac{1}{N} \mathbb{E} \left[ \frac{\sum_{i=1}^N e^{s(x, y_i)}}{\frac{1}{N}\sum_{i=1}^N e^{s(x, y_i)}} \right] = \frac{1}{N} \cdot N = 1
        $$
        
    5. 将这个完美的结果 $1$ 代回第 3 步的不等式中：$\log(N) - L - I(x; y_1) \leq \log(1) = 0$移项整理，即得证：$I(x; x^+) \geq \log(N) - L$。

## 2.2 Contrastive Predictive Coding (CPC)

Give the model the first half of a sequence and let it guess what comes next. If it guesses right, it learns.The fundamental breakthrough of CPC is that it frames the self-supervised task of predicting the future as a contrastive matching problem like a multiple-choice question instead of a generative regression task requiring direct reconstruction. This approach effectively bypasses the immense difficulty of generating high-dimensional raw signals while still forcing the model to learn highly useful, high-level representations.

![                        Figure 11: Instance-Level vs. Sequence-Level Learning
Instance-level learning, such as contrastive learning via data augmentation, answers the core question: "Are these two images different views of the same object?" It creates positive samples through data augmentation on a single instance to capture invariant features, essentially learning what things look like the same entity. 
In contrast, sequence-level learning like CPC answers the question: "Given the preceding sequence, what should come next?" It leverages temporal succession to create positive samples, where the true future segment is positive and random segments are negative, allowing the model to capture temporal dynamics and learn how sequences evolve over time.](Self%20supervised%20learning/image%209.png)

                        Figure 11: Instance-Level vs. Sequence-Level Learning
Instance-level learning, such as contrastive learning via data augmentation, answers the core question: "Are these two images different views of the same object?" It creates positive samples through data augmentation on a single instance to capture invariant features, essentially learning what things look like the same entity. 
In contrast, sequence-level learning like CPC answers the question: "Given the preceding sequence, what should come next?" It leverages temporal succession to create positive samples, where the true future segment is positive and random segments are negative, allowing the model to capture temporal dynamics and learn how sequences evolve over time.

The following is CPC core workflow:

**1) Encoding**
• **Action**: Slice the raw audio into segments ( $\dots, x_{t-1}, x_t$) and feed each into the encoder $g_{enc}$.
• **Output**: A low-dimensional feature vector $z_t$ for each time step.
• **Essence**: Transforms raw signals into meaningful latent representations.

**2) Summarizing History**
• **Action**: Feed all historical $z$ vectors ( $\dots, z_{t-1}, z_t$) up to the current step into an autoregressive model $g_{ar}$ (e.g., RNN/GRU).
• **Output**: A context vector $c_t$.
• **Essence**: $c_t$ summarizes all the content heard so far.

**3) Predicting the Future**
• **Action**: Rather than explicitly generating raw waveforms($x$), this step projects $c_t$ into the latent space (via a linear transformation $W_k c_t$) to score against future steps ($z_{t+1}, z_{t+2}, \dots z_{t+k}$). 
• **Key Point**: Note that each future step $k$ has its own independent weight matrix $W_k$.

$$
s_k(z_{t+k},c_t)=z_{t+k}^TW_kc_t
$$

> **Step 1 ( $z_{t+1}$)** uses $W_1$
**Step 2 ( $z_{t+2}$)** uses $W_2$
 $\dots$
**Step $K$ ( $z_{t+K}$)** uses $W_K$
 
**Intuition:** Predicting the "immediate next step" versus "5 steps ahead" are tasks of different difficulty and nature (further is harder and more uncertain). Assigning a dedicated $W_k$ for each step optimizes the unique projection from current context $c_t$ to that specific future distance.
> 

**4) Contrastive Learning**
• **Action**: Compute InfoNCE loss for each future k features.
    ◦ **Positive sample**: The actual next feature $z_{t+k}$.
    ◦ **Negative samples**: Randomly sampled $z$ features from elsewhere.

$$
\mathcal{L}_k = -\mathbb{E} \left[ \log \frac{\exp \left(s_k(z_{t+k}, c_t)\right)}{\sum_{z_j \in \{z_{t+k}\} \cup X_{\text{neg}}} \exp \left(s_k(z_j, c_t)\right)} \right]
$$

$$
\mathcal{L} = \frac{1}{K}\sum_{k=1} ^{K}\mathcal{L}_{k}
$$

• **Goal**: Identify the true future sample out of the noise.

![                               Figure 12: Workflow of Contrastive Predictive Coding(CPC)](Self%20supervised%20learning/image%2010.png)

                               Figure 12: Workflow of Contrastive Predictive Coding(CPC)

## 2.3 SimCLR: A Simple Framework for Contrastive Learning

Given an image **x**, SimCLR uses two different data augmentation schemes **t** and **t'** to generate the positive pair of images $\tilde{x}_i$ and $\tilde{x}_j$. 𝑓 is a basic encoder net that extracts representation vectors from the augmented data samples, which yields  $h_i$ and $h_j$, respectively. Finally, a small neural network projection head 𝑔 maps the representation vectors to the space where the contrastive loss is applied. The goal of the contrastive loss is to maximize agreement between the final vectors $z_i = g(h_i)$ and $z_j = g(h_j)$.

The base encoder 𝑓 extracts representation vectors for the augmented samples. The SimCLR paper found that using deeper and wider models improved performance and thus usually chose [ResNet](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1512.03385.pdf) to use as the base encoder. The output of the base encoder are the representation vectors $h_i = f(\tilde{x}_i)$and $h_j = f(\tilde{x}_j)$.

The projection head 𝑔 is a small neural network that maps the representation vectors $h_i$ and $h_𝑗$to the space where the contrastive loss is applied. The paper found that using a nonlinear projection head improved the representation quality of the layer before it. Specifically, they used a MLP with one hidden layer as the projection head 𝑔. The contrastive loss is then computed based on the outputs $z_i = g(h_i)$ and $z_i = g(h_i)$.

![image.png](Self%20supervised%20learning/image%2011.png)

After training is completed, we throw away the projection head $g$ and only use 𝑓 and the representation ℎ to perform downstream tasks, such as classification.

### 2.3.1 Design choices

SimCLR relies on **three key design choices**.

- Use a non-linear projection network $g(·)$ to project features to a space where contrastive
learning is applied. $g(·)$ is a 2-layer MLP, where the output is 128-dimensional. After training, $g$ is discarded, and downstream tasks use $h$ instead of $z$.
    
    $$
    z = W_2 \cdot ReLU(W_1 \cdot h)
    $$
    
    - Let me finalize the explanation why the projection head works.
        
        contrastive loss 强迫 z 对增强不变 → 增强相关信息(颜色、旋转等)在 z 中被丢弃 → 这些信息对下游任务可能有用 → g(·) 吸收该信息损失,h 保留更多信息 → **下游用 h,g 训练后丢弃**
        
        证据(两条,均来自原paper Chen et al., 2020):
        
        1. 用 h 能预测出施加的增强类型,用 z 不能 → 直接验证"信息丢在 z、留在 h"
        2. Fig. 2.4-1:Non-linear projection head 下的性能对 z 维度不敏感(32~2048 持平)→ 起作用的是非线性缓冲层本身,而非投影维度

![                          Figure 13: Non-linear projection head improves linear evaluation 
                                               top-1 by ~3% over linear and ~10% over no projection.](Self%20supervised%20learning/image%2012.png)

                          Figure 13: Non-linear projection head improves linear evaluation 
                                               top-1 by ~3% over linear and ~10% over no projection.

- Generate positive samples through composition of data augmentations. SimCLR conducted a systematic ablation study and found that no single data augmentation is sufficient on its own; they must be combined. The most critical combination is **random crop + color distortion** (color jitter/grayscale).

![Figure 14: Illustrations of the studied data augmentation operators. Each augmentation can transform data stochastically with some internal parameters (e.g. rotation degree, noise level).  Note that the paper only tested these operators in ablation, the augmentation policy used to train the models only includes random crop (with flip and resize), color distortion, and Gaussian blur.](Self%20supervised%20learning/image%2013.png)

Figure 14: Illustrations of the studied data augmentation operators. Each augmentation can transform data stochastically with some internal parameters (e.g. rotation degree, noise level).  Note that the paper only tested these operators in ablation, the augmentation policy used to train the models only includes random crop (with flip and resize), color distortion, and Gaussian blur.

![Figure 15: Linear evaluation (ImageNet top-1 accuracy) under individual or composition of data augmentations, applied only to one branch. For all columns but the last, diagonal entries correspond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.](Self%20supervised%20learning/image%2014.png)

Figure 15: Linear evaluation (ImageNet top-1 accuracy) under individual or composition of data augmentations, applied only to one branch. For all columns but the last, diagonal entries correspond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.

- **Use large batch size.** For example, a batch size of 4096 provides $2N = 8192$ augmented images per step, offering a large number of in-batch negative samples.

![                               Figure 16: Why does large batch size work?](Self%20supervised%20learning/image%2015.png)

                               Figure 16: Why does large batch size work?

In addition, using cosine similarity as the score function.

### 2.3.2 Implementation

![        Figure 17: SimCLR mini-batch training — affinity matrix as a 2N-way classification problem](Self%20supervised%20learning/image%2016.png)

        Figure 17: SimCLR mini-batch training — affinity matrix as a 2N-way classification problem

![       Figure 18: SimCLR algorithm — positive pairs via augmentation, in-batch negatives via InfoNCE](Self%20supervised%20learning/image%2017.png)

       Figure 18: SimCLR algorithm — positive pairs via augmentation, in-batch negatives via InfoNCE

[Data Augmentation](https://github.com/during-gt/Assignment3/blob/2ff70922c19937ca9aa97ad088ca7112d5587c31/cs231n/simclr/data_utils.py#L7)→ [Base Encoder(ResNet) and Projection Head(MLP)](https://github.com/during-gt/Assignment3/blob/2ff70922c19937ca9aa97ad088ca7112d5587c31/cs231n/simclr/model.py#L7) → [Contrastive Loss](https://github.com/during-gt/Assignment3/blob/2ff70922c19937ca9aa97ad088ca7112d5587c31/cs231n/simclr/contrastive_loss.py#L123)

## 2.4 Momentum Contrastive Learning (MoCo)

### 2.4.1 Design choices

- **Queue as Past Negative Sample Pool** which decouples negative sample size from batch size. Specifically, using a FIFO queue (65,536 keys) accumulates encoded keys across iterations: the current batch is enqueued, the oldest batch dequeued. Negatives come from many past batches, not like SimCLR.

![image.png](Self%20supervised%20learning/image%2018.png)

- **Momentum Encoder: Ensuring Representation Consistency**
    
    Using a “key” pool introduces a major challenge: the keys inside are encoded at different timesteps. If the key encoder updates rapidly via gradients, the pool becomes representationally inconsistent, causing the training to collapse (paper authors tried simply copying the query encoder then failed).
    
    The solution is a **momentum encoder** that bypasses gradient updates and evolves slowly:
    
    $$
    \theta_k \leftarrow m\theta_k + (1-m)\theta_q \quad (m=0.999)
    $$
    
    This extreme smoothness ensures all keys in the pool are represented in a virtually identical feature space. **This momentum mechanism was later adopted by BYOL and DINO (renamed as the "teacher"), becoming a foundational cornerstone of self-supervised learning.**
    
- **Shuffling BN**
    
    Because small batch sizes make BN statistics highly sensitive and easily 'hijacked' by individual samples, severe information leakage occurs. This is precisely why MoCo was forced to engineer Shuffling BN.
    
    MoCo shuffles the sample order across multiple GPUs before running the key encoder (and deshuffles afterward). This ensures that the query and its positive key are processed in different sub-batches, forcing the model to rely on semantic features rather than shared BN statistics.
    
    ![                  Figure 19: MoCo's Shuffling BN ](Self%20supervised%20learning/image%2019.png)
    
                      Figure 19: MoCo's Shuffling BN 
    
    [More details](https://app.notion.com/p/More-details-39e30cd0ce5980988426fca385af466e?pvs=21)
    
    > To address this same leakage issue, SimCLR employs **Global BN (SyncBN)**, while other modern architectures completely bypass the problem by replacing BN with **Layer Normalization (LN)**.
    > 

### 2.4.2 Implementation

![                                            Figure 20: MoCo's learning algorithm (He et al., 2020) ](Self%20supervised%20learning/image%2020.png)

                                            Figure 20: MoCo's learning algorithm (He et al., 2020) 

### 2.4.3 MoCo V2

A hybrid of ideas from SimCLR and MoCo:
● From SimCLR: non-linear projection head and strong data augmentation.
● From MoCo: momentum-updated queues that allow training on a large number of negative samples (no TPU required!).

# 3 Summary of Contrastive Representation Learning

![image.png](Self%20supervised%20learning/image%2021.png)

![image.png](Self%20supervised%20learning/2358c59b-8895-411e-a1ef-51d67f85e298.png)

![image.png](Self%20supervised%20learning/01fcee83-b6d8-4821-96ea-7ce34e0ec2c9.png)

# 4 DINO: Self-Distillation with No Labels

Models trained with vanilla contrastive learning methods such as SimCLR require very large batch sizes. This makes them computationally expensive and limits their accessibility. Subsequent works, like [BYOL](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2006.07733), propose an alternative approach that avoids the need for numerous negative samples by using a student-teacher framework. This method performs surprisingly well and was later adopted by [DINO](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2104.14294)(2021) .

DINO uses two separate encoders which are trained differently. The student network is updated via backpropagation to match the outputs of the teacher network. The teacher network is not updated via backpropagation; instead, its weights are updated using an exponential moving average (EMA) of the student's weights. This means that the teacher model evolves more slowly and provides a stable target for the student to learn from.

![                                Figure 21: Core idea of DINO (Caron et al., 2021) ](Self%20supervised%20learning/image%2022.png)

                                Figure 21: Core idea of DINO (Caron et al., 2021) 

**💡 Why DINO is Called "Self-Distillation"？**

In traditional Knowledge Distillation (KD), a lightweight **Student** learns from a massive, **pre-trained Teacher** (e.g., using a ResNet-152 to train a MobileNet). DINO completely flips this script, earning the name **"Self-Distillation"** due to three core design principles:

- **Identical Architectures:** The Student and Teacher share the exact same neural network structure (e.g., both are ViT-Base). There is no architectural or capacity gap between the two.
- **No Pre-trained Expert:** DINO does not require a pre-trained mentor to start. At initialization, the Teacher network is just as blank as the Student.
- **The "Past Me" Guides the "Current Me":** The Teacher’s weights are updated without gradients; instead, they are updated using the **Exponential Moving Average (EMA)** of the Student's weights. This makes the Teacher a temporally smoothed, more stable version of the Student's historical states.

So, the Teacher’s knowledge is entirely derived from the Student’s own historical parameters, the framework forms a closed loop where the model is essentially teaching itself to evolve. Hence, the name **Self-Distillation**.

## 4.1 Review

[Again](https://app.notion.com/p/Self-supervised-learning-38830cd0ce598099a1fdf63e3aca7f20?pvs=21), here is the conceptual essence of contrastive learning:

$$
score(f(x),f(x^+)) >> score(f(x),f(x^-))
$$

While **positive alignment** ("正对一致", the first half) is a universal objective shared by nearly all SSL frameworks, **negative separation** ("负对分离", the second half) is merely one specific realization to prevent representation collapse. 

换句话说，如果一种方法仅仅以最大化相似度得分 $score(f(x),f(x^+))$ 为目标，在理论上极易陷入**平凡解（Trivial Solution），**即模型发生坍塌，对所有输入都输出同一个常数向量。此时，虽然正样本对的对齐得分达到了完美，但最终得到的表征却不包含任何有效信息。
为了消除这种坍塌解，自监督学习演化出了三条技术路线：

- **对比学习方法（Contrastive Methods）**：依赖**负对分离**，通过主动将不相似的样本对推开来拉伸特征空间，如基于InfoNCE的SimCLR, MoCo。
- **非对比学习方法（Self-Distillation Methods）**：则通过一套精妙的架构“组合拳”来达到同样的效果，如 BYOL, SimSiam, DINO，其核心机制包括：
    - **梯度截断（Stop-gradient）** —— *必需组件：防止表征坍塌的灵魂锚点 (SimSiam 消融证明去掉即坍塌）。*
    - **非对称设计（Asymmetry）** —— *通过引入预测器（Predictor) 打破网络对称性。*
    - **Centering / Sharpening** —— *DINO 采用的输出分布调控：centering 防止坍缩到单一维度，sharpening 防止退化为均匀分布，两者相互制衡。*
    - **动量教师网络（EMA Teacher）** —— *可选组件：用于提供更稳定的优化目标。*
- **信息最大化 / 正则化方法（Information-Maximization Methods）**：不依赖负样本，也不依赖非对称架构，而是对特征的统计量施加显式约束来防止坍塌，如 VICReg（方差正则强制各维度保持方差下限）、Barlow Twins（交叉相关矩阵逼近单位阵以去除冗余）。

| Non-contrastive Method | stop-gradient | predictor
(asymmetric) | EMA teacher | Unique Mechanism |
| --- | --- | --- | --- | --- |
| **BYOL(2020)** | ✔️ | ✔️ | ✔️ | — (full component set) |
| [**SimSiam](https://arxiv.org/pdf/2011.10566)(2020)** | ✔️ | ✔️ | ✗ | Ablation shows EMA is unnecessary; stop-grad + predictor suffice |
| **DINO(2021)** | ✔️ | ✗ | ✔️ | Centering (prevents collapse to one dim) + sharpening (prevents uniform output) |
| VICReg(2021) | ✗ | ✗ | ✗ | Variance + covariance regularization — explicit statistical constraints, no architectural tricks |

> 
> 
> 
> **Aside**
> Despite algorithmic differences, SSL methods share the same ultimate goal: **preventing representation collapse and preserving feature discriminativeness.**
> 
> Furthermore, foundational theoretical work (e.g., Garrido et al., 2023) has proven a **mathematical duality** between contrastive objectives and **feature-regularized non-contrastive objectives** (e.g., Barlow Twins, VICReg). 
> 
> In contrast, 为什么自蒸馏（Self-Distillation）能稳妥避开坍缩并学出高阶语义，至今依然是深度学习理论界的一个**开放性问题（Open Problem）。**
> 

## 4.2 Design choices

- **Cross-entropy for distribution matching**
    
    Two views of the same input image are encoded by a student and a teacher network respectively. The teacher's output passes through a low-temperature softmax to produce a sharper distribution $t$, while the student outputs a distribution $s$, with the loss $H(t, s)$ requiring them to match.
    
    $$
    H(t, s) = - \sum_{i} t_i \log s_i
    $$
    
    where:
    
    $i$ denotes the dimension of the model's output.
    
    $t_i$ is the sharpened probability from the teacher, serving as the pseudo-label.
    
    $s_i$ is the predicted probability output by the student.
    
- **Momentum teacher**
    
    The student and teacher share identical architectures. The teacher's parameters are an exponential moving average (EMA) of the student's and receive no gradients (stop-gradient), providing a smoother and more stable optimization target than the student itself.
    
- **Centering + Sharpening**
    
    DINO has no negative samples and no predictor like BYOL or SimSiam. Centering subtracts a running average from the teacher output to prevent collapse into a single dimension. Sharpening uses a low temperature $\tau_t$ to sharpen the distribution, keeping it from becoming a flat uniform distribution.
    
- **Multi-crop with cross prediction**
    
    Each image generates 2 global crops and multiple local crops. The teacher encodes only the global crops, while the student encodes all of them; cross-prediction losses are then computed. The loss sums over all pairs (t(g), s(v)), where g ∈ globals, v ∈ all crops, and v ≠ g (no self-pairing). This forces the model to learn "local patch → global semantics" correspondences, and since local crops are low-resolution, they are computationally cheap.
    
    ```python
    # gs, gt: student and teacher networks
    # C: center (K)
    # tps, tpt: student and teacher temperatures
    # l, m: network and center momentum rates
    
    gt.params = gs.params
    for x in loader:
        # 1. 增强生成 Multi-crop：2 个 global crops + M 个 local crops (如 2 + 6 = 8 个 views)
        crops = multi_crop_augment(x)  # crops 为包含多个 Tensor 的列表
        
        # 2. teacher 只看前 2 个 global crops
        t_out = [gt(crop) for crop in crops[:2]]  # 输出列表：[t1, t2]
        
        # 3. student 看全部的 crops (global + local)
        s_out = [gs(crop) for crop in crops]     # 输出列表：[s1, s2, s_local1, s_local2, ...]
        
        # 4. 计算交叉预测 Loss
        loss = 0
        n_terms = 0
        for i, t in enumerate(t_out):
            for j, s in enumerate(s_out):
                if i == j: 
                    continue  # 排除同一 view 的自我预测 (例如跳过 H(t1, s1) 和 H(t2, s2))
                loss += H(t, s)
                n_terms += 1
                
        loss = loss / n_terms  # 均摊到所有有效 view 对上
    
        loss.backward()  # 梯度只更新 student
    
        # student, teacher and center updates
        update(gs)
        gt.params = l * gt.params + (1 - l) * gs.params
        C = m * C + (1 - m) * torch.cat(t_out).mean(dim=0)
    
    def H(t, s):
        t = t.detach()  # stop gradient
        s = softmax(s / tps, dim=1)
        t = softmax((t - C) / tpt, dim=1)  # center + sharpen
        return - (t * log(s)).sum(dim=1).mean()
    ```
    

## 4.3 Implementation

In the original DINO paper, to make the pseudocode simpler and easier to read, the authors provided this basic version without multi-crop below. So, let’s go through this step by step.

![                                                     Figure 22: DINO algorithm ](Self%20supervised%20learning/image%2023.png)

                                                     Figure 22: DINO algorithm 

**`gt.params = gs.params`** — The teacher is not pretrained; it starts from the same initialization as the student. It never receives gradients afterward, only passively tracks the student via EMA.

**`x1, x2 = augment(x), augment(x)`** — Two independent augmentations of the same batch, as in SimCLR; the augmentations define what the representation should be invariant to.

**`s1, s2 = gs(x1), gs(x2)` / `t1, t2 = gt(x1), gt(x2)`** — Four forward passes, each outputting $n×K$ ( $K = 65536$ in DINO). Note the $K$ dimensions correspond to learnable prototypes—the final weight-normalized linear layer (256 × 65536) holds 65536 learnable vectors, each acting as a soft cluster center that emerges during training with no predefined semantics; the outputs are logits over these prototypes, 

> 要区分两件事：
**维度 K 是固定的**——网络永远输出 65536 维，这是设计选择（论文 ablation 过，大 K 效果好，65536 之后饱和）；
**prototypes 的内容是学出来的**——那 65536 个向量指向哪里、各自捕捉什么视觉模式，完全由训练决定，而且不保证全部被有效使用。

类比监督学习：分类头输出维度也固定（如 ImageNet 的 1000），但那里每一维有预先绑定的语义（dim 3 = 鲨鱼）；DINO 的 65536 维没有任何预绑定语义，只是给了模型"最多可以用这么多个软聚类"的容量
> 

![  Figure 23: Projection head design ](Self%20supervised%20learning/image%2024.png)

  Figure 23: Projection head design 

**`loss = H(t1, s2)/2 + H(t2, s1)/2`** — Cross-entropy averaged over both directions. The information flow is fixed: **the teacher produces targets, the student learns to predict them**. This is "self-distillation" — no ground-truth labels; the teacher's output distribution serves as soft labels.

**`update(gs)`** — Gradients update only the student. The teacher's part of the loss is detached (see H), so no gradient flows through it.

**`gt.params = l*gt.params + (1-l)*gs.params`** — The teacher's EMA update, essentially MoCo's momentum encoder under a different name, serving the same purpose: providing a **slowly evolving, stable target**.

**`C = m*C + (1-m)*cat([t1,t2]).mean(dim=0)`** — The center is a running average of teacher outputs ( $C_n = T·(1−0.9ⁿ)$, asymptotically approaching the teacher's long-term mean). It tracks "which dimensions the teacher favors on average.”

**`def H(t, s)`**:

- **`t = t.detach()`** — Stop-gradient (even though the optimizer only holds the student's parameters, the gradient is cut here again).
- **`s = softmax(s / tps)`** — The student uses a higher temperature (0.1), yielding a flatter distribution that preserves learning flexibility.
- **`t = softmax((t − C) / tpt)`** — The teacher applies two operations (centering, then sharpening); **this line is the core of DINO's collapse prevention**. See 4.4.

**`return −(t * log(s)).sum(dim=1).mean()`** — Standard cross-entropy: the student's distribution s is fit to the teacher's distribution t.

## 4.4 How does ”Centering + Sharpening“  work?

Centering or sharpening alone leads to collapse——same output for every input; together they anchor the teacher's output in a sweet spot—neither one-hot nor uniform. This is DINO's most elegant design, and the paper's collapse study (see below) confirms that removing either causes collapse.

![       Figure 24: Collapse study — either alone collapses (KL → 0)](Self%20supervised%20learning/image%2025.png)

       Figure 24: Collapse study — either alone collapses (KL → 0)

---

![                          Figure 25: DINO's collapse analysis](Self%20supervised%20learning/image%2026.png)

                          Figure 25: DINO's collapse analysis

In summary, a "good" representation requires the teacher's outputs to be sharp individually, yet flat overall:

- **Per image, sharp**: each image's distribution decisively points to a few prototypes (ensured by sharpening, smaller $τ_t = 0.04$ than $\tau_s=0.1$)
    - Sharpening
        
        Teacher's output distribution over K prototypes; $p_i$ is the probability assigned to prototype $i$:
        
        $$
        p_i = \frac{\exp((t_i - c_i)/\tau_t)}{\sum_{j=1}^K \exp((t_j - c_j)/\tau_t)}, \quad \tau_t = 0.04
        $$
        
        Temperature acts as an **amplifier for logit gaps**. For any two dimensions, the probability ratio is:
        
        $$
        \frac{p_i}{p_j} = \exp\left(\frac{t_i - t_j}{\tau}\right)
        $$
        
        For a fixed gap $\Delta = t_i - t_j$, a smaller $\tau$ exponentially magnifies the ratio. For example, with $\Delta = 0.4$, $t - C = [0, 0, 0, 0.4]$ becomes $[0, 0, 0, 10]$ after dividing by $0.04$:
        
        $\tau = 1$: Ratio $e^{0.4} \approx 1.5$ (similar probabilities across dimensions).
        
        $\tau = 0.04$: Ratio $e^{10} \approx 22026$ (runner-up drops virtually to zero).
        
- **Across images, flat**: different images occupy different prototypes, so the average distribution stays near-uniform and the prototype space is well-utilized (ensured by centering, which subtracts chronically favored dimensions)
    - Centering
        
        $$
        C \leftarrow m C + (1-m)\frac{1}{B} \sum_{b=1}^B t(b), \quad m = 0.9
        $$
        
        where $C$ is the exponential moving average (EMA) of teacher outputs across batches.
        
        $$
        t_i \leftarrow t_i−c_i
        $$
        
        **Centering acts as a high-pass filter for logits.** $C$ tracks the teacher's average prediction bias across all images. For example, if the teacher constantly favors dimension 3, $C$ absorbs this pattern within tens of steps. For a collapsed output $t_i = [1.0, 1.0, 5.0, 1.0]$ after about 40 steps:
         
        
        $$
        t_i−C=[0,0,0,0] ⇒ softmax→uniform
        $$
        
        Betting on dimension 3 earns nothing. Only genuine per-image deviations survive and get amplified by sharpening.
        **Why EMA, not the batch mean:** The lag ( $\sim$tens of steps) filters by timescale—only ***chronic*** preferences (the signature of collapse) accumulate in $C$ and get removed; normal batch-to-batch fluctuations pass through.
        

**Now a concrete example.** Let K = 5 (5 prototypes), with 3 images in the batch: a cat, a dog, and a truck. The teacher outputs a 5-dim distribution for each image, and so does the student.

**情形一:无坍缩状态**

teacher的输出(每行一张图,已经过centering+sharpening):

```
        桶0    桶1    桶2    桶3    桶4
猫    [0.02,  0.90,  0.03,  0.03,  0.02]   ← 尖,指向桶1
狗    [0.03,  0.10,  0.82,  0.03,  0.02]   ← 尖,指向桶2
卡车  [0.85,  0.03,  0.02,  0.05,  0.05]   ← 尖,指向桶0
```

检查两个条件:

- **单张图的分布尖不尖?** 尖。每张图有明确归属(条件熵低)✓
- **三张图的平均分布平不平?** 平均 = [0.30, 0.34, 0.29, 0.04, 0.03],前三个桶被均衡使用(边缘熵较高)✓

此时student要把loss做低,**必须看图**:输入是猫就得输出接近[0.02, 0.90, ...],输入是卡车就得输出接近[0.85, 0.03, ...]。要做到"看到猫图→答桶1",它只能去学猫的视觉特征。监督信号有效。

**情形二:one-hot坍缩(sharpening单干,没有centering)**

teacher对**所有图**输出同一个尖峰:

```
猫    [0.01,  0.96,  0.01,  0.01,  0.01]
狗    [0.01,  0.96,  0.01,  0.01,  0.01]   ← 三行一模一样
卡车  [0.01,  0.96,  0.01,  0.01,  0.01]
```

- 单张图尖吗? 很尖 ✓
- 平均分布 = [0.01, 0.96, 0.01, 0.01, 0.01],极度失衡 ✗

student的必胜策略:把输出层bias第1维调大,**对任何输入都输出[0.01, 0.96, ...]**。三张图loss全部≈0,但"答桶1"这个行为和图片内容毫无关系——梯度不再要求encoder区分猫狗卡车,encoder退化成常数函数。

**情形三:均匀坍缩(centering单干,没有sharpening)**

teacher对**所有图**输出均匀分布:

```
猫    [0.20,  0.20,  0.20,  0.20,  0.20]
狗    [0.20,  0.20,  0.20,  0.20,  0.20]   ← 还是三行一模一样
卡车  [0.20,  0.20,  0.20,  0.20,  0.20]
```

- 平均分布 = [0.20 × 5],完美均衡 ✓
- 单张图尖吗?完全平,没有任何判断 ✗

student的必胜策略同样是无视输入、恒定输出均匀分布,loss到底,梯度无信息。

# 5 MAE

Let's then dive into the work from famous Masked Auto Encoder(He et al., 2021).

![image.png](Self%20supervised%20learning/image%2027.png)

![image.png](Self%20supervised%20learning/image%2028.png)

![image.png](Self%20supervised%20learning/image%2029.png)

![image.png](Self%20supervised%20learning/image%2030.png)

- The MSE (mean squared error loss) in the pixel space between the input image and the reconstructed image is adopted. (similar to reconstruction loss in Context Encoder)
- Loss is only computed for masked patches.

![image.png](Self%20supervised%20learning/image%2031.png)

**Some ablation studies in the paper:**

- **Masking sampling strategy**: among random, block-wise, and grid-wise sampling, **random masking works best** — combined with a high masking ratio, it removes low-level shortcuts and forces the model to rely on global semantics.
- **Masking ratio**: a high ratio (**~75%**) is optimal — unusually high compared to NLP (BERT masks ~15%).
- **Scaling / encoder size**: MAE scales well — a larger encoder (**ViT-B → ViT-L → ViT-H**) consistently improves performance, and the gains from scaling are more pronounced than with supervised pretraining. This is the core "**scalable vision learners**" claim.
- **Decoder design**: the decoder can be **lightweight** — its depth/width barely affects fine-tuning accuracy, so a small decoder suffices. This is a key source of MAE's efficiency (the heavy encoder only processes the visible 25%).
- **Reconstruction target**: using **per-patch normalized pixels** as the target works better than regressing raw pixels.(**raw pixels**: the output directly regresses the patch's original pixel values (baseline, works fine; **normalized pixels**: normalize the pixels within each patch (subtract mean / divide by std) first, then have the output regress these normalized values.)

# 6 Understanding Knowledge Distillation (KD)

<aside>

简单来说，知识蒸馏就像是“大模型（Teacher）手把手教小模型（Student）”，目的是让参数量小、运行快的小模型，学到大模型的超强能力。

这里将蒸馏划分为两类，核心区别在于**小模型到底向大模型“抄”什么**：

**1. Vanilla Distillation —— 抄“最终的概率分布”**

- **核心逻辑**：小模型去拟合（Match）大模型输出的**Softmax概率分布**。
- **什么是暗知识（Dark Knowledge）？**
    - 假设输入一张“哈士奇”的照片。传统的硬标签（Hard Label）只告诉你：`[狗: 1, 狼: 0, 猫: 0]`。
    - 但强大的 Teacher 模型会输出一个软标签（Soft Target）：`[狗: 0.7, 狼: 0.28, 猫: 0.02]`。
    - 这个概率分布承载了**类间暗知识**：它告诉小模型，**“虽然这是条狗，但它长得很像狼，不过绝对不是猫”**。小模型学到这个，会比只看标准答案聪明得多。
- **代表作**：Hinton 的经典 KD、我们正在看的 DINO。

**2. Generalized Distillation —— 抄“中间的解题思路”**

- **核心逻辑**：随着技术发展，“蒸馏”的概念被泛化了。现在不局限于只抄最后的概率分布，**只要是 Teacher 输出的任何有用信号，Student 都可以去跟它对齐（Match）**。
- **它抄什么？**
    - **Logits**：Softmax 之前的原始得分（没变成概率前的数值）。
    - **特征 (Features)**：神经网络中间层提取出来的特征图（Feature Map）或隐藏向量。
    - **关系 (Relations)**：不同样本在大模型特征空间里的结构关系。
- **代表作**：
    - **FitNets**：直接让小模型的中间层去拟合大模型的中间层（学你怎么画特征图）。
    - **BYOL**：自监督学习里的经典，让 Student 支路去预测 Teacher 支路产出的特征向量。
</aside>




