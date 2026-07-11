---
title: 'RL Series (2) - Proximal Policy Optimization(PPO)'
date:  2024-10-05
permalink: /posts/2024/10/blog-post-ppo/
tags:
  - Reinforcement Learning
  - RL
excerpt: |
  🐝  **TL;DR**

  Vanilla policy gradient is pure on-policy, **which suffers from** low sample efficiency because data cannot be reused. Importance sampling is introduced to enable off-policy gradient updates, unlocking data reuse. Moreover**, to** prevent the policy from shifting too drastically, PPO uses either **KL Penalty** (adaptive constraints) or **Clip** (direct ratio clipping) to bound the update size.
toc: true
---
<div class="notice--info" markdown="1">
🐝 TL;DR:

Vanilla policy gradient is pure on-policy, **which suffers from** low sample efficiency because data cannot be reused. Importance sampling is introduced to enable off-policy gradient updates, unlocking data reuse. Moreover**, to** prevent the policy from shifting too drastically, PPO uses either **KL Penalty** (adaptive constraints) or **Clip** (direct ratio clipping) to bound the update size.

</div>


# 1 On-policy v.s. Off-policy

Vallina policy gradient is fully on-policy, meaning that every gradient estimator
used samples drawn directly from the model being updated(i.e. we took 1 training step per inference batch, and training batch size was set equal to inference batch size).

Off-policy RL, which takes multiple training steps per inference batch and has the potential to speed up training, at the cost of potential instability and increased algorithmic complexity.

**From on-policy to off-policy**

To improve sample efficiency, we aim to freeze a behavior policy $\pi_{\theta'}$ (with $\theta'$ parameters) to collect data, and reuse this static dataset to update the target policy $\theta$ multiple times.

This introduces a **distribution mismatch**. The target objective is an expectation over $p_\theta$, but the available samples are drawn from $q = p_{\theta'}$. 

We correct for this mismatch by scaling the objective with an importance weight ratio —— Apply **Importance Sampling.**

# 2 Importance Sampling

用 $q$ 的样本估 $p$ 的期望,只要每个样本乘上 weight $\frac{p(x)}{q(x)}$ 即可。**期望上严格相等 → 估计 unbiased。**

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Proximal%20Policy%20Optimization%20(PPO)/image.png" alt="Importance Sampling">
</figure>

期望相等,**但方差不等**: 对比 $\mathrm{Var}_{x\sim p}[f(x)]$,差别就在第一项多了一个 $\frac{p(x)}{q(x)}$ 因子。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Proximal%20Policy%20Optimization%20(PPO)/image%201.png" alt="Variance of Importance Sampling">
</figure>

While Importance Sampling is **unbiased asymptotically**, it suffers from **high variance under finite sample sizes**. If the discrepancy between $p$ and $q$ is large, the IS weights can explode, leading to noisy gradients and unstable training. To mitigate this, a hard constraint is required to ensure $\pi_\theta$ does not drift too far from $\pi_{\theta'}$. This serves as the fundamental motivation for PPO, which utilizes **KL penalties or objective clipping** to restrict the policy ratio around 1.

# 3 Gradient Update for Off-policy

Let's examine the gradient update equation **in** the off-policy setting.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Proximal%20Policy%20Optimization%20(PPO)/image%202.png" alt="Gradient update in off-policy setting">
</figure>

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Proximal%20Policy%20Optimization%20(PPO)/c1383cdd-fab0-4766-8b57-12ac756c8c66.png" alt="Off-policy objective">
</figure>

Here, $f(x) = p_{\theta}(a_t|s_t)$ :

$$
\begin{aligned}
\nabla_\theta J &= \mathbb{E}_{\pi_{\theta'}} \left[ \frac{A^{\theta'}}{p_{\theta'}} \nabla_\theta p_\theta \right] \\
&= \mathbb{E}_{\pi_{\theta'}} \left[ \frac{A^{\theta'}}{p_{\theta'}} p_\theta \nabla_\theta \log p_\theta \right] \\
&= \mathbb{E}_{\pi_{\theta'}} \left[ \frac{p_\theta}{p_{\theta'}} A^{\theta'} \nabla_\theta \log p_\theta \right]
\end{aligned}
$$

Again, The **validity** of $J^{θ^{′}}(θ)$  strictly depends on  **$\theta$** remaining close to **$\theta'$**.

At what point **$\theta$** has **drifted too far**, signaling that we must stop and resample data?

# 4 PPO

**答案就是 PPO(2017)。** PPO 做的就是把这个"什么时候停"从"凭感觉"变成"写进目标函数里强制约束":

## 4.1 **PPO-penalty**

在目标函数中**减去** $\beta\,\mathrm{KL}(\pi_{\theta'}\|\pi_\theta)$。新旧策略 $\pi_{\theta'}$与 $\pi_{\theta}$ 偏离越多，惩罚越大。
*注：KL 约束施加于**行为（输出分布）***而非参数。*

If $\text{KL}(\pi_{\theta^k} \parallel \pi_{\theta}) > \text{KL}_{max}$, increase $\beta$ （策略偏离过大，加大惩罚）

If $\text{KL}(\pi_{\theta^k} \parallel \pi_{\theta}) < \text{KL}_{min}$, decrease $\beta$ （策略过于保守，减小惩罚）

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Proximal%20Policy%20Optimization%20(PPO)/image%203.png" alt="PPO 是 TRPO (2015) 的简化版——它将 TRPO 难以求解的硬约束，松弛为更易优化的惩罚项 (Penalty)。">
  <figcaption style="text-align: center;">PPO 是 TRPO (2015) 的简化版——它将 TRPO 难以求解的**硬约束**，松弛为更易优化的**惩罚项 (Penalty)。**</figcaption>
</figure>

## 4.2 PPO-clip

**实际工程中最常用的主流版本**。直接对新旧策略的比率（Ratio）进行裁剪，从而彻底替代了复杂的 KL 惩罚。通过 CLIP 将 weight  $\frac{p_\theta(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}$强行限制在 $[1-\varepsilon, 1+\varepsilon]$ 之间（通常 $\varepsilon = 0.2$），再与未裁剪的目标取 `min`。这样避免了复杂的自适应 KL 计算，实现更简单，训练更稳定高效。

$$
J_{PPO2}^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \min \left( \frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)} A^{\theta^k}(s_t, a_t), \, \text{clip}\left(\frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)}, 1 - \varepsilon, 1 + \varepsilon\right) A^{\theta^k}(s_t, a_t) \right)
$$