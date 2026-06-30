---
title: 'RL Series (1) - Policy Gradient'
date:  2024-10-03
permalink: /posts/2024/10/blog-post-policy-gradient/
tags:
  - Reinforcement Learning
  - RL
excerpt: |
  🔥 **TL;DR**

  1. Concepts: Actor,Env and Reward. 
  2. Policy Gradients
toc: true
---
<div class="notice--info" markdown="1">
🔥 **TL;DR**

1. Concepts: Actor,Env and Reward. 
2. Policy Gradients 

</div>

# 1 Basic Components

There are three basic components here: **Actor, Environment, and Reward**.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image.png" alt="Figure 1 Three basic components in RL">
  <figcaption style="text-align: center;">Figure 1 Three basic components in RL</figcaption>
</figure>

## 1.1 Policy of Actor

Inside the actor, there is a **policy** that decides its behavior.

$$
𝑎_𝑡 ∼ 𝜋_𝜃(⋅ | 𝑠_𝑡)
$$

Depending on the observability of the environment, there are two classic RL settings:

- **Fully Observable (MDP, Markov Decision Process) — $o_t = s_t$**: We consider a policy that takes in the current state $s_t$ and outputs an action $a_t$.
- **Partially Observable (POMDP,Partially Observable MDP) — $o_t \neq s_t$**: We consider a policy that takes in the entire history $h_t = (o_1, o_2, \dots, o_t)$ and outputs an action $a_t$.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%201.png" alt="Figure 2 Policy of Actor">
  <figcaption style="text-align: center;">Figure 2 Policy of Actor</figcaption>
</figure>

Take Figure 2 as an example, sample action based on softmax problibity, which mea 具体执行哪个,是再从这组概率里随机采样出来的。

## 1.2 Trajectory

The series of states and actions then forms a (finite horizon) **trajectory**(i.e. episodes or rollouts):

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%202.png" alt="Figure 3: Definition of Trajectory">
  <figcaption style="text-align: center;">Figure 3: Definition of Trajectory</figcaption>
</figure>

where $\tau$ terminates at time step $T$ either when the agent reaches a **terminal state** (e.g., goal achieved, game over) or when the interaction hits a predefined **maximum time horizon** $T$.

The probability of a trajectory $\tau$ occurring is：

$$
p_\theta(\tau) = p(s_1)p_\theta(a_1|s_1)p(s_2|s_1,a_1)p_\theta(a_2|s_2)p(s_3|s_2,a_2)\cdots = p(s_1)\prod_{t=1}^T p_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

From the above equation, we can see the environment $p(s_{t+1}|s_t,a_t)$ is uncontrollable, whereas the actor is under our control; thus, changing the actor changes the trajectory distribution.

## 1.3 Maximize Expected Reward

Given a reward function,  the immediate reward $r_t$ depends on $s_t$ and $a_t$:

$$
r_t = r(s_t,a_t)
$$

Each trajectory $\tau$ corresponds to a total reward **$R(\tau)$:** 

$$
R(\tau) = \sum_{t=1}^Tr_t
$$

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%203.png" alt="Figure 4:Expected Rewards">
  <figcaption style="text-align: center;">Figure 4:Expected Rewards</figcaption>
</figure>

Since different actors (parameterized by $\theta$) yield different trajectory distributions, our goal is to find an optimal actor $\theta$ that maximizes the probability of high-reward ("good") trajectories while minimizing low-reward(”bad”) ones. 

$$
\bar{R}_\theta = \sum_{\tau} R(\tau)p_\theta(\tau) =\mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)] 
$$

# 2 Policy Gradient

The objective of the agent is to maximize the expected return $\bar{R}_\theta$. To this end, we compute the gradient of $\bar{R}_\theta$: $\nabla \bar{R}_\theta = ？$

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%204.png" alt="Figure 5: $T_n$ represents the length (total time steps) of the $n$-th trajectory/episode.">
  <figcaption style="text-align: center;">Figure 5: $T_n$ represents the length (total time steps) of the $n$-th trajectory/episode.</figcaption>
</figure>

Note that, $p_\theta$ is same with $\pi_\theta$ which both means the probability of choosing action a under state s produced by the network with parameters $\theta$ through a softmax.

Part deviation of above equation:

$$
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T \log p_\theta(a_t|s_t) + \sum_{t=1}^T \log p(s_{t+1}|s_t, a_t) \\ \nabla \log p_\theta(\tau) = \nabla \log p(s_1) + \sum_{t=1}^T \nabla \log p_\theta(a_t|s_t) + \sum_{t=1}^T \nabla \log p(s_{t+1}|s_t, a_t) \\ since \ \nabla \log p(s_1) = 0 ,\ \nabla \log p(s_{t+1}|s_t, a_t) = 0 \\
so, \nabla \log p_\theta(\tau) = \sum_{t=1}^T \nabla \log p_\theta(a_t|s_t)

$$

Intuition here:

In a sampled trajectory $\tau$, if taking action $a_t$ at at state $s_t$ leads to a positive reward for the whole

$\tau$, then increase the probability of this action; otherwise, decrease it.

## 2.1 Implementation

Policy Gradient is essentially a supervised classification problem that "weights samples by their performance.”

### 2.1.1  Data Collection

Run N games (trajectories) using the current policy $\pi_\theta$, recording at every step:

- All state-action pairs: $(s_t, a_t)$
- The return of the **whole** trajectory: $R(\tau^n)$

### 2.1.2 Constructing the Objective

See the task as a classification problem (input state $s_t^n$, with the actual action $a_t^n$ as the one-hot label). Compared to ordinary cross-entropy, it multiplies by an extra weight $R(\tau^n)$. The goal is to maximize the objective function:

$$
\max_\theta \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^T R(\tau^n) \log p_\theta(a_t^n | s_t^n)
$$

### 2.1.3 Gradient Update

- **Autodiff**: Leave it to TensorFlow or PyTorch to handle automatically.
- **Gradient ascent**: Use gradient ascent (the parameter-update sign is `+`), because the core goal is to maximize the expected return.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%205.png" alt="Figure 6 Implementation of Naive Policy Gradient (On policy)">
  <figcaption style="text-align: center;">Figure 6 Implementation of Naive Policy Gradient (On policy)</figcaption>
</figure>

与普通监督分类的对比

| **特性** | **普通监督分类** | **Policy Gradient (策略梯度)** |
| --- | --- | --- |
| **样本权重** | 恒为 1（盲目模仿标签） | 乘以轨迹总回报 $R(\tau^n)$ |
| **学习机制** | 每一个样本同等重要 | * **高回报**动作：**加重学习**
* **低/负回报**动作：**削弱或反向学习** |

# 3 Tips

Instead of focusing on equations, let's explore why these techniques are necessary in practice.

## 3.1 Tip1: Add a Baseline

**Problem**:$R(\tau^n)$ 可能恒为正,于是

- 被采样到的动作 → 权重为正 → 概率被推高
- 没被采样到的动作 → 因为归一化(总和=1)→ 被动下降

后果:概率涨不涨取决于"有没有被采到",而非动作好不好。

**Solution**:减去 baseline,权重  $R(\tau^n)$to $R(\tau^n)-b$ (常取 $b\approx\mathbb{E}[R]$)。

- 比平均好(R>b)→ 正权重 → 推高
- 比平均差(R<b)→ 负权重 → 压低

判断标准从"有没有被采到"变成"比平均好还是差"。且只要 b 不依赖动作,此改动无偏,只降方差。

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%206.png" alt="Figure 7 Add a Baseline">
  <figcaption style="text-align: center;">Figure 7 Add a Baseline</figcaption>
</figure>

### 3.1.1 Example

Suppose we have one state, three actions(a, b and c). 

> Start with near-equal probabilities to focus on the rewards. 
This is a single-step trajectory, so $R(\tau)$= the reward of that sampled action.
> 

| Action | Prob | Reward |
| --- | --- | --- |
| a | 0.33 | **3**(best) |
| b | 0.33 | 2 |
| c | 0.34 | 1(worst) |

**Update rule**: when action $a_t$ is sampled with weight $R(\tau)$, the push on each action's probability is $R(\tau)\cdot\big(\mathbb{1}[i=a_t]-\pi(i)\big)$, where $\frac{∂logπ(a_t​)}{∂z_i}​=1[i=a_t​]​​−π(i)​​$.

---

**🌰 Before baseline:**

Suppose we sampled b, so weight = $R(b) = 2$ (positive).

| Action | Computation | Direction |
| --- | --- | --- |
| a | $2\cdot(0-0.33)=-0.66$ | ↓ |
| b(sampled) | $2\cdot(1-0.33)=+1.34$ | ↑ |
| c | $2\cdot(0-0.34)=-0.68$ | ↓ |

**Result** : b is pushed up; both a and c are pushed down since they‘re finally as inputs of softmax.

**Problem**: a is actually the best action ($R=3$), yet it gets pushed down just because it wasn't sampled this round. This is the flaw of always-positive $R$ — not being sampled means getting suppressed, regardless of how good the action is.

---

**🌰 After baseline:**

Weight becomes $R-b$, where $\frac{3+2+1}{3}=2$.  

| Action | $R-b$ |
| --- | --- |
| a | $3-2=+1$ |
| b | $2-2= 0$ |
| c | $1-2=-1$ |

Still sampled b, but now b's weight $=0$: 

| Action | Computation | Direction |
| --- | --- | --- |
| a | $0\cdot(0-0.33)=0$ | -(no change) |
| b(采到) | $0\cdot(1-0.33)=0$ | - |
| c | $0\cdot(1-0.34)=0$ | - |

**Result**: b's weight is 0 → this step barely moves any probability. Since b is exactly an "average" action, it deserves neither reinforcement nor suppression — the baseline correctly identifies "b is neither good nor bad, don't touch it."

(If c were sampled instead: weight =−1, so c would be actively pushed down and a would passively rise — the bad action yields room to the good one.)

---

**🌰 Side-by-side comparison**

|  | What happens when b is sampled |
| --- | --- |
| **No baseline** (b=0, weight=2) | b↑, a↓, c↓ — the best action a gets unfairly pushed down |
| **With baseline** (b=2, weight=0) | nothing moves — correctly recognizes b as average, no needless pushing |

**Core idea** : Without a baseline, any sampled action with $R>0$ gets blindly pushed up, dragging down the unsampled ones (including the better action a). 

With a baseline, the weight $R-b$ measures "better or worse than average" — good actions get positive weight and rise, bad actions get negative weight and fall, average actions get weight ≈0 and stay put. Only then does probability flow correctly according to quality.

### 3.1.2 Evolution of the Baseline: from PG to PPO to GRPO

The core idea stays the same throughout: **replace $R$ with $R-b$ to measure "better or worse than average."**

The three generations of methods just improve how the baseline $b$ is obtained: 

| Year | Method | How baseline b is obtained | Advantage | Cost |
| --- | --- | --- | --- | --- |
| 1992 | Average baseline | A single **scalar** mean over all samples | Simple, reduces variance | Coarse, **state-independent** |
| 2017 | PPO | Learn a **value network** $V_\phi(s)$ to estimate the average level per state | **Per-state** fine-grained baseline | Requires training an extra **critic** as large as the policy — expensive and hard to tune |
| 2024 | GRPO | Sample a **group** of responses for the same prompt, use the **group mean** as that prompt's baseline | **Per-prompt** fine-grained, and **no critic needed** | Requires sampling several responses per prompt |

## 3.2 Tip2: Assign Suitable Credits

Within the same trajectory, every $(s_t, a_t)$ pair is weighted by the same reward, $R(\tau)$.

But this is clearly unfair, because within one trajectory some actions may be good while others are not. Even if the final game outcome is good, it doesn't mean every action was right. So we want to give each different action its own different weight, one that truly reflects whether that particular action was good or bad.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%207.png" alt="Figure 8 (Reward to go)For each pair at time $t$, it is only weighted by the rewards that come after $t$.">
  <figcaption style="text-align: center;">Figure 8 (Reward to go)For each pair at time $t$, it is only weighted by the rewards that come after $t$.</figcaption>
</figure>

Next we need to add a discount factor. Why? An action taken at time $t$ may get credit for all the rewards that follow it — but in more realistic settings, the longer the time lag, the smaller its influence.

<figure class="align-center" style="max-width: 680px;">
  <img src="/images/Policy%20Gradient/image%208.png" alt="Figure 9 Advantage function">
  <figcaption style="text-align: center;">Figure 9 Advantage function</figcaption>
</figure>

This $A^\theta(s_t,a_t)$ is exactly the shared core of both PPO and GRPO: PPO uses $V_\phi(s)$ as the baseline, while GRPO uses the group mean as the baseline — but the "reward − baseline" structure is identical in both.
