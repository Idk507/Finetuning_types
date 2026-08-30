# Full Fine-Tuning (FFT): A Complete Technical Reference

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [FFT vs SFT vs PEFT: Positioning in the Training Taxonomy](#2-fft-vs-sft-vs-peft-positioning-in-the-training-taxonomy)
3. [Mathematical Foundations of Full Fine-Tuning](#3-mathematical-foundations-of-full-fine-tuning)
4. [The Transformer Parameter Space: What Gets Updated](#4-the-transformer-parameter-space-what-gets-updated)
5. [Forward Pass and Loss Computation in FFT](#5-forward-pass-and-loss-computation-in-fft)
6. [Backpropagation Through the Full Network](#6-backpropagation-through-the-full-network)
7. [Gradient Flow: Layer-by-Layer Analysis](#7-gradient-flow-layer-by-layer-analysis)
8. [The Optimizer: AdamW Mathematics in Full Depth](#8-the-optimizer-adamw-mathematics-in-full-depth)
9. [Learning Rate Scheduling and Its Effect on Parameter Landscape](#9-learning-rate-scheduling-and-its-effect-on-parameter-landscape)
10. [Memory Architecture of Full Fine-Tuning](#10-memory-architecture-of-full-fine-tuning)
11. [Catastrophic Forgetting: Theory, Detection, and Mitigation](#11-catastrophic-forgetting-theory-detection-and-mitigation)
12. [Mixed Precision Training: FP32, BF16, and FP16](#12-mixed-precision-training-fp32-bf16-and-fp16)
13. [Gradient Checkpointing: Trading Compute for Memory](#13-gradient-checkpointing-trading-compute-for-memory)
14. [Gradient Accumulation: Simulating Large Batches](#14-gradient-accumulation-simulating-large-batches)
15. [The Full Fine-Tuning Pipeline: Step-by-Step](#15-the-full-fine-tuning-pipeline-step-by-step)
16. [FFT with a Small Language Model: Practical Walkthrough](#16-fft-with-a-small-language-model-practical-walkthrough)
17. [Hyperparameter Sensitivity and Tuning Strategy](#17-hyperparameter-sensitivity-and-tuning-strategy)
18. [Evaluation: Metrics, Benchmarks, and Failure Modes](#18-evaluation-metrics-benchmarks-and-failure-modes)
19. [Production Considerations](#19-production-considerations)
20. [Trade-offs, Best Practices, and Recommendations](#20-trade-offs-best-practices-and-recommendations)

---

## 1. Introduction and Motivation

### What is Full Fine-Tuning?

Full Fine-Tuning (FFT) is the process of updating every single learnable parameter in a pre-trained neural network during the fine-tuning phase. Unlike Parameter-Efficient Fine-Tuning methods (LoRA, prefix tuning, adapter layers) which freeze the base model and train only a small number of additional parameters, FFT subjects the entire parameter space — every weight matrix, every bias vector, every embedding, every layer normalization scale and shift — to gradient-based optimization driven by the task-specific training signal.

When you perform FFT, the optimizer maintains state for all parameters simultaneously. Every token in every training batch produces gradients that flow backward through every layer of the network, updating every weight according to those gradients. The pre-trained weights are not frozen, not regularized toward their original values (unless you explicitly add such a term), and not protected in any way. They are simply updated with each optimizer step, just as they were during pre-training — only now the training signal comes from your curated task-specific dataset rather than a massive unsupervised corpus.

### Why Full Fine-Tuning Matters

FFT is the highest-capacity adaptation method available. Because every parameter is free to move, the model can make arbitrarily large adjustments to its representations, attention patterns, and output distributions. This ceiling on adaptation quality is the primary reason to use FFT over PEFT: when PEFT methods cannot achieve the required task performance, FFT will. Research consistently shows that on complex tasks requiring deep behavioral changes — domain-specific generation styles, specialized reasoning patterns, format adherence under adversarial prompting — FFT outperforms LoRA by meaningful margins.

FFT is also the training regime used in the original pre-training of language models and in the first stage of the RLHF pipeline that produced ChatGPT, Claude, and Gemini. The InstructGPT paper (Ouyang et al., 2022) used full fine-tuning for its supervised learning stage on human demonstrations. Understanding FFT is therefore not just academically important — it is the foundation on which all modern aligned language model training is built.

### The Core Challenge: Scale vs. Quality

The central tension in FFT is that the methods producing the highest quality fine-tuned models are the most computationally expensive. Fine-tuning a 7B parameter model with full precision AdamW requires approximately:

```
Model weights (bf16):         7B * 2 bytes  =  14 GB
Optimizer first moment (fp32): 7B * 4 bytes =  28 GB
Optimizer second moment (fp32):7B * 4 bytes =  28 GB
Gradients (fp32):              7B * 4 bytes =  28 GB
Activations (variable):                     ~  8-20 GB
Total:                                       ~98-110 GB
```

This exceeds the memory of a single A100 (80GB), requiring model parallelism, optimizer state sharding, or the use of mixed-precision and memory-reduction techniques. Understanding exactly where this memory goes — and how each optimization technique reduces it — is a prerequisite for productionizing FFT.

---

## 2. FFT vs SFT vs PEFT: Positioning in the Training Taxonomy

The term "Supervised Fine-Tuning" (SFT) describes the objective and dataset type: supervised cross-entropy on labeled (prompt, completion) pairs. SFT says nothing about which parameters are updated. The term "Full Fine-Tuning" describes the scope of parameter updates: all of them. These two dimensions are independent, which generates the following training taxonomy:

| Method | Objective | Parameters Updated | Memory Cost | Peak Quality |
|---|---|---|---|---|
| Pre-training | Self-supervised LM | All (from random init) | Maximum | N/A (defines the baseline) |
| Full Fine-Tuning (FFT) | Supervised / RLHF | All | Very High | Highest |
| LoRA Fine-Tuning | Supervised | ~0.1-2% (adapters only) | Low | High (95%+ of FFT) |
| Prefix Tuning | Supervised | ~0.01% (soft prefixes) | Very Low | Moderate |
| Adapter Tuning | Supervised | ~1-5% (bottleneck FFNs) | Low | High |
| Prompt Tuning | Supervised | ~0.001% (input embeddings) | Minimal | Moderate |
| In-Context Learning | None (no gradient) | 0% | Zero | Low to Moderate |

Full fine-tuning occupies the "highest cost, highest ceiling" position. The practical decision of whether to use FFT over LoRA requires answering: "Does the performance gap between FFT and LoRA on my target task justify the additional hardware cost and engineering complexity?"

---

## 3. Mathematical Foundations of Full Fine-Tuning

### 3.1 The Optimization Problem

Let the model be parameterized by the full parameter vector `theta in R^d` where `d` is the total number of parameters (e.g., d = 117,000,000 for GPT-2 small, d = 7,000,000,000 for LLaMA-2-7B). Full fine-tuning seeks:

```
theta* = argmin_{theta} L(theta; D_FT)
```

where `L` is the task-specific loss (cross-entropy for language modeling) and `D_FT` is the fine-tuning dataset. Critically, theta starts at `theta_pretrained` — the checkpoint from pre-training — not at a random initialization. This is the fundamental distinction from training from scratch.

The optimization landscape around `theta_pretrained` is not random: pre-training has placed the model in a region of parameter space where it already produces coherent language. Fine-tuning navigates the loss landscape of `L(theta; D_FT)` from this warm starting point, which is why fine-tuning converges in orders of magnitude fewer steps than pre-training.

### 3.2 The Fine-Tuning Loss

For a supervised fine-tuning dataset `D_FT = {(p_i, c_i)}_{i=1}^N` where `p_i` are prompts and `c_i` are desired completions, the fine-tuning loss is:

```
L(theta) = -(1/N) * sum_{i=1}^N (1/|c_i|) * sum_{t=1}^{|c_i|} log P_theta(c_t^i | p_i, c_1^i, ..., c_{t-1}^i)
```

The gradient of this loss with respect to the full parameter vector theta is:

```
nabla_theta L = -(1/N) * sum_{i=1}^N (1/|c_i|) * sum_{t=1}^{|c_i|} nabla_theta log P_theta(c_t^i | context_t^i)
```

In full fine-tuning, this gradient is non-zero for every parameter in the network. Every weight matrix, every bias, every embedding vector, every layer norm parameter receives a gradient signal at every step. The computational cost of computing this gradient is dominated by the backward pass through all L transformer layers.

### 3.3 The Jacobian of the Full Network

For a network with L layers where layer l computes `h_l = f_l(h_{l-1}; W_l)`, the gradient of the loss with respect to the parameters of layer l is:

```
dL/dW_l = (dL/dh_L) * prod_{k=l+1}^{L} (dh_k/dh_{k-1}) * (dh_l/dW_l)
```

The product of Jacobians `prod_{k=l+1}^{L} (dh_k/dh_{k-1})` is the "gradient highway" from the loss back to layer l. In full fine-tuning, this product must be computed for every layer. The computational cost grows with depth: deeper layers receive gradients that have been multiplied through more Jacobians, making them more susceptible to vanishing or exploding gradient pathologies in poorly conditioned architectures.

### 3.4 Relationship to Transfer Learning

Full fine-tuning is a form of transfer learning: knowledge from pre-training is transferred to the fine-tuning task through the initialization `theta_0 = theta_pretrained`. The pre-trained representations serve as an informed prior over the parameter space. Empirically, starting from `theta_pretrained` rather than random initialization reduces the fine-tuning steps required by 10x-100x and substantially improves the final task performance, especially for small fine-tuning datasets where starting from scratch would severely overfit.

The information-theoretic view: the pre-trained model has compressed enormous amounts of linguistic knowledge into its parameters. Fine-tuning accesses this compressed knowledge and redirects it toward a specific task distribution, rather than having to re-learn language from scratch.

---

## 4. The Transformer Parameter Space: What Gets Updated

For a decoder-only transformer with `L` layers, hidden dimension `d`, intermediate FFN dimension `d_ff = 4d`, and vocabulary size `|V|`, the complete parameter inventory that gets updated in FFT is:

### 4.1 Token Embedding Table

```
W_embed in R^{|V| x d}
Parameters: |V| * d
Example (GPT-2 small): 50,257 * 768 = 38,597,376
```

This table maps each token ID to a d-dimensional embedding vector. In full fine-tuning, these embeddings shift to better represent the tokens relevant to the fine-tuning task. Tokens that appear frequently in the fine-tuning data will receive larger gradient updates than rare tokens.

### 4.2 Attention Projection Matrices (per layer)

```
W_Q in R^{d x d}  — Query projection
W_K in R^{d x d}  — Key projection
W_V in R^{d x d}  — Value projection
W_O in R^{d x d}  — Output projection
Parameters per layer: 4 * d^2
Example (GPT-2 small, per layer): 4 * 768^2 = 2,359,296
```

In GPT-2's implementation, W_Q, W_K, W_V are fused into a single `c_attn` matrix of shape `(d, 3d)`, and W_O is the `c_proj` matrix of shape `(d, d)`. All of these receive gradient updates in FFT.

For models with multi-head attention (H heads), the projections split the d-dimensional space into H subspaces of dimension `d/H` each. The gradient flows through all heads simultaneously, allowing different heads to specialize toward different fine-tuning objectives.

### 4.3 Feed-Forward Network Matrices (per layer)

```
W_1 in R^{d x d_ff}  — Up-projection
b_1 in R^{d_ff}       — Bias
W_2 in R^{d_ff x d}  — Down-projection
b_2 in R^d            — Bias
Parameters per layer: 2 * d * d_ff + d_ff + d = 2 * d * 4d + 5d = 8d^2 + 5d ≈ 8d^2
Example (GPT-2 small, per layer): 2 * 768 * 3072 + 3072 + 768 = 4,721,664 + 3,840 ≈ 4.7M
```

The FFN layers contain the majority of the parameters in most transformer architectures (approximately 2/3 of total parameters for d_ff = 4d). In FFT, the W_1 and W_2 matrices are fully updated, allowing the model to learn entirely new feature transformations beyond those learned during pre-training.

### 4.4 Layer Normalization Parameters (per layer)

```
gamma in R^d  — Scale parameter (multiplicative)
beta  in R^d  — Shift parameter (additive)
Parameters per layer: 2 * d  (per LayerNorm; typically 2 LayerNorms per transformer block)
Example (GPT-2 small, per layer): 2 * 2 * 768 = 3,072
```

Layer normalization parameters are small in count but highly influential in practice. They control the scale and offset of the normalized activations entering each sublayer. Fine-tuning these parameters allows the model to adjust its internal activation distributions to better match the fine-tuning task statistics.

### 4.5 Final Language Model Head

```
W_lm_head in R^{d x |V|}  (or tied to W_embed if using weight tying)
Parameters: d * |V| (if not tied)
```

The language model head projects the final hidden state to vocabulary logits. In many architectures (including GPT-2), this matrix is weight-tied to the embedding table, meaning `W_lm_head = W_embed^T`. In this case, fine-tuning the embedding table simultaneously fine-tunes the language model head.

### 4.6 Total Parameter Count

```
Total = |V|*d + L*(4*d^2 + 8*d^2 + 4*d) + d
      = |V|*d + L*(12*d^2 + 4*d) + d
      ≈ |V|*d + 12*L*d^2  (dominant terms for large d)
```

For GPT-2 small: `50257*768 + 12*12*768^2 ≈ 38.6M + 85.5M ≈ 124M`
For LLaMA-2-7B: `32000*4096 + 12*32*4096^2 ≈ 131M + 6,442M ≈ 6.57B` (approximate)

---

## 5. Forward Pass and Loss Computation in FFT

The forward pass in full fine-tuning is identical to inference, with one critical difference: activations are retained in memory for use in the backward pass. This doubles (or more) the memory requirement compared to inference.

### 5.1 The Forward Pass Through One Transformer Block

Input `X` of shape `[B, T, d]` where B=batch size, T=sequence length, d=hidden dimension:

**Step 1: Pre-LayerNorm**
```
X_norm = LayerNorm(X)   =   (X - mean(X)) / (std(X) + eps) * gamma + beta
```
The mean and variance are computed over the d-dimension for each (batch, position) pair. The retained activations include `X` (for residual), `mean(X)`, and `std(X)` (needed for backprop through LayerNorm).

**Step 2: Multi-Head Causal Self-Attention**
```
Q = X_norm @ W_Q,  K = X_norm @ W_K,  V = X_norm @ W_V
S = Q @ K^T / sqrt(d/H) + CausalMask    (scores, shape [B, H, T, T])
A = softmax(S)                            (attention weights, shape [B, H, T, T])
head_out = A @ V                          (context, shape [B, H, T, d/H])
attn_out = Concat(head_out) @ W_O        (shape [B, T, d])
```

The attention matrix `A` is `O(T^2)` in memory. For T=2048, H=12, B=4, in float32: `4 * 12 * 2048 * 2048 * 4 bytes = 805 MB`. This is the dominant activation memory cost and the primary motivation for Flash Attention.

**Step 3: First Residual Connection**
```
X = X + attn_out
```

**Step 4: Pre-LayerNorm (second)**
```
X_norm2 = LayerNorm(X)
```

**Step 5: Feed-Forward Network**
```
h = activation(X_norm2 @ W_1 + b_1)    (shape [B, T, d_ff], d_ff = 4*d)
ffn_out = h @ W_2 + b_2                 (shape [B, T, d])
```

**Step 6: Second Residual Connection**
```
X = X + ffn_out
```

The total activations retained per transformer block (for backprop) include: input `X`, normalized inputs, Q, K, V tensors, the full attention weight matrix `A`, the FFN intermediate activation `h`, and the residual connections. The memory cost per block grows linearly with batch size and sequence length, and is the primary reason why FFT of large models requires gradient checkpointing.

### 5.2 Loss Computation with Loss Masking

After the final transformer block, the hidden state passes through the language model head:

```
logits = final_hidden_state @ W_lm_head    (shape [B, T, |V|])
```

The loss is computed with the autoregressive shift and loss masking:

```
shift_logits = logits[:, :-1, :]     (shape [B, T-1, |V|])
shift_labels = labels[:, 1:]         (shape [B, T-1])
loss = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)
```

The cross-entropy at position t for the correct token v*:
```
l_t = -logits_t[v*] + log(sum_v exp(logits_t[v]))
    = -logits_t[v*] + LogSumExp(logits_t)
```

The LogSumExp is computed in a numerically stable way to prevent overflow from large logit values.

---

## 6. Backpropagation Through the Full Network

### 6.1 Chain Rule Applied to the Full Parameter Space

The backward pass computes `dL/dtheta` for every parameter in theta by applying the chain rule from the loss all the way back to the first layer. For a parameter `W_l` in layer l:

```
dL/dW_l = dL/dh_L * (dh_L/dh_{L-1}) * ... * (dh_{l+1}/dh_l) * (dh_l/dW_l)
```

where `h_l` is the output of layer l. In PyTorch, this is computed automatically by the autograd engine, which traverses the computation graph in reverse order. The graph was constructed during the forward pass by tracking all operations on tensors with `requires_grad=True`.

In full fine-tuning, every tensor in the computation graph has `requires_grad=True`, so the autograd engine must compute and store gradients for every node. This is computationally equivalent to re-running the forward pass in reverse, with the additional cost of computing Jacobians at each operation.

### 6.2 Gradient of Cross-Entropy Loss

The gradient of the loss with respect to the logits at position t is the softmax residual:

```
dL/d(logits_t) = softmax(logits_t) - one_hot(v_t*)
               = p_t - e_{v_t*}
```

where `p_t` is the model's predicted probability distribution and `e_{v_t*}` is the one-hot vector for the true token. This gradient vector has a beautiful geometric interpretation: it points from the true token's probability to the current model distribution. When the model is confident and correct (p_t[v*] ≈ 1), the gradient is near zero. When the model is wrong or uncertain, the gradient is large and points in the direction of increasing probability for the correct token while decreasing probability for all incorrect tokens.

### 6.3 Gradient Through the Language Model Head

```
d_loss/d(W_lm_head) = final_hidden^T @ (p_t - e_{v_t*})
d_loss/d(final_hidden) = (p_t - e_{v_t*}) @ W_lm_head^T
```

The gradient flows from the language model head backward into the final transformer layer's hidden state, and from there backward through all L transformer blocks.

### 6.4 Gradient Through a Transformer Block

For a single transformer block with residual connections, the gradient of the loss with respect to the block's input `X` is:

```
dL/dX = dL/dX_out * (dX_out/dX) 
       = dL/dX_out * (I + d(attn_branch)/dX + d(ffn_branch)/dX)
```

The identity matrix `I` in this expression is the gradient contribution of the residual connection. It ensures that even if the attention and FFN branches have small or zero gradients (which can happen during early fine-tuning), the gradient signal is still transmitted to earlier layers. This is the mathematical reason residual connections solve the vanishing gradient problem.

---

## 7. Gradient Flow: Layer-by-Layer Analysis

### 7.1 Gradient Magnitude Decay in Deep Networks

Without residual connections, the gradient magnitude at layer l would be:

```
||dL/dX_l|| ≈ ||dL/dX_L|| * prod_{k=l+1}^{L} ||Jacobian_k||
```

If each Jacobian has spectral norm slightly less than 1, the product decays exponentially with depth. For L=32 layers and average spectral norm 0.9: `0.9^32 ≈ 0.034`. The gradient at the first layer would be 30x smaller than at the last layer, causing the first layer to train 30x slower — a severe imbalance.

With residual connections, the gradient at each layer has a minimum magnitude of `||dL/dX_L||` (from the identity term), preventing this decay. The actual gradient at layer l is the sum of direct contributions and contributions through sublayers.

### 7.2 Attention Gradient: Softmax Jacobian

The gradient through the softmax operation is:

```
d(softmax(s)_i)/d(s_j) = softmax(s)_i * (delta_{ij} - softmax(s)_j)
                       = p_i * (delta_{ij} - p_j)
```

The Jacobian of softmax has eigenvalues in [0, 1]: the minimum is 0 (when the softmax is perfectly peaked) and the maximum approaches 1 (for uniform distributions). When the attention distribution is peaked (the model is highly selective), gradients through the attention weights are small. This can slow down fine-tuning on tasks requiring different attention patterns than those learned during pre-training.

### 7.3 FFN Gradient: Through the Activation Function

For the GELU activation (used in GPT-2) at position i:

```
GELU(x) = x * Phi(x)  where Phi is the standard normal CDF
GELU'(x) = Phi(x) + x * phi(x)  where phi is the standard normal PDF
```

The gradient through the FFN is:

```
dL/dW_1 = X_norm^T @ (dL/d(FFN_out) @ W_2^T * GELU'(X_norm @ W_1))
```

The term `GELU'(·)` acts as a gate on the gradient: neurons with large positive pre-activations pass the gradient through approximately unchanged (GELU' ≈ 1 for large positive inputs), while neurons with large negative pre-activations block the gradient (GELU' ≈ 0 for large negative inputs). This selective gradient flow means that during fine-tuning, only a subset of FFN neurons are actively being updated at any given step.

---

## 8. The Optimizer: AdamW Mathematics in Full Depth

### 8.1 Adam's Adaptive Learning Rate

The core insight of Adam is that different parameters deserve different learning rates. A parameter that receives consistently large gradients (a frequently activated weight in a commonly attended head) should have a smaller effective step size than a parameter that receives rare, small gradients (a weight in a rarely activated attention head). Adam achieves this through per-parameter adaptive scaling.

For parameter `theta_i` at step `t`, with gradient `g_t`:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t           (first moment: exponential moving average of gradient)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         (second moment: exponential moving average of squared gradient)
```

The bias-corrected estimates account for the zero initialization of moments:

```
m_hat_t = m_t / (1 - beta1^t)     (corrected first moment)
v_hat_t = v_t / (1 - beta2^t)     (corrected second moment)
```

The parameter update:

```
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

The effective per-parameter learning rate is:

```
lr_effective_i = lr / (sqrt(v_hat_t^i) + eps)
```

For a parameter with large historical squared gradients (large `v_hat_t`), the effective learning rate is small. For a parameter with small historical squared gradients, the effective learning rate is close to `lr`. This self-normalization is what makes Adam robust to different parameter scales and gradient magnitudes across different layers and weight matrices.

### 8.2 The Weight Decay Correction in AdamW

In the original Adam implementation, weight decay was added to the gradient before the adaptive update:

```
# Adam (incorrect L2 regularization)
g_t_regularized = g_t + wd * theta_{t-1}
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t_regularized
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t_regularized^2
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

This is incorrect because the weight decay term `wd * theta` is divided by `sqrt(v_hat) + eps`, making the effective regularization strength vary per parameter. For parameters with large historical gradients, the regularization is weak; for parameters with small historical gradients, it is strong. This asymmetry is undesirable — we want consistent L2 regularization regardless of gradient magnitude.

AdamW fixes this by applying weight decay directly to the parameters after the adaptive update:

```
# AdamW (correct decoupled weight decay)
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t                     (no regularization in gradient)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)   (adaptive gradient step)
        - lr * wd * theta_{t-1}                                  (direct weight decay, separate)
```

The weight decay term `- lr * wd * theta_{t-1}` is equivalent to L2 regularization. It pulls each parameter toward zero by a fixed fraction `lr * wd` per step, independently of the gradient magnitude. This is the mathematically correct form of L2 regularization for adaptive optimizers.

### 8.3 Memory Cost of AdamW States

For each parameter in the model, AdamW stores:
- The parameter itself: `fp32 or bf16`
- The first moment `m`: `fp32` (maintains fp32 for numerical precision)
- The second moment `v`: `fp32` (maintains fp32 for numerical precision)

For full fine-tuning of a model with N parameters:

```
Memory_optimizer = N * (4 + 4) bytes  = 8N bytes  (first and second moments in fp32)
Memory_model_weights = N * 2 bytes                  (weights in bf16 for mixed precision)
Memory_gradients = N * 4 bytes                      (gradients in fp32 for precision)
Total_persistent = N * (8 + 2 + 4) bytes = 14N bytes

Plus activations: O(B * T * d * L) bytes
```

For a 7B model: `14 * 7e9 bytes ≈ 98 GB` just for persistent state, before accounting for activations. This is the hard constraint that makes FFT of large models require multiple high-end GPUs.

### 8.4 Parameter-Group-Specific Weight Decay

Not all parameters should receive weight decay. The correct practice, implemented rigorously in production FFT:

**Apply weight decay to**: All 2D weight matrices (W_Q, W_K, W_V, W_O, W_1, W_2, W_embed). These are the weight matrices where L2 regularization acts as a meaningful geometric prior (pulling weights toward lower-magnitude solutions).

**Do NOT apply weight decay to**: Bias vectors, LayerNorm gamma and beta parameters, positional embedding tables. These parameters are 1D and their optimal values are not biased toward zero — regularizing them toward zero actively degrades model quality. For bias vectors specifically, the optimal value depends on the data distribution, not on any sparsity prior.

---

## 9. Learning Rate Scheduling and Its Effect on Parameter Landscape

### 9.1 The Role of the Learning Rate in Parameter Space

The learning rate controls the step size in the 124-million or 7-billion dimensional parameter space. The loss landscape of a fine-tuned language model is not convex — it has saddle points, local minima, flat regions, and sharp ridges. The learning rate schedule determines how aggressively the optimizer moves through this landscape.

A learning rate that is too high causes the optimizer to overshoot good solutions, potentially escaping the basin of attraction of the pre-trained checkpoint's nearby minima and landing in a completely different region with poor generalization. A learning rate that is too low causes training to proceed too slowly, requiring many more steps and potentially getting stuck in suboptimal flat regions.

### 9.2 Linear Warmup: Mathematical Justification

The warmup phase increases the learning rate from 0 to `lr_max` over `W` steps:

```
lr(t) = lr_max * (t / W)   for 0 <= t < W
```

The justification is rooted in the Adam optimizer's moment estimates. At step 1, both `m_1` and `v_1` are initialized to 0. After one gradient step:

```
m_1 = (1 - beta1) * g_1 ≈ 0.1 * g_1    (for beta1=0.9)
v_1 = (1 - beta2) * g_1^2 ≈ 0.001 * g_1^2  (for beta2=0.999)
```

The bias correction is:
```
m_hat_1 = m_1 / (1 - 0.9^1) = m_1 / 0.1 = g_1
v_hat_1 = v_1 / (1 - 0.999^1) = v_1 / 0.001 = g_1^2
```

After bias correction, the effective update at step 1 is `lr * g_1 / (|g_1| + eps) ≈ lr * sign(g_1)`. This is a full step in the gradient direction, but the momentum `m` has not yet built up any history. The gradient estimate is based on a single batch, which is noisy for language modeling tasks. Using the full `lr_max` at step 1 means large steps based on noisy gradients from the very first batch — the most uncertain point in the training.

Warmup reduces the damage from this initial noise by starting with very small steps, allowing the momentum to accumulate reliable gradient statistics before full-sized steps are taken.

### 9.3 Cosine Annealing: Geometric Interpretation

The cosine annealing schedule:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

where `progress = (t - W) / (T - W) in [0, 1]`.

At `progress = 0` (just after warmup): `lr = lr_max`
At `progress = 0.5` (midpoint): `lr = 0.5 * (lr_max + lr_min)`
At `progress = 1.0` (end of training): `lr = lr_min`

The cosine schedule decreases slowly initially, spends a long time in the moderate range, then decreases rapidly near the end of training. This profile matches the intuition that:

1. Early in training, the model is far from a good solution and benefits from larger exploratory steps.
2. In the middle of training, moderate steps navigate the loss landscape efficiently.
3. Near the end of training, small steps allow fine-grained convergence to a sharp minimum.

Empirically, cosine annealing consistently outperforms linear decay and step decay schedules for transformer fine-tuning across model sizes and tasks.

---

## 10. Memory Architecture of Full Fine-Tuning

### 10.1 The Four Categories of Memory

GPU memory during FFT is consumed by four distinct categories, each with different scaling characteristics:

**Category 1: Model Parameters**
```
Memory = N_params * bytes_per_param
For bf16: N * 2 bytes
For fp32: N * 4 bytes
```
This is fixed regardless of batch size or sequence length.

**Category 2: Optimizer States**
```
Memory = N_params * (4 + 4) bytes  (first + second moments, always fp32)
= 8 * N_params bytes
```
Also fixed. This is the dominant memory consumer: for N=7B, this is 56 GB.

**Category 3: Gradients**
```
Memory = N_params * 4 bytes  (fp32 for numerical stability)
= 4 * N_params bytes
```
Also fixed. For N=7B: 28 GB.

**Category 4: Activations**
```
Memory ≈ B * T * d * L * bytes_per_activation
         (batch * seq_len * hidden_dim * num_layers)
```
This scales with batch size and sequence length. This is the variable component that gradient checkpointing reduces at the cost of recomputation.

**Total for fp32 FFT**: `N * (4 + 8 + 4) = 16N bytes`
**Total for bf16 FFT**: `N * (2 + 8 + 4) = 14N bytes` (model in bf16, states in fp32)

### 10.2 Memory Reduction Strategies

**Mixed Precision (bf16/fp32)**: Store model weights in bf16 (2 bytes), maintain optimizer states in fp32 (8 bytes). Gradients can be computed in bf16 and accumulated in fp32. Net reduction: roughly 16N → 14N bytes for the static component.

**Gradient Checkpointing**: Rather than retaining all intermediate activations from the forward pass, discard them and recompute them during the backward pass when needed. Reduces activation memory from `O(B*T*d*L)` to `O(B*T*d*sqrt(L))` with `sqrt(L)` recomputation overhead. This is a compute/memory trade-off.

**Gradient Accumulation**: Run smaller mini-batches but don't call the optimizer until N accumulation steps, simulating a larger effective batch size. Reduces activation memory by the accumulation factor. No increase in recomputation.

**ZeRO Optimizer States (DeepSpeed)**: Shards optimizer states, gradients, and optionally model parameters across GPUs, reducing per-GPU memory by the number of GPUs.

**CPU Offloading (Paged AdamW)**: Stores optimizer states in CPU RAM, paging them to GPU only when needed for updates. Allows very large models to train on consumer GPUs at the cost of PCIe bandwidth overhead.

---

## 11. Catastrophic Forgetting: Theory, Detection, and Mitigation

### 11.1 The Mechanism of Catastrophic Forgetting

Catastrophic forgetting occurs when full fine-tuning causes the model to lose previously learned capabilities while gaining performance on the fine-tuning task. The mechanism is clear: gradient descent on `L(theta; D_FT)` minimizes loss on the fine-tuning distribution without any constraint on performance on other distributions `D_pretrain`.

If the gradient `nabla_theta L(theta; D_FT)` points in a direction that increases `L(theta; D_pretrain)`, and the optimizer follows this gradient faithfully, the model will improve on the fine-tuning task while degrading on general language modeling. The parameters encoding general linguistic knowledge are overwritten by parameters tuned for the specific task.

Mathematically, the problem is that the fine-tuning dataset imposes a constraint on a subset of the input-output function, but the optimizer has freedom to change the function's behavior on all other inputs. Regularization toward the pre-trained weights provides an explicit constraint.

### 11.2 Detection: Measuring Forgetting During Training

The forgetting can be detected by monitoring performance on a set of general-purpose benchmarks throughout training:

```
forgetting_t = performance(model_pretrained, D_general)
             - performance(model_t, D_general)
```

A forgetting score that increases rapidly during training is a warning sign. The standard detection protocol is to evaluate on MMLU (knowledge retention), TruthfulQA (factuality), and a sample of the pre-training distribution every 200-500 training steps.

### 11.3 Mitigation Strategy 1: Low Learning Rate

The most direct mitigation. Using a learning rate 5x-10x lower than the default shifts the optimizer toward smaller parameter changes per step, reducing the magnitude of forgetting at the cost of slower convergence.

```
Recommended range for FFT: lr in [5e-6, 5e-5] for 7B models
Recommended range for FFT: lr in [1e-5, 2e-4] for <500M models
```

### 11.4 Mitigation Strategy 2: L2 Regularization Toward Pretrained Weights

Instead of standard L2 regularization (which pulls toward zero), regularize toward the pre-trained checkpoint `theta_pretrained`:

```
L_total(theta) = L_task(theta) + lambda * ||theta - theta_pretrained||^2
```

The gradient of the regularization term:

```
d/d(theta) [lambda * ||theta - theta_pretrained||^2] = 2 * lambda * (theta - theta_pretrained)
```

This pulls the fine-tuned parameters back toward the pre-trained values, with strength proportional to how far they have moved. It is a direct penalty on forgetting: parameters that have moved far from their pre-trained values are penalized quadratically.

### 11.5 Mitigation Strategy 3: Elastic Weight Consolidation (EWC)

EWC (Kirkpatrick et al., 2017) extends L2 regularization toward pre-trained weights by weighting the penalty by each parameter's importance to the pre-trained task. Parameters important to pre-training are heavily penalized for changing; unimportant parameters can change freely.

The importance weight is the diagonal of the Fisher Information Matrix `F`:

```
F_ii = E_{x ~ D_pretrain} [(d log P_theta(x) / d theta_i)^2]
```

The EWC loss is:

```
L_EWC(theta) = L_task(theta) + (lambda/2) * sum_i F_ii * (theta_i - theta_pretrained_i)^2
```

EWC is computationally expensive (requires computing the Fisher diagonal over the pre-training data) but theoretically principled. For FFT in practice, the simpler L2 regularization toward pre-trained weights often achieves similar results.

### 11.6 Mitigation Strategy 4: Data Replay

Mix a fraction of the pre-training data into the fine-tuning dataset:

```
D_training = (1-alpha) * D_fine_tuning + alpha * D_pretrain_sample
```

Training on this mixed dataset simultaneously minimizes the fine-tuning loss and maintains the pre-training loss, preventing catastrophic forgetting by construction. Alpha = 0.05 to 0.1 (5%-10% pre-training data) is typically sufficient to maintain most general capabilities.

The practical challenge is obtaining the original pre-training data, which is often not publicly available for commercial models.

---

## 12. Mixed Precision Training: FP32, BF16, and FP16

### 12.1 Floating Point Formats Compared

| Format | Sign | Exponent | Mantissa | Total Bits | Range | Precision |
|--------|------|----------|----------|------------|-------|-----------|
| FP32   | 1    | 8        | 23       | 32         | ±3.4e38 | ~7 decimal digits |
| BF16   | 1    | 8        | 7        | 16         | ±3.4e38 | ~2 decimal digits |
| FP16   | 1    | 5        | 10       | 16         | ±65504  | ~3 decimal digits |

BF16 and FP16 both use 16 bits, but differ crucially: BF16 has the same exponent range as FP32 (8 bits), while FP16 has a much smaller range (5 bits). This makes BF16 far safer for deep learning: gradient computations that produce values larger than 65504 will overflow to infinity in FP16 but remain finite in BF16. BF16 is preferred over FP16 for all transformer training tasks.

### 12.2 The Mixed Precision Training Protocol

The standard Automatic Mixed Precision (AMP) protocol for FFT:

1. **Forward pass**: Compute activations in BF16. Operations like matrix multiplications and attention use BF16, which is 2x faster on modern GPUs (Tensor Cores are optimized for 16-bit).

2. **Loss computation**: In FP32. The LogSumExp in cross-entropy can produce large values that overflow BF16; FP32 ensures numerical stability.

3. **Backward pass**: Gradients initially computed in BF16, then accumulated in FP32. The accumulation uses FP32 to prevent gradient underflow from small gradient values.

4. **Optimizer step**: In FP32 throughout. The optimizer states (first and second moments) must be in FP32 because the second moment `v` can be very small for rarely activated parameters, and FP16/BF16 would underflow to zero.

5. **Weight update**: The FP32 master copy of weights is updated. The BF16 working copy (used for forward/backward) is then cast from the FP32 master copy.

The net effect: 2x memory savings on activations and model weights (BF16 vs FP32), with 1.5x-2x training speedup from faster BF16 hardware operations, while maintaining FP32 numerical precision for the critical gradient accumulation and optimizer steps.

### 12.3 Loss Scaling for FP16 (Not Needed for BF16)

FP16's limited exponent range (max 65504) means gradients that are small but non-zero in FP32 may underflow to 0 in FP16. The standard fix is gradient scaling: multiply the loss by a large scale factor S before the backward pass, then divide the gradients by S before the optimizer step. BF16 does not need this due to its larger exponent range.

---

## 13. Gradient Checkpointing: Trading Compute for Memory

### 13.1 The Memory Problem

Without gradient checkpointing, the backward pass requires all intermediate activations from the forward pass to be available in GPU memory simultaneously. For a model with L layers, this means storing:

```
Activations ≈ L * B * T * d * bytes_per_activation
```

For L=32, B=4, T=2048, d=4096, bf16: `32 * 4 * 2048 * 4096 * 2 bytes ≈ 4.3 GB` just for one layer's intermediate activations, times 32 layers = `~137 GB`. This completely dominates the memory budget.

### 13.2 The Checkpointing Solution

Gradient checkpointing (Chen et al., 2016) saves only a subset of activations (the "checkpoints") and recomputes the discarded activations during the backward pass when they are needed for gradient computation.

The standard strategy is to checkpoint at transformer block boundaries:
- During the forward pass: save only the input to each transformer block. Discard all intermediate activations within the block.
- During the backward pass: when the backward pass reaches a block boundary, re-run the block's forward pass from its saved input to reconstruct all intermediate activations needed for gradient computation.

This reduces activation memory from `O(L)` (one set of activations per layer) to `O(sqrt(L))` with a `sqrt(L)` checkpointing strategy, at the cost of a single additional forward pass through the network (approximately 33% extra compute).

### 13.3 Memory Analysis with Gradient Checkpointing

Without checkpointing: activation memory `= O(L * B * T * d)`
With per-block checkpointing: activation memory `= O(B * T * d)` (only block boundary activations)

For the 32-layer example: reduction from ~137 GB to ~4.3 GB in activation memory alone. This makes FFT of large models feasible on hardware that would otherwise be insufficient.

---

## 14. Gradient Accumulation: Simulating Large Batches

### 14.1 The Batch Size Problem

The optimal batch size for FFT is typically in the range of 32-512 sequences. Larger batches provide more stable gradient estimates, allow faster convergence in terms of steps (though not wall-clock time per step), and often find better minima. However, each sequence in the batch requires storing its activations in memory during the backward pass, making large batches memory-intensive.

With gradient accumulation (accumulation_steps = K):
1. Run K forward-backward passes with micro-batch size B_micro.
2. Accumulate (sum) the gradients from all K passes without zeroing them.
3. After K passes, scale the accumulated gradient by 1/K.
4. Execute the optimizer step on the accumulated gradient.
5. Zero the gradients and begin the next accumulation window.

This simulates a batch size of `K * B_micro` with the memory of `B_micro`.

### 14.2 The Mathematical Equivalence

For a loss function that averages over the batch:

```
L_effective = (1/(K*B_micro)) * sum_{k=1}^K sum_{b=1}^{B_micro} l(x_{k,b})
```

The gradient of this is:

```
nabla L_effective = (1/(K*B_micro)) * sum_{k=1}^K sum_{b=1}^{B_micro} nabla l(x_{k,b})
```

With gradient accumulation, we compute:

```
nabla_k = (1/B_micro) * sum_{b=1}^{B_micro} nabla l(x_{k,b})   (gradient for micro-batch k)
accumulated = sum_{k=1}^K nabla_k / K = (1/(K*B_micro)) * sum_{k,b} nabla l(x_{k,b})
```

This is exactly equal to the gradient computed over the full effective batch. The optimizer step is mathematically identical to what would be computed with the full batch, so the training dynamics are equivalent. The only practical difference is that gradient accumulation prevents the use of batch normalization statistics across the full effective batch, but transformers use layer normalization (computed per-example), so this is not an issue.

---

## 15. The Full Fine-Tuning Pipeline: Step-by-Step

Full fine-tuning consists of eight sequential stages that must be executed in order with validation at each transition.

**Stage 1: Infrastructure and Hardware Assessment**

Before a single line of training code is written, the hardware constraints must be determined. For a model with N parameters using AdamW in mixed precision:

```
Required_GPU_memory = 14 * N_bytes (model + optimizer states + gradients)
                    + activation_memory(batch_size, seq_len, num_layers)
```

If this exceeds single-GPU capacity, distributed training with model parallelism, ZeRO optimizer sharding, or gradient checkpointing must be configured. Making this calculation before building the training pipeline prevents wasted effort on configurations that cannot run.

**Stage 2: Dataset Engineering**

The dataset must be constructed with extreme care because FFT amplifies both quality and noise in the training data. The full set of preprocessing steps: deduplication (exact and near-duplicate), format standardization using the model's chat template, loss masking validation (verify prompt tokens have label=-100), length distribution analysis, and quality scoring using an LLM-as-judge or human annotators. The dataset must be split into train, validation, and test sets before any preprocessing that could leak information.

**Stage 3: Model Initialization**

Load the pre-trained checkpoint into the training-configured model. Verify that the architecture matches the checkpoint exactly — mismatches in `num_layers`, `hidden_size`, or `vocab_size` will cause shape errors that are caught at load time. Set up gradient checkpointing if required by the memory budget.

**Stage 4: Optimizer and Scheduler Construction**

Build the AdamW optimizer with correctly separated parameter groups (weight decay for weight matrices, no weight decay for biases and norms). Compute the total training steps from dataset size, batch size, gradient accumulation steps, and number of epochs. Build the cosine LR schedule with appropriate warmup steps (3-10% of total steps).

**Stage 5: Training Loop Execution**

The core training loop: fetch batch, forward pass with mixed precision, compute masked cross-entropy loss, scale by accumulation steps, backward pass, accumulate gradients, optimizer step and LR scheduler step at accumulation boundaries, zero gradients.

**Stage 6: Continuous Evaluation and Monitoring**

At every evaluation interval (typically every 50-200 optimizer steps): compute validation loss and perplexity, generate sample outputs for qualitative assessment, evaluate on general-purpose benchmarks to monitor forgetting, log all metrics to the observability system.

**Stage 7: Checkpointing and Recovery**

Save complete training state (model weights, optimizer states, scheduler state, global step, configuration, best validation loss) at regular intervals and on new validation loss records. The checkpoint must contain everything needed to resume training identically from any saved point.

**Stage 8: Post-training Processing**

Evaluate the best checkpoint on the held-out test set and the full benchmark suite. Apply any post-processing (quantization for deployment, merging with other models, evaluating for safety regressions). Document the training run with the complete configuration, final metrics, and sample outputs for reproducibility.

---

## 16. FFT with a Small Language Model: Practical Walkthrough

### 16.1 Using GPT-2 as the Demonstration Model

GPT-2 (117M parameters) is the ideal vehicle for learning FFT because every concept applies identically to a 70B model — the only difference is scale. The GPT-2 architecture is a pure decoder-only transformer: 12 layers, 768 hidden dimension, 12 attention heads, 3072 FFN dimension, 50257 vocabulary.

For FFT of GPT-2 on a single GPU:
```
Model weights (fp32):   117M * 4 bytes = 468 MB
Optimizer states (fp32): 117M * 8 bytes = 936 MB
Gradients (fp32):        117M * 4 bytes = 468 MB
Activations (batch=4, T=512): ~200 MB
Total:                   ~2.1 GB
```

This comfortably fits in an 8GB GPU or can run on CPU for small datasets.

### 16.2 Expected Training Dynamics for FFT vs. LoRA

The key observable difference between FFT and LoRA on the same dataset:

**Loss curve**: FFT typically achieves a lower final training loss because all parameters are free to adapt. LoRA may plateau at a higher loss because the low-rank constraint limits the expressivity of the adaptation.

**Convergence speed**: FFT often converges faster in terms of steps because the parameter update space is larger, allowing the model to make more direct progress toward the minimum.

**Generalization gap**: FFT has a larger generalization gap (difference between training and validation loss) because the larger parameter space is more susceptible to overfitting on small datasets.

**Forgetting**: FFT shows measurable forgetting on general benchmarks after more than 2 epochs on a narrow dataset. LoRA shows minimal forgetting because the base weights are frozen.

### 16.3 The Optimal Learning Rate Regime

For GPT-2 FFT: `lr in [1e-5, 2e-4]` with cosine decay. The optimal value depends on dataset size and effective batch size. The general rule from scaling laws: as model size increases, the optimal learning rate decreases. A 7B model should use `lr ~= 1e-5` to `3e-5`, while a 117M model can use `lr ~= 1e-4` to `2e-4`.

---

## 17. Hyperparameter Sensitivity and Tuning Strategy

### 17.1 The Learning Rate is the Most Critical Hyperparameter

The learning rate determines whether fine-tuning succeeds or fails. Too high: destructive parameter updates, training divergence, catastrophic forgetting. Too low: insufficient adaptation, slow convergence, underfitting.

The recommended tuning strategy for production FFT: run a learning rate range test (LR finder) where you increase the learning rate from 1e-7 to 1e-3 over 100 steps and plot the loss vs. learning rate. The optimal learning rate is typically at the steepest descent point on this curve, divided by 3-10.

### 17.2 Batch Size and Gradient Accumulation

The effective batch size affects both the noise level of gradient estimates and the computational efficiency. For language model fine-tuning:

- **Effective batch size too small (< 8)**: Noisy gradients, unstable loss curves, poor generalization.
- **Effective batch size optimal (32-256)**: Stable training, good generalization.
- **Effective batch size too large (> 1024)**: Stable but may find flatter minima; requires increased learning rate.

The linear scaling rule: when doubling the effective batch size, multiply the learning rate by `sqrt(2)` (square root scaling) or by 2 (linear scaling). For fine-tuning, square root scaling tends to perform better.

### 17.3 Number of Epochs

For FFT specifically:
- **1 epoch**: Often sufficient for large, high-quality datasets (>100K examples). Low risk of overfitting.
- **2-3 epochs**: Standard for medium datasets (10K-100K examples). Monitor validation loss closely.
- **3-5 epochs**: May be needed for very small datasets (<5K examples), but requires strong regularization (weight decay, dropout, early stopping).
- **>5 epochs**: Almost always overfits in full fine-tuning. Use LoRA instead for small datasets.

---

## 18. Evaluation: Metrics, Benchmarks, and Failure Modes

### 18.1 Task-Specific Metrics

**Perplexity**: The exponentiated cross-entropy loss on held-out data. Primary metric for language modeling tasks. Lower is better. A drop from pre-training perplexity (~45 on web text) to fine-tuned perplexity (~5-10 on task data) indicates successful adaptation.

**Exact Match / F1**: For question answering tasks. Exact Match is 1 if the model's output exactly matches the ground truth, 0 otherwise. F1 measures token-level overlap between prediction and reference.

**ROUGE-L**: Longest common subsequence recall. Standard for summarization tasks.

**Pass@k**: For code generation. The fraction of problems solved within k attempts. HumanEval and MBPP are standard benchmarks using Pass@1.

### 18.2 General Capability Retention Metrics

These should be evaluated before and after FFT to quantify forgetting:

**MMLU (Massive Multitask Language Understanding)**: 57-subject multiple-choice test covering math, science, humanities, law. A good measure of knowledge retention.

**TruthfulQA**: Tests whether the model generates factually true statements vs. plausible-sounding falsehoods. Fine-tuning can increase hallucination rate.

**ARC (AI2 Reasoning Challenge)**: Grade-school science questions requiring multi-step reasoning.

### 18.3 Common Failure Modes

**Mode 1: Repetition loops**. The model generates the same phrase repeatedly. Cause: the fine-tuning data had repetitive completions, or the model was trained past the point of overfit. Detection: log generation samples during training.

**Mode 2: Format collapse**. The model responds in the correct format for a few turns then degrades to free-form output. Cause: inconsistent formatting in the training data. Detection: run format compliance evaluation.

**Mode 3: Factual hallucination increase**. The fine-tuned model produces more confident but wrong answers than the base model. Cause: the fine-tuning data contained incorrect facts, or the model learned to produce confident-sounding output regardless of accuracy.

**Mode 4: Length degeneration**. The model systematically produces shorter or longer outputs than desired. Cause: length distribution in training data is skewed, or the model is overfitting to average length.

---

## 19. Production Considerations

### 19.1 Distributed Training Strategies

For models that exceed single-GPU memory, three parallelism strategies are used:

**Data Parallelism (DP)**: Each GPU holds a complete copy of the model. Different batches are processed on different GPUs. Gradients are synchronized across GPUs via all-reduce after each backward pass. Scales training throughput but not memory capacity.

**Tensor Parallelism (TP)**: Individual weight matrices are split across GPUs along one dimension. The attention projections are split by heads, and FFN matrices are split along the intermediate dimension. Each GPU holds a shard of each weight matrix. Requires tight GPU-GPU communication (NVLink) for the all-reduce operations within each forward pass. Used within a single node.

**Pipeline Parallelism (PP)**: Different layers are placed on different GPUs. Each batch is split into micro-batches that pipeline through the GPU stages. Reduces communication overhead (only activations at stage boundaries need to be communicated) but introduces pipeline bubbles (idle GPUs waiting for the pipeline to fill).

**ZeRO (Zero Redundancy Optimizer)**: DeepSpeed's ZeRO shards optimizer states (ZeRO-1), gradients (ZeRO-2), or model parameters (ZeRO-3) across data-parallel GPUs, combining the throughput of data parallelism with the memory reduction of model parallelism.

### 19.2 Observability Stack for Production FFT

A production FFT training run requires:

**Metric tracking**: Weights & Biases or MLflow for logging training loss, validation loss, perplexity, learning rate, gradient norm, GPU memory, and throughput (tokens/second) at every logging step.

**Sample generation logging**: At every evaluation step, log a fixed set of prompts and the model's current responses. This provides the qualitative monitoring that perplexity alone cannot provide.

**Benchmark tracking**: Run MMLU, TruthfulQA, and task-specific benchmarks every 500-1000 steps to monitor forgetting in real time rather than discovering it at the end of training.

**Checkpoint management**: Save at least the last 3 checkpoints and the best checkpoint (by validation loss). Automatic resumption from the latest checkpoint on job failure.

**Alert thresholds**: Set automated alerts for: validation loss increasing for more than 3 consecutive evaluation intervals (overfitting), gradient norm consistently exceeding 5x the max_grad_norm threshold (training instability), GPU memory approaching 95% capacity.

### 19.3 Inference Optimization Post-FFT

After FFT, the model weights are in the same format as any other pre-trained model and can be optimized for inference using identical techniques:

**GPTQ quantization**: Post-training quantization to int4 or int8 using the GPTQ algorithm, which minimizes quantization error by finding the optimal quantization grid for each layer. Reduces model memory by 4x-8x with minimal quality degradation.

**KV Cache management**: During autoregressive generation, the key-value tensors for all previous tokens are cached to avoid recomputation. The KV cache size for a 7B model at 2048 tokens: `2 * L * d_kv * T * B * 2 bytes = 2 * 32 * 4096 * 2048 * 1 * 2 bytes ≈ 1 GB`.

**Flash Attention 2 for inference**: Even at inference time, Flash Attention 2 provides significant speedups and memory savings for long-context requests.

---

## 20. Trade-offs, Best Practices, and Recommendations

### Core Trade-offs

**FFT vs. LoRA**: FFT achieves higher peak quality but requires 5x-10x more GPU memory and produces a completely separate model copy per fine-tuned task. LoRA is memory-efficient, avoids catastrophic forgetting, and requires storing only small adapter files (<100MB) per task versus full model copies (14GB-140GB). Choose FFT when: (a) LoRA demonstrably underperforms on your evaluation benchmarks, (b) you have a single well-defined task with abundant data, and (c) hardware budget is not a constraint.

**More epochs vs. early stopping**: Additional training epochs reduce training loss indefinitely but risk overfitting and catastrophic forgetting. Always use early stopping based on held-out validation loss. Never report test set performance until training is complete.

**Higher learning rate vs. lower learning rate**: Higher learning rate converges faster and can find sharper minima but increases the risk of catastrophic forgetting and training divergence. Lower learning rate is safer but slower. Use a learning rate range test to find the optimal value experimentally rather than guessing.

**Gradient checkpointing vs. larger batch size**: Without gradient checkpointing, memory freed from not storing activations can be used for a larger batch size. With gradient checkpointing, you can train with a smaller per-GPU batch size and use gradient accumulation to simulate a large effective batch. For most FFT use cases, the memory savings from gradient checkpointing are more valuable than the batch size flexibility.

### Key Best Practices

**Always measure forgetting** on at least two general benchmarks (MMLU and TruthfulQA) at every evaluation checkpoint. Do not declare fine-tuning successful based on task performance alone without verifying that general capabilities are preserved.

**Use bf16 mixed precision** universally. There is no reason to use fp32 for the forward pass and activations in modern transformer training. The quality difference between bf16 and fp32 training is negligible for fine-tuning, and the memory and speed benefits are substantial.

**Separate weight decay parameter groups** rigorously. Biases and layer normalization parameters should never receive L2 regularization. Applying weight decay uniformly to all parameters (a common mistake) degrades model quality and causes instability in the layer normalization parameters.

**Monitor gradient norms at every step**. The gradient norm before clipping is a real-time indicator of training health. A norm consistently near the clipping threshold (1.0) means the gradient is being clipped at almost every step — reduce the learning rate. A norm that suddenly spikes indicates a problematic batch or impending divergence.

**Log sample generations at every evaluation**. No numerical metric captures the full picture of model behavior. Qualitative monitoring through generated samples catches hallucination increases, format regressions, repetition loops, and off-task behavior that perplexity metrics miss entirely.
