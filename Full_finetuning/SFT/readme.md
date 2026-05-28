# Supervised Fine-Tuning (SFT): A Complete Technical Reference

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [The SFT Pipeline: Step-by-Step](#4-the-sft-pipeline-step-by-step)
5. [Loss Function Derivation](#5-loss-function-derivation)
6. [Gradient Flow and Backpropagation in Transformers](#6-gradient-flow-and-backpropagation-in-transformers)
7. [Optimization Algorithms for Fine-Tuning](#7-optimization-algorithms-for-fine-tuning)
8. [Parameter-Efficient Fine-Tuning (PEFT) vs Full Fine-Tuning](#8-parameter-efficient-fine-tuning-peft-vs-full-fine-tuning)
9. [LoRA: Mathematical Derivation and Implementation](#9-lora-mathematical-derivation-and-implementation)
10. [Data Engineering for SFT](#10-data-engineering-for-sft)
11. [Training Dynamics, Stability, and Hyperparameters](#11-training-dynamics-stability-and-hyperparameters)
12. [Evaluation Methodology](#12-evaluation-methodology)
13. [SFT with a Small Language Model (SLM): Practical Walkthrough](#13-sft-with-a-small-language-model-slm-practical-walkthrough)
14. [Production Considerations](#14-production-considerations)
15. [Trade-offs, Best Practices, and Recommendations](#15-trade-offs-best-practices-and-recommendations)

---

## 1. Introduction and Motivation

### What is Supervised Fine-Tuning?

Supervised Fine-Tuning (SFT) is the process of continuing the training of a pre-trained language model on a curated, labeled dataset of (input, desired_output) pairs, using standard supervised learning objectives. The goal is to shift the model's behavior from general-purpose next-token prediction toward a specific, structured task or interaction style — such as instruction following, question answering, code generation, or dialogue.

A pre-trained large or small language model has already learned rich representations of language from vast unsupervised corpora. During pre-training, the model learned syntax, semantics, factual knowledge, reasoning patterns, and stylistic conventions. However, pre-training alone does not make a model "aligned" — it predicts the next token without any preference for helpfulness, accuracy, or safety. SFT bridges this gap.

The intuition is powerful: rather than training from scratch, we take a model with rich general-purpose knowledge and nudge its parameters so that its outputs on specific input distributions match human-preferred reference outputs. This is computationally far cheaper than pre-training, typically requiring 0.01% to 1% of the original training compute, yet can dramatically change the model's behavior.

### Why SFT Matters in the Modern ML Stack

SFT occupies the first stage of the RLHF (Reinforcement Learning from Human Feedback) pipeline, made famous by InstructGPT and ChatGPT. The standard alignment pipeline is:

```
Pre-training  →  SFT  →  Reward Modeling  →  RLHF/PPO or DPO
```

Even without the reinforcement learning stages, SFT alone produces a model that can follow instructions coherently. Models like Alpaca (fine-tuned LLaMA), Vicuna, and Mistral-Instruct were created with SFT only. The SFT stage is foundational because the reward model in stage three is itself trained on SFT model outputs, and PPO/DPO fine-tunes from the SFT checkpoint.

### Comparison with Pre-training

| Dimension | Pre-training | SFT |
|---|---|---|
| Dataset size | Trillions of tokens | Thousands to millions of examples |
| Objective | Unsupervised next-token prediction | Supervised cross-entropy on target tokens |
| Compute | Months on thousands of GPUs | Hours to days on 1-8 GPUs |
| Goal | General language modeling | Task-specific behavior alignment |
| Data quality requirement | Moderate (large scale dominates) | Very high (quality over quantity) |
| Risk | Underfitting, slow convergence | Catastrophic forgetting, overfitting |

---

## 2. Mathematical Foundations

### 2.1 The Language Modeling Objective

A language model parameterized by theta defines a probability distribution over sequences. Given a sequence of tokens `x = (x_1, x_2, ..., x_T)`, the model factorizes the joint probability autoregressively:

```
P(x; theta) = product_{t=1}^{T} P(x_t | x_1, ..., x_{t-1}; theta)
```

The model computes, at each position `t`, a probability distribution over the full vocabulary `V` conditioned on all preceding tokens. The output at each step is a logit vector `z_t` of size `|V|`, which is converted to probabilities via softmax:

```
P(x_t = v | x_{<t}; theta) = softmax(z_t)[v] = exp(z_t[v]) / sum_{v'} exp(z_t[v'])
```

### 2.2 The SFT Dataset and Formatting

An SFT dataset `D` consists of N examples, each being a (prompt, completion) pair:

```
D = {(p_i, c_i)}_{i=1}^{N}
```

These are concatenated with special tokens to form a training sequence. The most common format is:

```
[BOS] [SYSTEM: instruction_text] [USER: user_message] [ASSISTANT: response_text] [EOS]
```

The critical distinction in SFT versus general language modeling is the loss mask. During training, the cross-entropy loss is computed **only over the completion tokens** `c_i`, not over the prompt tokens `p_i`. This is because we want to train the model to generate correct completions given the prompt, not to predict the prompt itself.

Formally, for a training example with tokenized sequence `(p_1, ..., p_L, c_1, ..., c_M)`, the loss is:

```
L(theta) = - (1/M) * sum_{t=L+1}^{L+M} log P(x_t | x_1, ..., x_{t-1}; theta)
```

where the sum runs only over completion tokens from position `L+1` to `L+M`.

### 2.3 The Cross-Entropy Loss in Detail

For a single token position `t` where the ground truth token is `x_t = v*`, the cross-entropy loss is:

```
l_t = -log P(x_t = v* | x_{<t}; theta)
     = -log [exp(z_t[v*]) / sum_{v} exp(z_t[v])]
     = -z_t[v*] + log sum_{v} exp(z_t[v])
     = -z_t[v*] + LogSumExp(z_t)
```

The gradient of this loss with respect to the logit vector `z_t` is:

```
d(l_t)/d(z_t[v]) = P(x_t = v | x_{<t}; theta) - 1{v = v*}
                 = softmax(z_t)[v] - 1{v = v*}
```

This gradient has a beautiful interpretation: the model receives a negative signal proportional to the probability it assigned to all wrong tokens, and a positive signal proportional to `1 - P(correct)`. When the model is confident and correct, the gradient magnitude is near zero. When it is wrong or uncertain, the gradient is large.

### 2.4 The Empirical Risk

The total SFT loss over the dataset is the average cross-entropy over all completion tokens across all examples:

```
L(theta) = (1/|D|) * sum_{i=1}^{|D|} (1/|c_i|) * sum_{t in completion_i} -log P(x_t | x_{<t}; theta)
```

In practice, implementations vary: some take the mean over all completion tokens across the entire batch (flattened), while others compute per-example losses and average them. The flattened version implicitly weights shorter examples less than longer ones, which can introduce subtle biases if completion lengths vary widely across the dataset.

---

## 3. Architecture Deep Dive

### 3.1 The Transformer Architecture for SFT

All modern LLMs and SLMs used in SFT are decoder-only Transformers. The architecture processes sequences causally: at position `t`, the model can only attend to positions `1, ..., t` (causal masking). This is essential for autoregressive generation.

The core Transformer block, repeated `L` times, consists of:

**Multi-Head Causal Self-Attention**

For a layer with hidden dimension `d` and `H` attention heads, the input `X` of shape `[seq_len, d]` is projected into queries, keys, and values:

```
Q = X * W_Q,  W_Q in R^{d x d_k}
K = X * W_K,  W_K in R^{d x d_k}
V = X * W_V,  W_V in R^{d x d_v}
```

where `d_k = d_v = d / H`. The attention output for head `h` is:

```
Attention(Q_h, K_h, V_h) = softmax( (Q_h * K_h^T) / sqrt(d_k) + M ) * V_h
```

where `M` is the causal mask: `M[i,j] = 0` if `j <= i`, else `-infinity`. The softmax of `-infinity` gives zero attention weight, preventing the model from attending to future tokens.

The multi-head output is concatenated and linearly projected:

```
MultiHead(X) = Concat(head_1, ..., head_H) * W_O,  W_O in R^{d x d}
```

**Feed-Forward Network (FFN)**

After attention, each position is passed through a position-wise FFN:

```
FFN(x) = W_2 * activation(W_1 * x + b_1) + b_2
```

where `W_1 in R^{d x d_ff}` expands to a larger intermediate dimension (typically `d_ff = 4*d`), and `W_2` projects back. Modern models use SwiGLU or GELU activations rather than ReLU:

```
SwiGLU(x) = Swish(W_1 * x) * (W_gate * x)
```

**Residual Connections and Layer Normalization**

Each sub-layer uses a residual connection. Modern models use Pre-LayerNorm (norm before the sublayer):

```
X = X + MultiHead(LayerNorm(X))
X = X + FFN(LayerNorm(X))
```

Pre-LayerNorm is preferred over Post-LayerNorm in fine-tuning because it provides more stable gradients when updating from a pre-trained checkpoint.

**Position Embeddings**

Modern SLMs use Rotary Position Embeddings (RoPE), which encode position information directly into the query-key dot products rather than adding absolute position embeddings to the input. RoPE applies a rotation matrix `R_theta_m` to the query and key vectors at position `m`:

```
q_m = R_theta_m * q,  k_m = R_theta_m * k
```

The attention score between positions `m` and `n` then depends only on the relative offset `m - n`, enabling better length generalization.

### 3.2 Parameter Count for SLMs

For a typical SLM like GPT-2 small (117M parameters) or Phi-2 (2.7B parameters):

```
Parameters = vocab_size * d_model                   (embedding table)
           + L * (4 * d_model^2 + 2 * d_model * d_ff)  (attention + FFN per layer)
           + d_model                                 (final layer norm)
```

For GPT-2 small: `d=768, L=12, d_ff=3072, vocab=50257`:
```
Embedding:  50257 * 768  ~= 38.6M
Per layer:  4 * 768^2 + 2 * 768 * 3072 = 2.36M + 4.72M = 7.08M
All layers: 12 * 7.08M = 85.0M
Total:      ~123.6M
```

---

## 4. The SFT Pipeline: Step-by-Step

The complete SFT pipeline consists of seven stages that must be executed in the correct order with careful validation at each transition.

### Stage 1: Pre-training Checkpoint Selection

The choice of base model is the most consequential architectural decision in the SFT workflow. Pre-trained models differ not only in parameter count but in tokenizer vocabulary, context length, architectural variants (grouped query attention, sliding window attention), and the composition of pre-training data.

For task-specific SFT, a model pre-trained on domain-relevant data will converge faster and to a better optimum than a general-purpose model. For example, fine-tuning CodeLlama for code generation will produce superior results compared to fine-tuning Llama-2-Base for the same task, even if parameter counts are identical.

The base model should be loaded with full precision (fp32) or brain float (bfloat16) weights, avoiding quantized formats (int8, int4) during training unless using QLoRA specifically, as quantized representations introduce approximation errors into the gradient computation.

### Stage 2: Dataset Construction and Curation

The dataset is the most important driver of SFT quality. The adage "garbage in, garbage out" applies with particular force here because the model will imitate the patterns in the training data. Dataset construction involves:

**Collection**: Gathering raw (prompt, response) pairs from human demonstrators, distillation from a larger teacher model (e.g., GPT-4), or extraction from structured sources.

**Deduplication**: Near-duplicate examples waste compute and can cause the model to memorize specific outputs rather than generalizing. MinHash LSH or exact hash deduplication should be applied at the character, token, and semantic levels.

**Quality Filtering**: Rule-based filters remove examples that are too short, contain harmful content, have formatting artifacts, or use languages not in the model's vocabulary. A secondary classifier or LLM-as-judge can score response quality.

**Diversity Analysis**: The dataset must cover the intended input distribution broadly. Embedding the prompts and clustering them reveals gaps. A dataset of 10,000 highly similar examples is worse than 1,000 diverse examples.

**Format Standardization**: All examples must be formatted consistently using the model's chat template. Inconsistent formatting is a leading cause of poor SFT results.

### Stage 3: Tokenization and Dataset Preprocessing

Each training example is tokenized and formatted into the model's expected input format. The tokenizer must be the same as the one used during pre-training — using a different tokenizer introduces vocabulary mismatches that make the embeddings incoherent.

The key preprocessing decisions are:

**Maximum Sequence Length**: Truncation at a fixed length `L_max` affects which examples are fully included. Examples longer than `L_max` are either discarded or truncated from the prompt side (preserving the completion). The batch padding strategy pads shorter sequences with a special `[PAD]` token and uses an attention mask to prevent the model from attending to padding positions.

**Packing**: Rather than padding each example individually (wasteful when sequence lengths vary), packing concatenates multiple examples into a single sequence of length `L_max`, separated by EOS tokens. This dramatically increases GPU utilization (from ~40% to ~90%) but requires careful handling of attention masks to prevent cross-contamination between packed examples. Modern implementations use block-diagonal attention masks for this purpose.

**Loss Masking**: As described in Section 2.2, the loss is computed only on completion tokens. The implementation uses a label tensor where prompt token positions are set to -100 (PyTorch's `ignore_index`), and completion token positions contain the actual token IDs.

### Stage 4: Model Initialization and Configuration

The pre-trained weights are loaded into the model, followed by configuration of which parameters will be updated during training. For full fine-tuning, all parameters are trainable. For LoRA fine-tuning, only the adapter matrices are trainable and the original weights are frozen.

Mixed precision training is configured here. The standard is:
- Model weights in bfloat16 (better numerical range than fp16 for transformer training)
- Gradient accumulation in fp32 (prevents underflow in accumulated gradients)
- Master copy of weights in fp32 for the optimizer

### Stage 5: Training Loop Execution

The training loop iterates over the dataset for one or more epochs. At each step:

1. Fetch a batch of tokenized sequences and their attention masks and label masks.
2. Forward pass through the model to compute logits.
3. Compute the masked cross-entropy loss.
4. Backward pass to compute gradients with respect to all trainable parameters.
5. Gradient clipping to prevent exploding gradients.
6. Optimizer step to update parameters.
7. Learning rate scheduler step.
8. Log metrics (loss, gradient norm, learning rate, throughput).

Gradient accumulation is used to simulate larger batch sizes than fit in GPU memory. With `accumulation_steps=8` and per-GPU batch size 4, the effective batch size is 32, matching the gradient dynamics of a 32-example batch while using memory for only 4.

### Stage 6: Checkpointing and Evaluation

At regular intervals (every N steps or every epoch), the model weights are saved along with the optimizer state, scheduler state, and training metadata. This enables resuming from interruptions — critical for multi-day training runs.

Evaluation runs on a held-out validation set to compute held-out loss. A rapidly decreasing training loss with a plateauing or increasing validation loss signals overfitting, requiring early stopping or regularization adjustment.

### Stage 7: Post-training Analysis and Model Merging

After training completes, LoRA adapters (if used) are merged into the base model weights for deployment, producing a single set of weights without the adapter overhead. The final model is evaluated on task-specific benchmarks, and optionally submitted to human evaluators for side-by-side comparisons.

---

## 5. Loss Function Derivation

### 5.1 Deriving the Cross-Entropy from Maximum Likelihood

SFT is maximum likelihood estimation (MLE) of the model parameters given the training data. We want to find:

```
theta* = argmax_{theta} prod_{i=1}^{N} P(c_i | p_i; theta)
```

Taking the logarithm (which is monotonic, so does not change the argmax):

```
theta* = argmax_{theta} sum_{i=1}^{N} log P(c_i | p_i; theta)
```

Expanding the autoregressive factorization of the completion:

```
log P(c_i | p_i; theta) = sum_{t=1}^{|c_i|} log P(c_t^i | p_i, c_1^i, ..., c_{t-1}^i; theta)
```

Converting from maximization to minimization by negating:

```
L(theta) = -(1/N) * sum_{i=1}^{N} (1/|c_i|) * sum_{t=1}^{|c_i|} log P(c_t^i | context; theta)
```

This is exactly the standard cross-entropy loss evaluated only on completion tokens.

### 5.2 Connection to KL Divergence

The SFT objective is equivalent to minimizing the KL divergence between the empirical distribution `P_data(x)` (concentrated on the training examples) and the model distribution `P_theta(x)`:

```
KL(P_data || P_theta) = E_{x ~ P_data} [log P_data(x) - log P_theta(x)]
                      = H(P_data) + CrossEntropy(P_data, P_theta)
```

Since `H(P_data)` is a constant with respect to `theta`, minimizing `KL(P_data || P_theta)` is equivalent to minimizing the cross-entropy loss. This perspective reveals a fundamental limitation of SFT: by minimizing forward KL, the model is driven to cover all modes of the training distribution (mode-covering behavior), which can lead to averaged, "safe" responses when the training data contains diverse or inconsistent demonstrations.

### 5.3 Perplexity as an Evaluation Metric

The exponentiated average cross-entropy is called perplexity:

```
PPL = exp(L(theta)) = exp( -(1/T) * sum_t log P(x_t | x_{<t}) )
```

where `T` is the total number of completion tokens evaluated. Perplexity has the intuitive interpretation of the effective vocabulary size the model is uncertain between at each step. A perplexity of 10 means the model is on average choosing among 10 roughly equally probable tokens — a perplexity of 1 means perfect prediction.

For SFT evaluation, we report validation perplexity on held-out examples. Reductions from, say, 45 to 8 indicate the model has learned the task distribution well.

---

## 6. Gradient Flow and Backpropagation in Transformers

### 6.1 The Vanishing/Exploding Gradient Problem

In deep transformer models, gradients flow backward through `L` layers. If the average gradient magnitude at each layer is `g`, after L layers the magnitude is approximately `g^L`. For `g < 1` (vanishing) or `g > 1` (exploding), training becomes unstable.

Modern transformers mitigate this through:

**Residual Connections**: The gradient can bypass any sublayer through the residual path, creating a "highway" for gradient flow. The gradient through a residual block is:

```
d(L)/d(X) = d(L)/d(Y) * (I + d(f)/d(X))
```

where `f` is the sublayer function. Even if `d(f)/d(X)` vanishes, the identity term `I` ensures the gradient signal is preserved.

**Pre-Layer Normalization**: Placing LayerNorm before the sublayer (rather than after) normalizes the inputs to attention and FFN, keeping activations in a well-conditioned range throughout training.

**Gradient Clipping**: The L2 norm of the full parameter gradient is clipped to a maximum value (typically 1.0):

```
if ||grad|| > max_norm:
    grad = grad * max_norm / ||grad||
```

### 6.2 Attention Gradient Computation

During backpropagation through the attention mechanism, the gradient flows through the softmax and into the score computation. The gradient of the attention output with respect to the queries is:

```
d(Attention)/d(Q) = (1/sqrt(d_k)) * dA/dS * K^T
```

where `S = QK^T / sqrt(d_k)` are the pre-softmax scores and `A = softmax(S)` are the attention weights. The Jacobian of softmax has the form:

```
dA_i/dS_j = A_i * (delta_{ij} - A_j)
```

This can lead to numerical instability when attention weights become very peaked (near 0 or 1). Flash Attention (Dao et al., 2022) addresses this by computing attention in tiled blocks with online normalization, avoiding materializing the full attention matrix and reducing memory usage from O(seq_len^2) to O(seq_len).

---

## 7. Optimization Algorithms for Fine-Tuning

### 7.1 AdamW: The Standard Choice

AdamW is the de facto optimizer for transformer fine-tuning. It combines adaptive moment estimation (Adam) with decoupled weight decay. For parameter `theta_i` at step `t`:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (first moment / momentum)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (second moment / adaptive scaling)
m_hat_t = m_t / (1 - beta1^t)                       (bias correction)
v_hat_t = v_t / (1 - beta2^t)                       (bias correction)
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps) - lr * lambda * theta_{t-1}
```

The key difference from Adam is the final term: weight decay is applied directly to the parameters (`lambda * theta`), not folded into the gradient. This separation ensures weight decay acts as L2 regularization rather than modifying the adaptive step size, which is critical for preventing overfitting in fine-tuning.

Recommended defaults for SFT: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `lambda=0.01`.

### 7.2 Learning Rate Schedule

The learning rate schedule for SFT typically consists of:

**Linear Warmup**: The learning rate increases linearly from 0 to `lr_max` over the first `W` steps. This is critical because early in fine-tuning, the gradient estimates are noisy (small batch, random direction). Starting with a large learning rate would cause destructive updates to the pre-trained weights.

```
lr(t) = lr_max * t / W   for t < W
```

**Cosine Decay**: After warmup, the learning rate follows a cosine annealing schedule from `lr_max` to `lr_min`:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * (t - W) / (T - W)))   for t >= W
```

Typical values: `lr_max = 1e-5` to `2e-4`, `lr_min = lr_max / 10`, `W = 3-10%` of total steps.

### 7.3 Paged AdamW for Memory Efficiency

For large models, the optimizer states (first and second moments) occupy 2x the memory of the model weights in fp32. For a 7B parameter model, this is `7B * 3 * 4 bytes = 84GB` just for optimizer states. Paged AdamW (from bitsandbytes) offloads optimizer states to CPU RAM using memory paging, allowing fine-tuning of larger models on consumer hardware.

---

## 8. Parameter-Efficient Fine-Tuning (PEFT) vs Full Fine-Tuning

### 8.1 Full Fine-Tuning

Full fine-tuning updates all model parameters. It produces the highest quality fine-tuned model for a given dataset and training budget but has significant drawbacks:

- Memory: Requires storing model weights, gradients, and optimizer states (approximately 16x model size in bytes for fp32 Adam).
- Catastrophic forgetting: Large parameter updates can overwrite knowledge from pre-training that is not represented in the fine-tuning dataset.
- Storage: Each fine-tuned model requires storing a full copy of the weights.

### 8.2 Parameter-Efficient Methods

PEFT methods update only a small fraction of parameters while freezing the rest. The dominant approach for LLMs is LoRA (Low-Rank Adaptation), which is described in detail in the next section. Other methods include:

**Prefix Tuning**: Prepends a set of trainable "soft" tokens (prefix) to the input or key/value sequences of every attention layer. Only the prefix token embeddings are trained.

**Prompt Tuning**: A simpler variant that prepends trainable tokens only to the input embedding layer.

**Adapter Layers**: Inserts small bottleneck FFN modules (down-project, nonlinearity, up-project) after each attention and FFN block, training only the adapter parameters.

---

## 9. LoRA: Mathematical Derivation and Implementation

### 9.1 The Core Idea

Low-Rank Adaptation (LoRA) is based on the hypothesis that the weight updates during fine-tuning have low intrinsic rank. That is, the matrix of weight changes `Delta_W` for each weight matrix `W` can be well-approximated by a low-rank decomposition:

```
Delta_W = B * A,   where B in R^{d_out x r}, A in R^{r x d_in}, r << min(d_in, d_out)
```

Instead of learning `Delta_W` directly (which would require updating all `d_in * d_out` parameters), we learn the low-rank factors `A` and `B`, requiring only `r * (d_in + d_out)` parameters. For `r=8`, `d_in=d_out=4096`, this is:

```
Full update:  4096 * 4096 = 16,777,216 parameters
LoRA update:  8 * (4096 + 4096) = 65,536 parameters (0.39% of full)
```

### 9.2 LoRA Forward Pass

During fine-tuning, the adapted weight matrix is:

```
W_adapted = W_frozen + (alpha/r) * B * A
```

where `alpha` is a scaling hyperparameter (typically set equal to `r`). The factor `alpha/r` normalizes the learning rate with respect to `r`, so changing `r` without changing `alpha` does not require re-tuning the learning rate.

`A` is initialized from a Gaussian distribution and `B` is initialized to zero, ensuring that `Delta_W = B*A = 0` at the start of training and the model begins from the pre-trained checkpoint.

### 9.3 Applying LoRA to Which Matrices?

The original LoRA paper (Hu et al., 2021) applied adapters to the query and value projection matrices in attention. Subsequent work found that applying LoRA to all four attention projections (Q, K, V, O) and sometimes to the FFN matrices improves performance. The decision of which modules to target is a key hyperparameter.

For a model with embedding dimension `d=4096` and 32 layers, targeting Q and V only with `r=16` gives:
```
Parameters per layer: 2 * 16 * (4096 + 4096) = 262,144
Total LoRA params:    32 * 262,144 = 8,388,608 (~8.4M out of 7B total = 0.12%)
```

### 9.4 QLoRA: 4-bit Quantized LoRA

QLoRA (Dettmers et al., 2023) extends LoRA by quantizing the frozen base model weights to 4-bit NormalFloat (NF4) format while keeping the LoRA adapter weights in bfloat16. NF4 is an information-theoretically optimal quantization for normally distributed weights. The combination reduces VRAM usage by ~4x compared to bfloat16 LoRA while retaining most of the fine-tuning quality.

---

## 10. Data Engineering for SFT

### 10.1 The Alpaca Format and Chat Templates

The Alpaca format (Stanford, 2023) established a simple but effective structure:

```json
{
    "instruction": "Translate the following English text to French.",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
}
```

Modern instruction-tuned models use the ChatML format or model-specific chat templates. For Phi-2 or GPT-2 fine-tuning, a simple format is:

```
### Instruction:
{instruction}

### Response:
{response}
```

For models with tokenizer chat templates (LLaMA-3, Mistral, Phi-3):

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": response}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

### 10.2 Dataset Quality Guidelines

A high-quality SFT dataset exhibits the following properties:

**Correctness**: Responses must be factually accurate. Incorrect demonstrations teach the model wrong facts and are extremely difficult to unlearn.

**Consistency**: The response style, format, and level of detail should be consistent across the dataset. Inconsistent style forces the model to average across conflicting formats.

**Diversity of Tasks**: For general-purpose instruction following, the dataset should cover diverse task types: summarization, Q&A, code, reasoning, creative writing, extraction. The Open-Platypus paper (Lee et al., 2023) showed that 25,000 carefully curated examples can outperform much larger datasets.

**Response Length Distribution**: Very short responses (one word) and very long responses (thousands of tokens) should be monitored. An imbalanced length distribution can cause the model to generate outputs of systematically incorrect length.

---

## 11. Training Dynamics, Stability, and Hyperparameters

### 11.1 Batch Size

The effective batch size for SFT is typically 32-256 examples. Larger batch sizes provide more stable gradient estimates but require more memory and may find flatter minima (which can generalize better but may converge more slowly). For SLMs fine-tuned on a single GPU, gradient accumulation with per-step batch size 4-8 and accumulation steps 8-16 is standard.

### 11.2 Number of Training Epochs

SFT on high-quality datasets typically requires 1-5 epochs. Training for too many epochs causes catastrophic overfitting — the model memorizes training examples and loses the ability to generalize. Monitoring validation loss and stopping when it plateaus or increases is the most reliable strategy.

For very small datasets (< 1,000 examples), 3-5 epochs with dropout can work. For large datasets (> 100,000 examples), 1 epoch is often sufficient and multiple epochs risk memorization.

### 11.3 Sequence Length and Memory

The memory consumption of attention is `O(seq_len^2)` due to the attention matrix. For a sequence length of 2048 with bfloat16:

```
Attention matrix per layer per head: 2048 * 2048 * 2 bytes = 8MB
With 32 layers, 32 heads: 32 * 32 * 8MB = 8GB just for attention matrices
```

Flash Attention 2 reduces this to `O(seq_len)` by computing attention in tiles, enabling training with much longer sequences.

### 11.4 Regularization Strategies

**Dropout**: Applying dropout (rate 0.05-0.1) to the attention weights and FFN hidden layers provides regularization. Pre-trained models often have dropout disabled (rate 0.0), so enabling it during fine-tuning is a regularization technique.

**Weight Decay**: As noted in Section 7.1, AdamW's weight decay acts as L2 regularization on the weights themselves. Values of 0.01-0.1 are typical.

**Data Augmentation**: For instruction-following, paraphrasing the instruction while preserving the semantics increases effective dataset size. This can be done with another LLM or rule-based transformations.

---

## 12. Evaluation Methodology

### 12.1 Automatic Metrics

**Perplexity**: Measures how well the model predicts held-out completion tokens. Lower is better but does not directly measure task performance or output quality.

**ROUGE**: Recall-Oriented Understudy for Gisting Evaluation. Measures n-gram overlap between generated output and reference. ROUGE-1, ROUGE-2, and ROUGE-L are reported for summarization tasks.

**BLEU**: Bilingual Evaluation Understudy. Measures modified precision of n-grams. Standard for machine translation but less common for open-ended generation.

**BERTScore**: Computes semantic similarity between generated and reference texts using BERT embeddings. More robust to paraphrasing than n-gram metrics.

### 12.2 LLM-as-Judge Evaluation

For instruction-following and open-ended generation, automatic n-gram metrics correlate poorly with human judgment. A scalable alternative is using a frontier LLM (GPT-4, Claude) as a judge. The judging prompt asks the evaluator to rate the fine-tuned model's response on dimensions like helpfulness, accuracy, and conciseness, producing a score from 1 to 10. This correlates well with human evaluations at much lower cost.

### 12.3 Benchmark Suites

For SLMs, standard benchmarks include:
- **MT-Bench**: 80 multi-turn questions across reasoning, coding, math, writing, roleplay.
- **MMLU**: 57-subject multiple-choice questions testing world knowledge.
- **HumanEval**: Python coding benchmarks with unit test execution.
- **TruthfulQA**: Tests the model's tendency to generate false but plausible-sounding answers.

---

## 13. SFT with a Small Language Model (SLM): Practical Walkthrough

This section explains how to apply the concepts above to fine-tuning GPT-2 (117M parameters) or Phi-2 (2.7B), which are the most accessible SLMs for educational purposes.

### 13.1 Why GPT-2 for Learning SFT?

GPT-2 is ideal for learning SFT because:
- It is small enough to fine-tune on a single consumer GPU (even CPU for the 117M version).
- Its architecture (decoder-only transformer) is identical to GPT-3, GPT-4, and Llama in structure.
- The tokenizer (BPE with 50,257 vocabulary) is well-understood.
- The model weights are fully open and widely studied.

Every concept demonstrated on GPT-2 — loss masking, gradient flow, LoRA adapters, learning rate scheduling — applies identically to a 70B parameter model. The only differences are scale and the number of GPUs required.

### 13.2 The Training Data Format for GPT-2

GPT-2 does not have a built-in instruction format (it was pre-trained on raw web text). For SFT, we impose a format:

```
Below is an instruction that describes a task. Write a response that completes it.

### Instruction:
{instruction}

### Response:
{response}
```

The tokenizer does not have special chat tokens, so we use these literal text markers. The loss is computed only on tokens after and including `### Response:\n`.

### 13.3 Memory Footprint Analysis

For GPT-2 small (117M parameters) in fp32 full fine-tuning:
```
Model weights:    117M * 4 bytes = 468 MB
Gradients:        117M * 4 bytes = 468 MB (same size as weights)
AdamW m (fp32):  117M * 4 bytes = 468 MB
AdamW v (fp32):  117M * 4 bytes = 468 MB
Activations:      Depends on batch size and sequence length, typically ~200MB per batch
Total:            ~2 GB
```

This fits comfortably in an 8GB GPU, even with batch size 8 and sequence length 512.

### 13.4 Expected Training Dynamics

When fine-tuning GPT-2 on a 5,000-example instruction-following dataset:

- **Epoch 1**: Training loss drops from ~4.0 to ~1.5. Validation loss follows. The model learns the response format.
- **Epoch 2**: Training loss drops to ~0.8. Validation loss may start to diverge slightly from training loss.
- **Epoch 3**: Training loss approaches ~0.3 but validation loss plateaus or increases — early stopping should trigger.

These numbers will vary significantly with dataset quality. High-quality, consistent datasets will show faster convergence and smaller train-validation gaps.

---

## 14. Production Considerations

### 14.1 Catastrophic Forgetting Mitigation

The primary risk of SFT is that aggressively updating all parameters on a narrow distribution can overwrite general knowledge. Mitigation strategies include:

**Low Learning Rate**: Using a learning rate 10x-100x smaller than pre-training keeps parameter changes small.

**LoRA / PEFT**: By freezing most weights, PEFT prevents catastrophic forgetting by construction — the original weights are preserved and only the adapter deltas change.

**Replay**: Mixing a small fraction (5-10%) of the pre-training data into the SFT dataset maintains general capabilities.

**Elastic Weight Consolidation (EWC)**: Adds a regularization term that penalizes changes to parameters identified as important for previous tasks (using the Fisher information matrix). Computationally expensive but theoretically principled.

### 14.2 Inference Optimization Post-SFT

After SFT, the model is deployed for inference. Key optimizations:

**Weight Quantization**: Converting weights from bfloat16 to int8 or int4 reduces memory by 2x-4x with minimal quality degradation, using libraries like bitsandbytes, GPTQ, or AWQ.

**KV Cache**: During autoregressive generation, storing past key and value tensors prevents recomputation. The KV cache size is `2 * L * H * d_k * seq_len * precision_bytes`.

**vLLM / TGI**: Production inference servers use continuous batching (PagedAttention in vLLM) to serve many concurrent requests efficiently, dramatically increasing throughput versus naive batching.

### 14.3 Observability and Monitoring

Production SFT training should emit:
- **Training loss** at every step
- **Validation loss** at every evaluation interval
- **Gradient norm** before and after clipping
- **Learning rate** at every step
- **GPU memory utilization** (peak and current)
- **Training throughput** (tokens/second)
- **Sample generations** logged at every eval interval for qualitative monitoring

Tools: Weights & Biases (`wandb`), MLflow, TensorBoard.

---

## 15. Trade-offs, Best Practices, and Recommendations

### Core Best Practices

**Data quality dominates quantity.** A dataset of 1,000 carefully crafted, consistent, accurate examples will produce a better fine-tuned model than 100,000 examples with mixed quality, inconsistent formatting, and factual errors. Invest the majority of your SFT budget in dataset curation, not compute.

**Start with LoRA, not full fine-tuning.** For most use cases, LoRA with `r=16` targeting Q, K, V, O matrices and the FFN gate/up/down projections achieves 95%+ of full fine-tuning quality at 5% of the memory cost. Only graduate to full fine-tuning if evaluation shows LoRA is insufficient.

**Monitor for catastrophic forgetting.** Run your model on general-purpose benchmarks (MMLU, TruthfulQA) before and after fine-tuning. A significant regression on general tasks signals over-fitting or catastrophic forgetting.

**Use cosine learning rate decay with linear warmup.** This schedule consistently outperforms constant and linear decay schedules for fine-tuning. The warmup prevents destructive early updates and the cosine decay enables finding sharp, deep minima.

**Evaluate with LLM-as-judge in addition to perplexity.** Perplexity measures how well the model predicts held-out tokens, which does not directly measure output quality or task performance. Always complement quantitative metrics with qualitative assessment.

**Log sample outputs during training.** Numerical metrics can look healthy while the model is producing degenerate outputs (repetition, format violations, hallucinations). Logging a fixed set of generations every 100-200 steps catches these problems early.

### Key Trade-offs

**Full fine-tuning vs LoRA**: Full fine-tuning achieves higher peak quality but requires more memory, risks catastrophic forgetting, and requires storing complete model copies. LoRA is memory-efficient, preserves general knowledge, and requires only adapter storage (~10MB vs ~14GB for a 7B model), but caps maximum performance.

**More epochs vs early stopping**: More epochs decrease training loss but increase overfitting risk. For production SFT, always use a held-out validation set and stop training when validation loss stops improving.

**Larger rank r in LoRA vs smaller r**: Larger rank increases the expressivity of the adaptation but also the number of trainable parameters and the risk of overfitting. Start with `r=8` or `r=16` and increase only if the model underfits.

**Long context vs throughput**: Fine-tuning with longer sequence lengths enables the model to handle longer inputs but exponentially increases memory requirements and decreases training throughput. Use Flash Attention 2 and choose the minimum sequence length that covers your target use case.
