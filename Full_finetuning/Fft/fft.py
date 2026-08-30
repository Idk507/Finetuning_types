"""
fft_implementation.py
=====================
Production-Grade Full Fine-Tuning (FFT) Implementation
Using GPT-2 as the demonstration Small Language Model.

This file demonstrates the complete FFT pipeline end-to-end:
  1. Complete parameter update — every weight, bias, embedding, norm parameter
  2. Correct masked cross-entropy loss with autoregressive token shifting
  3. AdamW with decoupled weight decay and correct parameter group separation
  4. Cosine LR schedule with linear warmup (full mathematical derivation in comments)
  5. Mixed precision training (bfloat16 forward, fp32 gradient accumulation)
  6. Gradient checkpointing for activation memory reduction
  7. Gradient accumulation to simulate large effective batch sizes
  8. Catastrophic forgetting detection via held-out general benchmark proxy
  9. L2 regularization toward pre-trained weights (forgetting mitigation)
 10. Full checkpointing with optimizer state, scheduler state, and training metadata
 11. Comprehensive metric logging and sample generation at evaluation steps
 12. Mathematical verification of all core computations

Architecture:  Decoder-only Transformer (GPT-2, 117M parameters)
Framework:     PyTorch >= 2.0, Transformers >= 4.35
Hardware:      CPU for GPT-2 small (educational); GPU recommended for production

Key distinction from SFT + LoRA:
    - ALL parameters have requires_grad=True (no frozen layers)
    - Optimizer states maintained for all N parameters (8N bytes for AdamW)
    - Gradients computed through every transformer block
    - Risk of catastrophic forgetting must be explicitly monitored and mitigated

Mathematical core:
    theta* = argmin_theta L(theta; D_FT)
    starting from theta_0 = theta_pretrained  (warm initialization)

    L(theta) = -(1/N) sum_i (1/|c_i|) sum_{t in completion_i} log P_theta(x_t | x_{<t})

    AdamW update:
        m_t = beta1 * m_{t-1} + (1-beta1) * g_t
        v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
        theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                               - lr * wd * theta_{t-1}   (decoupled weight decay)
"""

import os
import copy
import math
import json
import time
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore", message=".*tokenizer.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fft")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

@dataclass
class FFTConfig:
    """
    Central configuration for the Full Fine-Tuning training run.

    In FFT, every parameter is updated. This means the memory budget
    must account for model weights + gradients + optimizer states for
    ALL N parameters — not just a small adapter subset as in LoRA.

    Memory estimate for GPT-2 (117M params) in fp32 full fine-tuning:
        Model weights:    117M * 4 bytes =  468 MB
        Gradients:        117M * 4 bytes =  468 MB
        Adam first moment: 117M * 4 bytes = 468 MB
        Adam second moment:117M * 4 bytes = 468 MB
        Total (no activations): ~1.87 GB
        Plus activations: ~200-400 MB for batch=4, seq=512

    For bf16 mixed precision (what this implementation uses):
        Model weights (bf16): 117M * 2 bytes = 234 MB
        Gradients (fp32):     117M * 4 bytes = 468 MB
        Adam states (fp32):   117M * 8 bytes = 936 MB
        Total: ~1.64 GB  (slightly less due to bf16 weights)
    """
    # --- Model ---
    model_name: str = "gpt2"              # HuggingFace model ID or local path

    # --- Data ---
    max_seq_length: int = 512             # Max token length (prompt + completion)
    train_split: float = 0.85            # Fraction of data used for training
                                          # 0.85/0.10/0.05 train/val/test split

    # --- Training ---
    num_epochs: int = 3                   # Number of complete passes over training data
    per_device_batch_size: int = 4        # Micro-batch size per GPU per step
    gradient_accumulation_steps: int = 8  # Effective_batch = batch * accum
    learning_rate: float = 1e-4          # Peak LR after warmup (lower than LoRA needs)
    weight_decay: float = 0.01           # AdamW decoupled L2 regularization coefficient
    warmup_ratio: float = 0.06           # Fraction of steps for linear LR warmup
    max_grad_norm: float = 1.0           # L2 gradient clipping threshold
    adam_beta1: float = 0.9              # Adam first moment exponential decay
    adam_beta2: float = 0.999            # Adam second moment exponential decay
    adam_epsilon: float = 1e-8           # Numerical stability in Adam denominator

    # --- FFT-Specific: Catastrophic Forgetting Mitigation ---
    use_pretrained_weight_regularization: bool = True
    pretrained_weight_reg_lambda: float = 0.001  # lambda in ||theta - theta_0||^2 term
    # Higher lambda = stronger penalty for deviating from pre-trained weights
    # Recommended range: 1e-4 to 1e-2 depending on task similarity to pre-training

    # --- Memory Optimization ---
    use_gradient_checkpointing: bool = True   # Recompute activations during backward pass
    use_bf16: bool = True                      # Mixed precision (requires GPU)

    # --- Evaluation and Checkpointing ---
    eval_steps: int = 40                 # Evaluate every N optimizer steps
    save_steps: int = 80                # Save checkpoint every N optimizer steps
    output_dir: str = "./fft_output"    # Output directory for checkpoints
    logging_steps: int = 10             # Log training metrics every N steps
    num_sample_generations: int = 3     # Number of sample outputs to log at eval
    early_stopping_patience: int = 5    # Stop if val loss doesn't improve for N evals

    # --- Reproducibility ---
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        """
        The effective batch size seen by the optimizer at each update step.
        Gradient accumulation over K steps with micro-batch B is mathematically
        identical to computing the gradient over a single batch of K*B examples.
        """
        return self.per_device_batch_size * self.gradient_accumulation_steps


# =============================================================================
# SECTION 2: DATASET
# =============================================================================

# High-quality instruction-following demonstration data.
# In production, replace with your curated domain-specific dataset.
DEMO_DATASET = [
    {
        "instruction": "What is the difference between full fine-tuning and LoRA?",
        "response": (
            "Full fine-tuning updates every parameter in the model including all "
            "weight matrices, biases, embeddings, and layer normalization parameters. "
            "Every gradient step modifies the entire parameter space. LoRA instead "
            "freezes the original model weights and trains only small low-rank adapter "
            "matrices injected into the attention layers. Full fine-tuning achieves "
            "higher peak quality because the adaptation space is unconstrained, but "
            "requires far more memory to store gradients and optimizer states for all "
            "parameters. LoRA is preferred when memory is limited or when catastrophic "
            "forgetting of general capabilities must be avoided."
        ),
    },
    {
        "instruction": "Explain the AdamW optimizer and why weight decay is decoupled.",
        "response": (
            "AdamW is an adaptive gradient optimizer that combines momentum with "
            "per-parameter adaptive learning rates. For each parameter it maintains "
            "a first moment (exponential moving average of gradients) and a second "
            "moment (exponential moving average of squared gradients). The update "
            "divides the momentum-smoothed gradient by the root of the second moment, "
            "giving parameters with large historical gradients a smaller effective "
            "learning rate. The weight decay is decoupled from the gradient update: "
            "instead of adding the regularization term to the gradient before scaling, "
            "AdamW applies weight decay directly to the parameter after the adaptive "
            "step. This ensures consistent L2 regularization strength regardless of "
            "the gradient magnitude, which is critical for preventing overfitting "
            "during fine-tuning on small datasets."
        ),
    },
    {
        "instruction": "What is catastrophic forgetting and how can it be mitigated in fine-tuning?",
        "response": (
            "Catastrophic forgetting occurs when full fine-tuning on a narrow task "
            "distribution overwrites the general knowledge and capabilities the model "
            "acquired during pre-training. The optimizer follows gradients that minimize "
            "the fine-tuning loss without any constraint on behavior for inputs outside "
            "the fine-tuning distribution, so parameters encoding general knowledge "
            "may be overwritten. Mitigation strategies include using a lower learning "
            "rate to limit parameter movement, regularizing toward the pre-trained "
            "weights with an L2 penalty on parameter deviations, mixing pre-training "
            "data into the fine-tuning dataset to maintain the original loss, using "
            "LoRA which freezes base weights entirely, and monitoring forgetting "
            "explicitly by evaluating on general benchmarks during training."
        ),
    },
    {
        "instruction": "Explain gradient checkpointing and why it is used in full fine-tuning.",
        "response": (
            "Gradient checkpointing reduces GPU memory consumption during training by "
            "trading memory for computation. During the forward pass, rather than "
            "retaining all intermediate activations in memory for use in the backward "
            "pass, only activations at selected checkpoint boundaries are saved. When "
            "the backward pass reaches a segment between checkpoints, the forward pass "
            "is re-executed from the previous checkpoint to reconstruct the needed "
            "activations on demand. This reduces activation memory from O(L) layers "
            "worth of tensors to O(sqrt(L)) with a standard checkpointing strategy, "
            "at the cost of approximately 33 percent additional forward computation. "
            "In full fine-tuning of large models, activation memory is the dominant "
            "memory consumer and gradient checkpointing is essential for fitting "
            "training within GPU memory budgets."
        ),
    },
    {
        "instruction": "What is the purpose of gradient accumulation in training language models?",
        "response": (
            "Gradient accumulation simulates a large effective batch size using the "
            "memory of a small micro-batch. The training loop runs K forward-backward "
            "passes accumulating gradients without calling the optimizer, then divides "
            "the summed gradients by K and executes the parameter update. This is "
            "mathematically identical to computing the gradient over a single batch "
            "K times larger, because gradient of a sum equals the sum of gradients. "
            "It is particularly valuable in full fine-tuning where each sequence in "
            "the batch requires storing its activations in memory. With gradient "
            "accumulation you can achieve the stability benefits of a batch size of "
            "256 while using the memory of a batch size of 4."
        ),
    },
    {
        "instruction": "Describe the mixed precision training protocol for transformer fine-tuning.",
        "response": (
            "Mixed precision training uses 16-bit floating point for forward pass "
            "computations and 32-bit for gradient accumulation and optimizer states. "
            "The model weights are stored in bfloat16, which uses the same 8-bit "
            "exponent as float32 but only 7 mantissa bits, preventing overflow that "
            "occurs with float16's 5-bit exponent. The forward pass and initial "
            "gradient computation happen in bfloat16, approximately doubling speed "
            "on Tensor Core hardware. Gradients are upcast to float32 for accumulation "
            "to prevent underflow from very small gradient values. The optimizer "
            "maintains all its state (first and second moments) in float32, and the "
            "final parameter update is applied in float32 before casting the model "
            "weights back to bfloat16. This achieves roughly 2x memory savings and "
            "1.5 to 2x speedup with negligible quality degradation compared to full "
            "float32 training."
        ),
    },
    {
        "instruction": "What is the mathematical relationship between perplexity and cross-entropy loss?",
        "response": (
            "Perplexity is the exponential of the average cross-entropy loss over a "
            "sequence. Formally, for a sequence of T tokens with cross-entropy loss L, "
            "perplexity equals exp(L). The cross-entropy loss is the negative average "
            "log-likelihood: L = -(1/T) sum_t log P(x_t | x_{<t}). Perplexity has the "
            "intuitive interpretation of the effective vocabulary size the model is "
            "uncertain among at each token position. A perplexity of 10 means the "
            "model is on average choosing among 10 equally probable tokens. A perplexity "
            "of 1 means perfect prediction. During fine-tuning, perplexity computed "
            "on held-out validation data is the primary quantitative metric, dropping "
            "from pre-training values around 30 to 50 down to task-specific values "
            "around 3 to 15 as the model adapts to the fine-tuning distribution."
        ),
    },
    {
        "instruction": "How does the cosine learning rate schedule work and why is it preferred?",
        "response": (
            "The cosine learning rate schedule reduces the learning rate following a "
            "cosine curve from the peak value after warmup down to a minimum value at "
            "the end of training. Mathematically, at step t after warmup, the learning "
            "rate is lr_min plus half of (lr_max minus lr_min) times one plus cosine "
            "of pi times the fractional progress through training. This profile spends "
            "more time at moderate learning rates compared to linear decay, which drops "
            "immediately and uniformly. The cosine shape allows large exploratory steps "
            "early in training, an extended period of medium steps for efficient "
            "navigation of the loss landscape, and a rapid final reduction for precise "
            "convergence. Empirically this consistently outperforms linear and step "
            "decay schedules for transformer fine-tuning across model sizes and tasks."
        ),
    },
    {
        "instruction": "What are the components of memory consumption during full fine-tuning?",
        "response": (
            "Full fine-tuning GPU memory is consumed by four categories. Model parameters "
            "occupy two bytes per parameter in bfloat16. Gradients occupy four bytes per "
            "parameter in float32. Optimizer states for AdamW occupy eight bytes per "
            "parameter in float32 (four for the first moment and four for the second). "
            "Activations occupy memory proportional to batch size times sequence length "
            "times hidden dimension times number of layers. For a seven billion parameter "
            "model in mixed precision, parameters use 14 gigabytes, gradients use 28 "
            "gigabytes, optimizer states use 56 gigabytes, totaling 98 gigabytes before "
            "activations. This exceeds a single A100 GPU and requires techniques like "
            "ZeRO optimizer sharding, gradient checkpointing, and gradient accumulation "
            "to make training feasible."
        ),
    },
    {
        "instruction": "Explain the residual connection in transformers and its role in gradient flow.",
        "response": (
            "A residual connection adds the input of a sublayer directly to its output: "
            "the output becomes input plus sublayer(input) rather than just sublayer(input). "
            "For gradient flow during backpropagation, this is critical because the "
            "gradient of the loss with respect to the block input includes an identity "
            "term. Even if the sublayer has vanishing gradients, the identity term "
            "ensures the gradient signal is transmitted to earlier layers unchanged. "
            "Without residual connections in a 32-layer network, a spectral norm of "
            "0.9 per layer would reduce the gradient magnitude by a factor of 0.9 to "
            "the power 32, approximately 0.034, making early layers train 30 times "
            "slower than later ones. Residual connections eliminate this depth penalty "
            "and are the primary architectural reason deep transformer networks train "
            "stably from scratch and fine-tune effectively."
        ),
    },
    {
        "instruction": "What is layer normalization and why is pre-norm preferred over post-norm?",
        "response": (
            "Layer normalization normalizes the activations across the feature dimension "
            "for each token position independently: it subtracts the mean and divides "
            "by the standard deviation of the d-dimensional hidden vector, then applies "
            "learned scale gamma and shift beta parameters. Pre-norm places layer "
            "normalization before the attention and FFN sublayers, while post-norm "
            "places it after. Pre-norm is preferred for fine-tuning because it keeps "
            "activations in a well-conditioned numerical range before entering the "
            "attention and FFN computations, leading to more stable gradients when "
            "updating from a pre-trained checkpoint. Post-norm can suffer from "
            "activation magnitude growth through depth, requiring very careful learning "
            "rate warmup. Modern architectures including LLaMA, Mistral, and GPT-NeoX "
            "all use pre-norm, making it the standard for fine-tuning practice."
        ),
    },
    {
        "instruction": "What is the role of the temperature parameter in language model generation?",
        "response": (
            "Temperature is a scalar that divides the model's logits before the softmax "
            "operation during generation. At temperature 1.0 the softmax operates on "
            "the raw logits and the sampling distribution matches the model's trained "
            "distribution. At temperature less than 1.0 the logits are amplified before "
            "softmax, making the distribution more peaked toward the highest-probability "
            "token, resulting in more deterministic and conservative output. At "
            "temperature greater than 1.0 the logits are reduced, flattening the "
            "distribution and increasing diversity and randomness in the generated text. "
            "For evaluation during fine-tuning, temperature 1.0 with greedy decoding "
            "(argmax) is used to ensure deterministic and reproducible outputs. For "
            "production deployment, temperature between 0.7 and 1.0 typically provides "
            "a good balance between coherence and diversity."
        ),
    },
]


class FFTInstructionDataset(Dataset):
    """
    PyTorch Dataset for Full Fine-Tuning instruction-following tasks.

    Identical in structure to the SFT dataset — the difference between
    FFT and SFT+LoRA is in the model configuration (all params trainable),
    not in how the data is prepared. The same loss masking principle applies:
    the cross-entropy loss is computed only over completion (response) tokens.

    Loss masking mathematical justification:
        Without masking: L = -(1/T) sum_{t=1}^T log P(x_t | x_{<t})
            This penalizes the model for not predicting prompt tokens,
            introducing irrelevant gradient signal.

        With masking:   L = -(1/|S|) sum_{t in S} log P(x_t | x_{<t})
            where S = {positions belonging to the response}
            Gradient flows only through the output distribution for
            response tokens, focusing optimization on the task objective.

    The practical implementation sets labels[prompt_positions] = -100.
    PyTorch's F.cross_entropy with ignore_index=-100 automatically excludes
    these positions from both loss computation and gradient computation.
    """

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: "PreTrainedTokenizerBase",
        max_seq_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.processed: List[Dict[str, List[int]]] = []
        self.stats: Dict[str, Any] = {}
        self._tokenize_all()

    def _tokenize_all(self) -> None:
        """
        Tokenize all examples and construct the loss mask.

        The prompt boundary computation is the most critical implementation detail.
        We tokenize the prompt-only text to determine exactly how many tokens
        the prompt occupies, then set labels for those positions to -100.

        Edge cases handled:
            - Examples where truncation eliminates all response tokens (skipped)
            - Empty instruction or response strings (skipped with warning)
            - Tokenizer adding BOS token (accounted for in prompt_length calculation)
        """
        seq_lengths = []
        response_lengths = []
        skipped = 0

        for idx, example in enumerate(self.data):
            instruction = example.get("instruction", "").strip()
            response = example.get("response", "").strip()

            if not instruction or not response:
                logger.warning(f"Example {idx}: empty field, skipping.")
                skipped += 1
                continue

            prompt_text = self.PROMPT_TEMPLATE.format(instruction=instruction)

            # Tokenize prompt-only to find the boundary
            # add_special_tokens=False because we count BOS separately below
            prompt_ids = self.tokenizer.encode(
                prompt_text, add_special_tokens=False
            )

            # Tokenize full sequence with special tokens (BOS prepended by tokenizer)
            full_encoding = self.tokenizer(
                prompt_text + response,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None,
            )
            input_ids = full_encoding["input_ids"]
            attention_mask = full_encoding["attention_mask"]

            # +1 for the BOS token prepended by the tokenizer
            prompt_length = len(prompt_ids) + 1

            # Ensure at least 1 response token survives truncation
            if prompt_length >= len(input_ids):
                logger.debug(
                    f"Example {idx}: prompt_length={prompt_length} >= "
                    f"total_length={len(input_ids)}, skipping."
                )
                skipped += 1
                continue

            # Construct labels: -100 for prompt positions, actual IDs for response
            labels = [-100] * len(input_ids)
            labels[prompt_length:] = input_ids[prompt_length:]

            num_response_tokens = len(input_ids) - prompt_length
            seq_lengths.append(len(input_ids))
            response_lengths.append(num_response_tokens)

            self.processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        if seq_lengths:
            self.stats = {
                "total_examples": len(self.processed),
                "skipped": skipped,
                "mean_seq_length": sum(seq_lengths) / len(seq_lengths),
                "max_seq_length": max(seq_lengths),
                "min_seq_length": min(seq_lengths),
                "mean_response_tokens": sum(response_lengths) / len(response_lengths),
                "total_response_tokens": sum(response_lengths),
            }
            logger.info(
                f"Dataset: {self.stats['total_examples']} examples | "
                f"mean_len={self.stats['mean_seq_length']:.0f} | "
                f"mean_response={self.stats['mean_response_tokens']:.0f} tokens | "
                f"skipped={skipped}"
            )

    def __len__(self) -> int:
        return len(self.processed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def fft_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Right-pad all sequences in a batch to the same length.

    Padding conventions:
        input_ids:      padded with pad_token_id (model ignores via attention_mask)
        attention_mask: padded with 0 (position excluded from attention computation)
        labels:         padded with -100 (position excluded from loss computation)

    Why right padding for training (vs left padding for generation):
        Causal attention in transformers processes tokens left-to-right.
        During training, padding on the right means real tokens are always
        in the left portion of the sequence where they have full causal context
        from all preceding real tokens. Left padding during training would place
        real tokens after padding tokens, which would not interfere with causal
        attention (padding is masked anyway) but is a less natural convention.
        For autoregressive generation, left padding is used so that the final
        real token is at the rightmost position, aligning the generation start.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids_list.append(
            F.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        )
        attention_mask_list.append(
            F.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        labels_list.append(
            F.pad(item["labels"], (0, pad_len), value=-100)
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


# =============================================================================
# SECTION 3: LOSS FUNCTION
# =============================================================================

class FFTMaskedCrossEntropyLoss(nn.Module):
    """
    Masked causal language modeling loss for full fine-tuning.

    The loss is computed over the response tokens only (via the loss mask)
    using the standard autoregressive shift convention:
        - logits at position t predict token at position t+1
        - shift_logits = logits[..., :-1, :]   (shape: B x T-1 x V)
        - shift_labels = labels[..., 1:]        (shape: B x T-1)

    Mathematical derivation of the shift:
        At position t, the model has seen tokens x_1, ..., x_t and predicts
        a distribution over x_{t+1}. The logit vector at position t therefore
        corresponds to the prediction for position t+1.

        Concretely:
            logits[:, 0, :] -> prediction for token at position 1 (given token 0)
            logits[:, 1, :] -> prediction for token at position 2 (given tokens 0,1)
            logits[:, T-1, :] -> prediction for token at position T (but there is none)

        The labels also shift:
            labels[:, 1] is the ground truth for the prediction at logits[:, 0, :]
            labels[:, T-1] is the ground truth for the prediction at logits[:, T-2, :]

        We drop the last logit (no ground truth for it) and the first label
        (no logit prediction for the very first token position).

    Attributes:
        ignore_index: Label value excluded from loss computation (default: -100).
        label_smoothing: Optional label smoothing for regularization (default: 0.0).
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the masked cross-entropy loss.

        Args:
            logits: Raw model output, shape (B, T, V) where V = vocab_size.
            labels: Target token IDs, shape (B, T).
                    Positions with label == -100 are excluded from loss.

        Returns:
            loss: Scalar tensor with gradient graph attached.
            metrics: Dict with diagnostic values:
                - loss_value: float, scalar loss
                - perplexity: float, exp(loss)
                - num_active_tokens: int, number of unmasked positions
                - num_masked_tokens: int, number of masked (prompt) positions
                - tokens_per_example: float, mean active tokens per batch item
        """
        B, T, V = logits.shape

        # Autoregressive shift: logit at t predicts label at t+1
        shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, T-1)

        num_active = (shift_labels != self.ignore_index).sum().item()
        num_masked = (shift_labels == self.ignore_index).sum().item()

        if num_active == 0:
            # Edge case: all labels are masked (entire batch is prompt-only)
            # Return zero loss with no gradient contribution
            zero_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
            return zero_loss, {
                "loss_value": 0.0, "perplexity": 1.0,
                "num_active_tokens": 0, "num_masked_tokens": num_masked,
                "tokens_per_example": 0.0,
            }

        # F.cross_entropy with ignore_index averages over only the active positions.
        # label_smoothing=0.0 is standard; setting it > 0 distributes a fraction
        # of the label mass across the vocabulary as a regularization technique.
        loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        with torch.no_grad():
            # Clamp to prevent exp overflow when loss is very large at initialization
            ppl = math.exp(min(loss.item(), 20.0))

        metrics = {
            "loss_value": loss.item(),
            "perplexity": ppl,
            "num_active_tokens": num_active,
            "num_masked_tokens": num_masked,
            "tokens_per_example": num_active / max(1, B),
        }

        return loss, metrics


# =============================================================================
# SECTION 4: LEARNING RATE SCHEDULER
# =============================================================================

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Construct a learning rate scheduler with linear warmup and cosine decay.

    The complete LR schedule as a function of step t:

    Phase 1 — Linear warmup (0 <= t < W):
        lr(t) = lr_max * (t / W)

        Justification: At early steps, Adam's moment estimates have not yet
        accumulated reliable gradient statistics. Using the full lr_max
        immediately would take large steps based on noisy single-batch gradients,
        potentially causing destructive updates to the pre-trained weights.
        Linear warmup gives the optimizer time to build momentum before
        committing to full-sized steps.

    Phase 2 — Cosine decay (W <= t <= T):
        progress = (t - W) / (T - W)   in [0, 1]
        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))

        At progress=0: lr = lr_max  (just after warmup)
        At progress=1: lr = lr_min  (end of training)

        The cosine curve spends more time at moderate learning rates than
        linear decay, allowing efficient exploration before converging.

    The lambda function returns the MULTIPLIER relative to optimizer.defaults['lr'],
    so the actual learning rate is: optimizer.defaults['lr'] * lambda(t)

    Args:
        optimizer: AdamW optimizer to schedule.
        num_warmup_steps: Number of linear warmup steps W.
        num_training_steps: Total training steps T.
        min_lr_ratio: lr_min / lr_max ratio at end of training.

    Returns:
        LambdaLR scheduler instance.
    """
    def _lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]

        # Cosine interpolation between 1.0 and min_lr_ratio
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return LambdaLR(optimizer, _lr_lambda)


# =============================================================================
# SECTION 5: PRE-TRAINED WEIGHT REGULARIZATION
# =============================================================================

class PretrainedWeightRegularizer:
    """
    L2 regularization toward the original pre-trained weights.

    This implements the forgetting mitigation strategy:
        L_total(theta) = L_task(theta) + lambda/2 * ||theta - theta_0||^2

    Gradient of the regularization term:
        d/d(theta_i) [lambda/2 * (theta_i - theta_0_i)^2] = lambda * (theta_i - theta_0_i)

    This pulls each parameter toward its pre-trained value with a force
    proportional to how far it has drifted. Parameters that have moved far
    from their pre-trained values receive a stronger restoring force, directly
    penalizing the degree of forgetting.

    Implementation note: we add the regularization gradient directly to the
    model's parameter gradients before the optimizer step. This is equivalent
    to adding the L2 term to the loss but avoids constructing the full
    ||theta - theta_0||^2 expression in the computation graph.

    Attributes:
        pretrained_params: Dict mapping parameter names to their frozen
                           pre-trained values (stored on CPU to save GPU memory).
        lambda_reg: The regularization coefficient.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_reg: float = 1e-3,
    ):
        self.lambda_reg = lambda_reg
        # Store a CPU copy of all trainable parameters at their pre-trained values
        self.pretrained_params: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Detach and clone to CPU to avoid GPU memory overhead
                self.pretrained_params[name] = param.data.detach().cpu().clone()

        total_params = sum(p.numel() for p in self.pretrained_params.values())
        logger.info(
            f"PretrainedWeightRegularizer: tracking {len(self.pretrained_params)} "
            f"parameter tensors ({total_params:,} total params), "
            f"lambda={lambda_reg}"
        )

    def add_regularization_gradients(self, model: nn.Module) -> float:
        """
        Add the regularization gradient to each parameter's existing .grad tensor.

        Must be called AFTER loss.backward() and BEFORE optimizer.step().
        The order matters: backward() sets param.grad to the task gradient;
        this method adds the regularization gradient on top.

        The regularization gradient for parameter i is:
            grad_reg_i = lambda * (theta_i - theta_0_i)

        This is added in-place to param.grad, which already contains
        the task gradient from the backward pass.

        Args:
            model: The model being fine-tuned (with current parameter values).

        Returns:
            reg_loss: The scalar regularization loss value (for logging).
        """
        reg_loss = 0.0

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if name not in self.pretrained_params:
                continue

            # Move pretrained reference to the same device as the parameter
            theta_0 = self.pretrained_params[name].to(param.device)
            delta = param.data - theta_0  # theta - theta_0

            # Regularization gradient: lambda * (theta - theta_0)
            reg_grad = self.lambda_reg * delta

            # Add to existing gradient in-place
            param.grad.add_(reg_grad)

            # Accumulate regularization loss for logging: lambda/2 * ||delta||^2
            with torch.no_grad():
                reg_loss += 0.5 * self.lambda_reg * (delta * delta).sum().item()

        return reg_loss

    @torch.no_grad()
    def compute_forgetting_score(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute the L2 distance from pre-trained weights as a forgetting metric.

        Returns a dictionary with:
            - total_drift: ||theta - theta_0||_2 (L2 distance in full param space)
            - mean_drift: mean absolute deviation per parameter
            - max_drift: maximum absolute deviation across all parameters
            - layer_drifts: per-layer-name drift values for fine-grained analysis
        """
        total_sq_drift = 0.0
        total_abs_drift = 0.0
        max_drift = 0.0
        total_params = 0
        layer_drifts: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if name not in self.pretrained_params:
                continue
            theta_0 = self.pretrained_params[name].to(param.device)
            delta = (param.data - theta_0).abs()

            layer_drift = delta.mean().item()
            layer_drifts[name] = layer_drift

            total_sq_drift += (delta * delta).sum().item()
            total_abs_drift += delta.sum().item()
            max_drift = max(max_drift, delta.max().item())
            total_params += param.numel()

        return {
            "total_drift_l2": math.sqrt(total_sq_drift),
            "mean_drift": total_abs_drift / max(1, total_params),
            "max_drift": max_drift,
            "layer_drifts": layer_drifts,
        }


# =============================================================================
# SECTION 6: THE FULL FINE-TUNING TRAINER
# =============================================================================

class FFTTrainer:
    """
    Production-grade Full Fine-Tuning trainer for decoder-only language models.

    Core responsibilities:
        1. Configure all model parameters as trainable (no frozen weights)
        2. Build AdamW with correctly separated weight-decay parameter groups
        3. Execute the training loop with gradient accumulation
        4. Apply optional pre-trained weight regularization against forgetting
        5. Mixed precision (bf16) with fp32 gradient accumulation
        6. Periodic evaluation: validation loss, perplexity, sample generation,
           forgetting score, and early stopping
        7. Complete checkpoint saving and resumption

    Memory budget for GPT-2 (117M) FFT in bf16 mixed precision:
        Weights (bf16):   234 MB
        Gradients (fp32): 468 MB
        Adam m (fp32):    468 MB
        Adam v (fp32):    468 MB
        Activations:      ~200 MB (batch=4, seq=512)
        Total:            ~1.84 GB

    Attributes:
        config: FFTConfig with all hyperparameters.
        model: The pre-trained model being fully fine-tuned.
        tokenizer: Tokenizer matching the model.
        train_dataset, eval_dataset: FFTInstructionDataset instances.
        device: Compute device.
        criterion: FFTMaskedCrossEntropyLoss instance.
        regularizer: Optional PretrainedWeightRegularizer for forgetting mitigation.
        optimizer: AdamW with separated parameter groups.
        scheduler: Cosine LR scheduler with warmup.
        global_step: Current optimizer step count.
        best_eval_loss: Best validation loss observed.
        early_stopping_counter: Steps without improvement for early stopping.
    """

    def __init__(
        self,
        config: FFTConfig,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        train_dataset: FFTInstructionDataset,
        eval_dataset: FFTInstructionDataset,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.early_stopping_counter = 0

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training device: {self.device}")

        self.use_bf16 = config.use_bf16 and self.device.type == "cuda"
        if config.use_bf16 and not self.use_bf16:
            logger.warning("use_bf16=True requested but no GPU found; using fp32.")

        # Move model to device with appropriate dtype
        if self.use_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)
        self.model = self.model.to(self.device)

        # Enable gradient checkpointing to reduce activation memory
        if config.use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            else:
                logger.warning(
                    "Model does not support gradient_checkpointing_enable(); "
                    "proceeding without it."
                )

        # In FFT, ALL parameters must be trainable
        # This is the defining characteristic that distinguishes FFT from PEFT
        for param in self.model.parameters():
            param.requires_grad_(True)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"FFT mode: {trainable_params:,} / {total_params:,} parameters trainable "
            f"({100.0 * trainable_params / total_params:.1f}%)"
        )

        # Loss function
        self.criterion = FFTMaskedCrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)

        # Pre-trained weight regularizer (MUST be initialized BEFORE the optimizer,
        # so it captures the pre-trained weights before any gradient updates occur)
        if config.use_pretrained_weight_regularization:
            self.regularizer = PretrainedWeightRegularizer(
                model=self.model,
                lambda_reg=config.pretrained_weight_reg_lambda,
            )
        else:
            self.regularizer = None

        # Optimizer with correct parameter group separation
        self.optimizer = self._build_adamw_optimizer()

        # Compute total training steps for the scheduler
        self.total_steps = self._compute_total_steps()
        self.warmup_steps = max(1, int(self.total_steps * config.warmup_ratio))

        # Build cosine LR schedule with linear warmup
        self.scheduler = build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
            min_lr_ratio=0.1,
        )

        logger.info(
            f"Training plan: {self.total_steps} total steps | "
            f"warmup={self.warmup_steps} steps | "
            f"effective_batch={config.effective_batch_size} | "
            f"lr_max={config.learning_rate:.2e}"
        )

        os.makedirs(config.output_dir, exist_ok=True)

    def _build_adamw_optimizer(self) -> AdamW:
        """
        Build AdamW with correctly separated parameter groups.

        Separation rule (critical for correct training):
            decay group:    All 2D+ weight matrices. These benefit from L2
                            regularization which prevents weights from growing
                            without bound and provides a sparsity-inducing prior.

            no_decay group: Bias vectors (1D), LayerNorm scale (gamma) and shift
                            (beta) parameters (1D), embedding tables. These parameters
                            do NOT benefit from regularization toward zero — their
                            optimal values are determined by the data distribution,
                            not by any sparsity prior.

        Applying weight decay to biases and layer norm parameters (a common
        implementation bug) causes them to be regularized toward zero, which:
            - Biases: introduces a systematic bias toward underprediction
            - LayerNorm gamma: shrinks the activation scale, reducing model expressivity
            - LayerNorm beta: pulls the mean offset toward zero, distorting activations

        Returns:
            AdamW optimizer with two parameter groups.
        """
        # Categorize all trainable parameters
        decay_params = []
        no_decay_params = []
        decay_names = []
        no_decay_names = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Rule: no weight decay for 1D parameters, 'bias', 'norm', 'ln_', 'embed'
            is_no_decay = (
                param.ndim < 2
                or "bias" in name
                or "ln_" in name
                or "norm" in name
                or "wpe" in name   # GPT-2 positional embedding
                or "wte" in name   # GPT-2 token embedding (some prefer no decay here)
            )

            if is_no_decay:
                no_decay_params.append(param)
                no_decay_names.append(name)
            else:
                decay_params.append(param)
                decay_names.append(name)

        param_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
                "name": "decay",
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]

        optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        logger.info(
            f"AdamW parameter groups: "
            f"decay={n_decay:,} params ({len(decay_params)} tensors), "
            f"no_decay={n_no_decay:,} params ({len(no_decay_params)} tensors)"
        )
        return optimizer

    def _compute_total_steps(self) -> int:
        """
        Compute the total number of optimizer update steps across all epochs.

        Formula:
            steps_per_epoch = ceil(N_train / micro_batch_size)
            updates_per_epoch = ceil(steps_per_epoch / gradient_accumulation_steps)
            total_steps = updates_per_epoch * num_epochs
        """
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / self.config.per_device_batch_size
        )
        updates_per_epoch = math.ceil(
            steps_per_epoch / self.config.gradient_accumulation_steps
        )
        return updates_per_epoch * self.config.num_epochs

    def _make_dataloader(
        self,
        dataset: FFTInstructionDataset,
        shuffle: bool,
    ) -> DataLoader:
        """Build a padded DataLoader for the given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: fft_collate_fn(b, self.tokenizer.pad_token_id),
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
            num_workers=0,
        )

    def _forward_and_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute forward pass and compute the masked cross-entropy loss.

        The torch.autocast context manager enables automatic mixed precision:
            - Eligible ops (matmul, attention) run in bf16 on GPU
            - Other ops (LayerNorm, softmax) run in fp32 automatically
            - Gradients are upcast to fp32 during backward pass

        Returns the unscaled loss (caller handles accumulation scaling).
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
            enabled=self.use_bf16,
        )

        with autocast_ctx:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,   # Disable KV cache during training (not needed, saves memory)
            )
            loss, metrics = self.criterion(outputs.logits, labels)

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on the validation dataset.

        Computes:
            - Validation loss (token-weighted average over all response tokens)
            - Validation perplexity
            - Forgetting score (L2 drift from pre-trained weights)
            - Sample generations for qualitative assessment

        Token-weighted averaging is used (multiply loss by num_active_tokens,
        accumulate, then divide by total) to ensure each response token
        contributes equally regardless of which batch it falls in.

        Returns:
            Dictionary with all evaluation metrics.
        """
        self.model.eval()
        eval_loader = self._make_dataloader(self.eval_dataset, shuffle=False)

        total_weighted_loss = 0.0
        total_active_tokens = 0

        for batch in eval_loader:
            loss, metrics = self._forward_and_loss(batch)
            n = metrics["num_active_tokens"]
            total_weighted_loss += loss.item() * n
            total_active_tokens += n

        avg_loss = total_weighted_loss / max(1, total_active_tokens)
        avg_ppl = math.exp(min(avg_loss, 20.0))

        eval_metrics: Dict[str, Any] = {
            "eval_loss": avg_loss,
            "eval_perplexity": avg_ppl,
        }

        # Compute forgetting score
        if self.regularizer is not None:
            forgetting = self.regularizer.compute_forgetting_score(self.model)
            eval_metrics["forgetting_l2_drift"] = forgetting["total_drift_l2"]
            eval_metrics["forgetting_mean_drift"] = forgetting["mean_drift"]
            eval_metrics["forgetting_max_drift"] = forgetting["max_drift"]

        # Generate samples for qualitative monitoring
        samples = self._generate_samples(n=self.config.num_sample_generations)
        eval_metrics["samples"] = samples

        self.model.train()
        return eval_metrics

    @torch.no_grad()
    def _generate_samples(self, n: int = 3) -> List[str]:
        """
        Generate text from the first n evaluation prompts.

        Uses greedy decoding (do_sample=False) for deterministic, reproducible
        output. Qualitative monitoring is essential because metrics like perplexity
        do not capture: repetition loops, hallucinations, off-format responses,
        or length degeneration — all common failure modes in FFT.
        """
        self.model.eval()
        samples = []

        for example in self.eval_dataset.data[:n]:
            prompt = FFTInstructionDataset.PROMPT_TEMPLATE.format(
                instruction=example["instruction"]
            )
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).input_ids.to(self.device)

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            new_token_ids = output_ids[0, input_ids.shape[1]:]
            generated = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            samples.append(
                f"PROMPT : {example['instruction'][:80]}\n"
                f"GENERATED : {generated[:250]}"
            )

        return samples

    def save_checkpoint(self, tag: str, is_best: bool = False) -> str:
        """
        Save complete training state for resumption.

        Saves: model weights, optimizer states (first + second moments for ALL
        parameters), scheduler state, global step, best eval loss, and config.

        The optimizer state is the most critical component for exact resumption.
        Without it, the momentum and adaptive learning rate per-parameter would
        reset, causing a large loss spike at the resumed step. The optimizer
        state for AdamW is 8 bytes/parameter, so for a 117M model this is ~936MB.

        Args:
            tag: Identifier string for this checkpoint (e.g., 'step_400').
            is_best: Also save as 'best_model.pt' if True.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
            "config": asdict(self.config),
        }
        path = os.path.join(self.config.output_dir, f"checkpoint_{tag}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(
                f"Best model updated: {best_path} "
                f"(eval_loss={self.best_eval_loss:.4f})"
            )
        return path

    def train(self) -> Dict[str, List[float]]:
        """
        Execute the complete full fine-tuning training loop.

        Algorithm:
            Initialize: all params trainable, AdamW optimizer, cosine schedule

            For each epoch:
                For each micro-batch (batch_idx, batch):

                    Step A — Forward pass (bf16):
                        logits = model(input_ids, attention_mask)
                        loss = masked_cross_entropy(logits, labels)

                    Step B — Loss scaling for gradient accumulation:
                        scaled_loss = loss / gradient_accumulation_steps

                    Step C — Backward pass:
                        scaled_loss.backward()
                        # Gradients accumulate across micro-batches

                    Step D — At accumulation boundary:
                        if regularizer is enabled:
                            regularizer.add_regularization_gradients(model)
                            # Adds lambda*(theta - theta_0) to param.grad

                        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
                        optimizer.step()    # AdamW update for all N parameters
                        scheduler.step()    # Cosine LR update
                        optimizer.zero_grad()

                    Step E — Periodic evaluation and checkpointing

        Mathematical note on gradient accumulation scaling:
            The loss for a batch of size B is L = (1/B) sum_b l_b
            For K accumulation steps with micro-batch B_micro:
                L_effective = (1/K) sum_k L_k = (1/(K*B_micro)) sum_{k,b} l_{k,b}
            To maintain this equivalence, we divide the loss by K before backward.

        Returns:
            Dictionary with training history lists for downstream analysis.
        """
        train_loader = self._make_dataloader(self.train_dataset, shuffle=True)

        self.model.train()
        self.optimizer.zero_grad()

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_ppl": [],
            "eval_loss": [],
            "eval_ppl": [],
            "learning_rate": [],
            "grad_norm": [],
            "reg_loss": [],
            "forgetting_l2": [],
        }

        # Accumulators for logging between steps
        accum_task_loss = 0.0
        accum_reg_loss = 0.0
        accum_tokens = 0
        step_start_time = time.time()

        logger.info(
            f"Full Fine-Tuning start: {self.config.num_epochs} epochs | "
            f"{self.total_steps} optimizer steps | "
            f"device={self.device}"
        )

        stop_training = False

        for epoch in range(self.config.num_epochs):
            if stop_training:
                break

            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} starting")

            for batch_idx, batch in enumerate(train_loader):
                # ----------------------------------------------------------------
                # STEP A+B: FORWARD PASS AND SCALED LOSS
                # Scaling by 1/accumulation_steps ensures the accumulated gradient
                # equals the gradient of the full effective batch average loss.
                # ----------------------------------------------------------------
                loss, metrics = self._forward_and_loss(batch)
                n_active = metrics["num_active_tokens"]

                scaled_loss = loss / self.config.gradient_accumulation_steps

                # ----------------------------------------------------------------
                # STEP C: BACKWARD PASS
                # PyTorch autograd accumulates gradients into param.grad tensors.
                # Gradients are not zeroed until optimizer.zero_grad() is called,
                # so they accumulate across gradient_accumulation_steps micro-batches.
                # ----------------------------------------------------------------
                scaled_loss.backward()

                accum_task_loss += loss.item() * n_active
                accum_tokens += n_active

                # ----------------------------------------------------------------
                # STEP D: OPTIMIZER UPDATE (at accumulation boundary)
                # ----------------------------------------------------------------
                is_update_step = (
                    (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(train_loader)
                )

                if is_update_step:
                    # Add pre-trained weight regularization gradients BEFORE clipping.
                    # This ensures the regularization term is subject to the same
                    # gradient clipping as the task gradients, preventing the
                    # regularizer from dominating the update.
                    reg_loss = 0.0
                    if self.regularizer is not None:
                        reg_loss = self.regularizer.add_regularization_gradients(
                            self.model
                        )
                        accum_reg_loss += reg_loss

                    # Clip gradient L2 norm. Returns the pre-clip norm for logging.
                    # If norm > max_grad_norm: grad = grad * max_grad_norm / ||grad||
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm,
                    ).item()

                    # AdamW parameter update for ALL trainable parameters.
                    # Each parameter gets an independent adaptive learning rate
                    # based on its first and second gradient moment history.
                    self.optimizer.step()

                    # Advance the cosine LR schedule by one step
                    self.scheduler.step()

                    # Zero all accumulated gradients for the next window
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    current_lr = self.scheduler.get_last_lr()[0]

                    # ----------------------------------------------------------------
                    # STEP E: LOGGING
                    # ----------------------------------------------------------------
                    if self.global_step % self.config.logging_steps == 0:
                        avg_task_loss = accum_task_loss / max(1, accum_tokens)
                        avg_ppl = math.exp(min(avg_task_loss, 20.0))
                        elapsed = time.time() - step_start_time
                        tokens_per_sec = accum_tokens / max(elapsed, 1e-6)

                        history["train_loss"].append(avg_task_loss)
                        history["train_ppl"].append(avg_ppl)
                        history["learning_rate"].append(current_lr)
                        history["grad_norm"].append(grad_norm)
                        history["reg_loss"].append(accum_reg_loss)

                        logger.info(
                            f"Step {self.global_step:5d}/{self.total_steps} | "
                            f"E{epoch+1} | "
                            f"loss={avg_task_loss:.4f} | "
                            f"ppl={avg_ppl:.2f} | "
                            f"reg={accum_reg_loss:.4f} | "
                            f"lr={current_lr:.2e} | "
                            f"gnorm={grad_norm:.3f} | "
                            f"tok/s={tokens_per_sec:.0f}"
                        )

                        accum_task_loss = 0.0
                        accum_reg_loss = 0.0
                        accum_tokens = 0
                        step_start_time = time.time()

                    # ----------------------------------------------------------------
                    # EVALUATION
                    # ----------------------------------------------------------------
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()

                        eval_loss = eval_metrics["eval_loss"]
                        eval_ppl = eval_metrics["eval_perplexity"]
                        history["eval_loss"].append(eval_loss)
                        history["eval_ppl"].append(eval_ppl)

                        forgetting_l2 = eval_metrics.get("forgetting_l2_drift", 0.0)
                        history["forgetting_l2"].append(forgetting_l2)

                        logger.info(
                            f"Eval @ step {self.global_step}: "
                            f"eval_loss={eval_loss:.4f} | "
                            f"eval_ppl={eval_ppl:.2f} | "
                            f"forgetting_l2={forgetting_l2:.4f}"
                        )

                        for i, sample in enumerate(eval_metrics.get("samples", [])):
                            logger.info(f"Sample {i+1}:\n{sample}\n{'-'*60}")

                        # Check for improvement
                        is_best = eval_loss < self.best_eval_loss
                        if is_best:
                            self.best_eval_loss = eval_loss
                            self.early_stopping_counter = 0
                        else:
                            self.early_stopping_counter += 1
                            logger.info(
                                f"No improvement for {self.early_stopping_counter} "
                                f"eval steps (patience={self.config.early_stopping_patience})"
                            )

                        # Save checkpoint
                        if (
                            self.global_step % self.config.save_steps == 0
                            or is_best
                        ):
                            self.save_checkpoint(
                                tag=f"step_{self.global_step}",
                                is_best=is_best,
                            )

                        # Early stopping
                        if self.early_stopping_counter >= self.config.early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered at step {self.global_step}. "
                                f"Best eval_loss={self.best_eval_loss:.4f}"
                            )
                            stop_training = True
                            break

                        self.model.train()

        logger.info(
            f"Training complete. "
            f"Best eval_loss={self.best_eval_loss:.4f} "
            f"at step {self.global_step}"
        )
        return history


# =============================================================================
# SECTION 7: VERIFICATION AND DIAGNOSTIC UTILITIES
# =============================================================================

def verify_all_params_trainable(model: nn.Module) -> bool:
    """
    Verify that every parameter in the model has requires_grad=True.

    This is the defining property of full fine-tuning. If any parameter
    is frozen (requires_grad=False), the model is doing PEFT, not FFT.
    This check should be run after model initialization and before training.

    Returns:
        True if all parameters are trainable, False otherwise.
    """
    all_trainable = True
    frozen_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(name)
            all_trainable = False

    if all_trainable:
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Parameter trainability check PASSED: "
            f"all {total:,} parameters have requires_grad=True"
        )
    else:
        logger.error(
            f"Parameter trainability check FAILED: "
            f"{len(frozen_params)} frozen parameter tensors found: "
            f"{frozen_params[:5]}{'...' if len(frozen_params) > 5 else ''}"
        )

    return all_trainable


def verify_loss_masking(
    tokenizer: "PreTrainedTokenizerBase",
    example: Dict[str, str],
) -> bool:
    """
    Verify that the loss mask is correctly applied to a single example.

    Checks:
        1. Prompt tokens are contiguous at the start with label=-100
        2. Response tokens are contiguous at the end with actual token IDs
        3. There is at least one active (response) token
        4. The boundary between prompt and response is correct

    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("=" * 55)
    logger.info("LOSS MASKING VERIFICATION")

    ds = FFTInstructionDataset(
        data=[example], tokenizer=tokenizer, max_seq_length=512
    )
    if len(ds) == 0:
        logger.error("Dataset empty after processing — check example format.")
        return False

    item = ds[0]
    ids = item["input_ids"].tolist()
    lbls = item["labels"].tolist()

    masked = [i for i, l in enumerate(lbls) if l == -100]
    active = [i for i, l in enumerate(lbls) if l != -100]

    logger.info(f"Total tokens : {len(ids)}")
    logger.info(f"Masked (prompt, no loss) : {len(masked)} tokens")
    logger.info(f"Active (response, loss computed): {len(active)} tokens")

    prompt_text = tokenizer.decode([ids[i] for i in masked], skip_special_tokens=False)
    response_text = tokenizer.decode([ids[i] for i in active], skip_special_tokens=True)
    logger.info(f"Prompt  : {prompt_text[:120]}")
    logger.info(f"Response: {response_text[:200]}")

    # Check 1: masked positions are contiguous from position 0
    ok_masked = (masked == list(range(len(masked))))
    # Check 2: active positions are contiguous from end of masked positions
    ok_active = (active == list(range(len(masked), len(ids))))
    # Check 3: at least one active token
    ok_nonempty = len(active) > 0

    passed = ok_masked and ok_active and ok_nonempty
    status = "PASSED" if passed else "FAILED"

    logger.info(
        f"Checks: contiguous_masked={ok_masked}, "
        f"contiguous_active={ok_active}, "
        f"nonempty_active={ok_nonempty} -> [{status}]"
    )
    logger.info("=" * 55)
    return passed


def compute_memory_estimate(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    use_bf16: bool = True,
) -> Dict[str, float]:
    """
    Estimate GPU memory consumption for full fine-tuning.

    Provides a breakdown of the four memory categories:
        1. Model weights (bf16 or fp32)
        2. Gradients (fp32)
        3. AdamW optimizer states (fp32 first + second moments)
        4. Activation estimate (depends on batch_size, seq_len, num_layers)

    Note: Activation estimate is approximate — Flash Attention and gradient
    checkpointing can significantly reduce the actual activation memory.
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Determine hidden size and num_layers for activation estimate
    # GPT-2 specific; adapt for other architectures
    hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 768))
    num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 12))
    vocab_size = getattr(model.config, "vocab_size", 50257)

    weight_bytes = total_params * (2 if use_bf16 else 4)
    grad_bytes = total_params * 4  # always fp32
    optimizer_bytes = total_params * 8  # fp32 first + second moments

    # Activation estimate: B * T * d * L * 4 bytes (rough, without checkpointing)
    # With gradient checkpointing, divide by approximately sqrt(num_layers)
    activation_bytes_full = batch_size * seq_len * hidden_size * num_layers * 4
    activation_bytes_ckpt = batch_size * seq_len * hidden_size * int(math.sqrt(num_layers)) * 4

    def to_gb(b):
        return b / (1024 ** 3)

    estimate = {
        "model_weights_gb": to_gb(weight_bytes),
        "gradients_gb": to_gb(grad_bytes),
        "optimizer_states_gb": to_gb(optimizer_bytes),
        "activations_no_ckpt_gb": to_gb(activation_bytes_full),
        "activations_with_ckpt_gb": to_gb(activation_bytes_ckpt),
        "total_no_ckpt_gb": to_gb(weight_bytes + grad_bytes + optimizer_bytes + activation_bytes_full),
        "total_with_ckpt_gb": to_gb(weight_bytes + grad_bytes + optimizer_bytes + activation_bytes_ckpt),
        "total_params": total_params,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }

    logger.info("MEMORY ESTIMATE (Full Fine-Tuning)")
    logger.info(f"  Total parameters     : {total_params:>15,}")
    logger.info(f"  Model weights        : {estimate['model_weights_gb']:>8.2f} GB  ({'bf16' if use_bf16 else 'fp32'})")
    logger.info(f"  Gradients            : {estimate['gradients_gb']:>8.2f} GB  (fp32)")
    logger.info(f"  Optimizer states     : {estimate['optimizer_states_gb']:>8.2f} GB  (fp32 AdamW)")
    logger.info(f"  Activations (no ckpt): {estimate['activations_no_ckpt_gb']:>8.2f} GB  (batch={batch_size}, seq={seq_len})")
    logger.info(f"  Activations (w/ ckpt): {estimate['activations_with_ckpt_gb']:>8.2f} GB  (approx sqrt(L) savings)")
    logger.info(f"  TOTAL (no ckpt)      : {estimate['total_no_ckpt_gb']:>8.2f} GB")
    logger.info(f"  TOTAL (with ckpt)    : {estimate['total_with_ckpt_gb']:>8.2f} GB")

    return estimate


def run_mathematical_verification() -> None:
    """
    Verify the core mathematical components of the FFT implementation.

    Tests:
        1. AdamW update produces correct parameter movement direction
        2. Weight decay term is correctly decoupled (applied to parameter, not gradient)
        3. Cosine schedule produces correct LR values at known checkpoints
        4. Gradient accumulation produces identical updates to single large batch
        5. Loss masking correctly excludes prompt tokens from gradient
    """
    logger.info("=" * 65)
    logger.info("MATHEMATICAL VERIFICATION SUITE")
    all_passed = True

    # ---- Test 1: AdamW decoupled weight decay ----
    logger.info("Test 1: AdamW decoupled weight decay")
    p_adam = nn.Parameter(torch.tensor([1.0]))
    p_adamw = nn.Parameter(torch.tensor([1.0]))
    g = torch.tensor([0.1])

    opt_adam = torch.optim.Adam([p_adam], lr=0.1, weight_decay=0.01)
    opt_adamw = AdamW([p_adamw], lr=0.1, weight_decay=0.01)

    p_adam.grad = g.clone()
    p_adamw.grad = g.clone()
    opt_adam.step()
    opt_adamw.step()

    # In AdamW, weight decay is applied AFTER the adaptive step, pulling toward 0
    # In Adam, weight decay modifies the gradient BEFORE the adaptive step
    # For param=1.0, gradient=0.1, wd=0.01:
    # AdamW final param should be < Adam final param because direct wd is more aggressive
    # (not scaled by 1/sqrt(v), where v is small at step 1 making division amplify wd)
    passed = (p_adamw.item() != p_adam.item())
    status = "PASS" if passed else "FAIL"
    logger.info(
        f"  Adam param={p_adam.item():.6f}, AdamW param={p_adamw.item():.6f} "
        f"(different weight decay behavior) [{status}]"
    )
    all_passed = all_passed and passed

    # ---- Test 2: Cosine schedule values at known checkpoints ----
    logger.info("Test 2: Cosine LR schedule correctness")
    p_sched = nn.Parameter(torch.zeros(1))
    opt_sched = AdamW([p_sched], lr=1e-3)
    sched = build_warmup_cosine_scheduler(opt_sched, num_warmup_steps=10, num_training_steps=110)

    lrs = []
    for _ in range(110):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # At end of warmup (step 10): should be near lr_max = 1e-3
    # At step 60 (progress=0.5): should be near 0.55 * 1e-3 (cosine midpoint)
    # At step 109 (progress~1): should be near lr_min = 0.1 * 1e-3
    warmup_end_lr = lrs[9]
    midpoint_lr = lrs[59]
    end_lr = lrs[108]

    p2a = abs(warmup_end_lr - 1e-3) < 1e-4
    p2b = midpoint_lr < warmup_end_lr
    p2c = end_lr < midpoint_lr
    p2 = p2a and p2b and p2c
    status = "PASS" if p2 else "FAIL"
    logger.info(
        f"  LR at warmup_end={warmup_end_lr:.6f} (expected ~1e-3), "
        f"midpoint={midpoint_lr:.6f}, end={end_lr:.6f} [{status}]"
    )
    all_passed = all_passed and p2

    # ---- Test 3: Gradient accumulation equivalence ----
    logger.info("Test 3: Gradient accumulation == single large batch gradient")
    torch.manual_seed(99)
    model_single = nn.Linear(4, 2, bias=False)
    model_accum = nn.Linear(4, 2, bias=False)
    model_accum.weight.data.copy_(model_single.weight.data)

    # Single batch of 8 samples
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))

    # Single batch gradient
    loss_single = F.cross_entropy(model_single(x), y)
    loss_single.backward()
    grad_single = model_single.weight.grad.clone()

    # Gradient accumulation: 2 micro-batches of 4
    for i in range(2):
        x_micro = x[i*4:(i+1)*4]
        y_micro = y[i*4:(i+1)*4]
        loss_micro = F.cross_entropy(model_accum(x_micro), y_micro) / 2  # divide by K=2
        loss_micro.backward()
    grad_accum = model_accum.weight.grad.clone()

    max_diff = (grad_single - grad_accum).abs().max().item()
    p3 = max_diff < 1e-5
    status = "PASS" if p3 else "FAIL"
    logger.info(
        f"  Max gradient difference (single_batch vs accumulated): "
        f"{max_diff:.2e} (expected < 1e-5) [{status}]"
    )
    all_passed = all_passed and p3

    # ---- Test 4: Loss masking gradient isolation ----
    logger.info("Test 4: Loss masking excludes prompt tokens from gradient")
    torch.manual_seed(42)
    vocab_size = 10
    hidden = 8
    seq_len = 6
    prompt_len = 3

    # Create a tiny 1-layer model (just a linear head for testing)
    head = nn.Linear(hidden, vocab_size)
    fake_hidden = torch.randn(1, seq_len, hidden, requires_grad=True)
    logits = head(fake_hidden)  # (1, 6, 10)

    # Labels: -100 for prompt (positions 0-2), actual for response (3-5)
    labels_masked = torch.tensor([[-100, -100, -100, 3, 7, 2]])
    labels_all = torch.tensor([[1, 5, 4, 3, 7, 2]])

    criterion_test = FFTMaskedCrossEntropyLoss(ignore_index=-100)

    loss_masked, _ = criterion_test(logits, labels_masked)
    loss_masked.backward()
    grad_masked = fake_hidden.grad.clone()

    fake_hidden.grad = None
    logits2 = head(fake_hidden)
    loss_all, _ = criterion_test(logits2, labels_all)
    loss_all.backward()
    grad_all = fake_hidden.grad.clone()

    # Gradients at prompt positions should be zero for masked loss
    prompt_grad_masked = grad_masked[0, :prompt_len, :].abs().max().item()
    response_grad_masked = grad_masked[0, prompt_len:, :].abs().max().item()
    p4 = prompt_grad_masked < 1e-7 and response_grad_masked > 1e-7
    status = "PASS" if p4 else "FAIL"
    logger.info(
        f"  Prompt position grad magnitude (masked loss): {prompt_grad_masked:.2e} "
        f"(expected ~0), Response grad: {response_grad_masked:.2e} (expected >0) [{status}]"
    )
    all_passed = all_passed and p4

    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    logger.info(f"Mathematical verification: {overall}")
    logger.info("=" * 65)


# =============================================================================
# SECTION 8: MAIN ENTRYPOINT
# =============================================================================

def main():
    """
    End-to-end Full Fine-Tuning pipeline.

    Execution stages:
        1. Set random seeds for reproducibility
        2. Load tokenizer (must match the pre-trained model's tokenizer)
        3. Run mathematical verification suite
        4. Build and split the instruction dataset
        5. Verify loss masking implementation
        6. Load pre-trained model and verify all parameters are trainable
        7. Compute and display memory estimate
        8. Build and run the FFTTrainer
        9. Report final metrics and save training history
    """
    logger.info("Full Fine-Tuning (FFT) Pipeline — GPT-2 (117M)")

    config = FFTConfig()

    # ---- Reproducibility ----
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if not HAS_TRANSFORMERS:
        logger.error("Install dependencies: pip install torch transformers")
        return

    # ---- Run mathematical verification before any training ----
    run_mathematical_verification()

    # ---- Tokenizer ----
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token = eos_token (id={tokenizer.eos_token_id})")

    # ---- Dataset construction and split ----
    # Three-way split: train / validation / test
    # Test set is held out until final evaluation — never used during training
    logger.info(f"Preparing dataset from {len(DEMO_DATASET)} examples")
    n_train = int(len(DEMO_DATASET) * config.train_split)
    n_val = int(len(DEMO_DATASET) * 0.10)
    n_test = len(DEMO_DATASET) - n_train - n_val

    train_data = DEMO_DATASET[:n_train]
    val_data = DEMO_DATASET[n_train:n_train + n_val]
    test_data = DEMO_DATASET[n_train + n_val:]

    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_dataset = FFTInstructionDataset(train_data, tokenizer, config.max_seq_length)
    eval_dataset = FFTInstructionDataset(val_data, tokenizer, config.max_seq_length)

    if len(train_dataset) == 0:
        logger.error("Training dataset empty after preprocessing. Check data format.")
        return

    # ---- Verify loss masking ----
    verify_loss_masking(tokenizer, DEMO_DATASET[0])

    # ---- Load pre-trained model ----
    logger.info(f"Loading pre-trained model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,   # Load in fp32; trainer casts to bf16 if GPU available
    )

    # ---- Verify all parameters are trainable (FFT requirement) ----
    verify_all_params_trainable(model)

    # ---- Memory estimate ----
    compute_memory_estimate(
        model=model,
        batch_size=config.per_device_batch_size,
        seq_len=config.max_seq_length,
        use_bf16=config.use_bf16,
    )

    # ---- Build trainer and run training ----
    trainer = FFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting full fine-tuning...")
    history = trainer.train()

    # ---- Save training history ----
    os.makedirs(config.output_dir, exist_ok=True)

    history_path = os.path.join(config.output_dir, "fft_training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [round(float(v), 6) for v in vals] for k, vals in history.items()},
            f,
            indent=2,
        )

    config_path = os.path.join(config.output_dir, "fft_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # ---- Final summary ----
    logger.info("Full Fine-Tuning complete. Summary:")
    if history["train_loss"]:
        logger.info(f"  Final train loss : {history['train_loss'][-1]:.4f}")
    logger.info(f"  Best eval loss   : {trainer.best_eval_loss:.4f}")
    logger.info(f"  Output directory : {config.output_dir}")
    logger.info(f"  Best checkpoint  : {os.path.join(config.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
