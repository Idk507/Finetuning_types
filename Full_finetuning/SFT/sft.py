"""
sft_implementation.py
=====================
Production-Grade Supervised Fine-Tuning (SFT) Implementation
Using GPT-2 (Small Language Model) as the demonstration model.

This file demonstrates the complete SFT pipeline end-to-end:
  1. Dataset construction and preprocessing with loss masking
  2. Model loading and LoRA adapter injection
  3. Full fine-tuning and LoRA fine-tuning training loops
  4. Correct masked cross-entropy loss computation
  5. AdamW optimizer with cosine LR schedule and warmup
  6. Gradient clipping and mixed precision (bfloat16)
  7. Checkpointing, evaluation, and sample generation
  8. Mathematical verification of loss and gradient computations

Architecture: Decoder-only Transformer (GPT-2, 117M parameters)
Framework:    PyTorch >= 2.0, Transformers >= 4.35, PEFT >= 0.7
Hardware:     Works on CPU (GPT-2 small), GPU recommended

Mathematical background:
  - Cross-entropy loss computed ONLY on completion tokens (loss masking)
  - AdamW update: theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * theta
  - LoRA: W_adapted = W_frozen + (alpha/r) * B @ A
  - Gradient clipping: grad = grad * max_norm / ||grad|| if ||grad|| > max_norm
"""

import os
import math
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Suppress tokenizer warnings for cleaner output
warnings.filterwarnings("ignore", message=".*tokenizer.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Configure structured logging for production observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sft")


# =============================================================================
# SECTION 1: CONFIGURATION
# All training hyperparameters are centralized in a dataclass.
# =============================================================================

@dataclass
class SFTConfig:
    """
    Central configuration for the SFT training run.

    All hyperparameters are documented with their mathematical meaning
    and recommended search ranges for production tuning.

    Mathematical context:
        - max_seq_length: determines O(L^2) memory usage of attention
        - learning_rate: controls step size in parameter space; too large
          destroys pre-trained weights, too small yields no adaptation
        - warmup_ratio: fraction of total steps used for linear LR warmup
        - weight_decay: lambda in the AdamW L2 regularization term
        - lora_r: rank r of the LoRA decomposition Delta_W = B @ A
        - lora_alpha: scaling factor; effective lr scales as alpha/r
    """
    # --- Model ---
    model_name: str = "gpt2"              # HuggingFace model ID or local path
    use_lora: bool = True                  # Whether to use LoRA (PEFT) or full FT

    # --- Data ---
    max_seq_length: int = 512             # Maximum token length (prompt + completion)
    train_split: float = 0.9             # Fraction of data used for training

    # --- Training ---
    num_epochs: int = 3                   # Number of passes over the training data
    per_device_batch_size: int = 4        # Samples per GPU per gradient step
    gradient_accumulation_steps: int = 4  # Simulates effective_batch = batch * accum
    learning_rate: float = 2e-4          # Peak learning rate after warmup
    weight_decay: float = 0.01           # AdamW L2 regularization coefficient
    warmup_ratio: float = 0.05           # Fraction of steps used for linear warmup
    max_grad_norm: float = 1.0           # L2 gradient clipping threshold
    adam_beta1: float = 0.9              # AdamW first moment decay
    adam_beta2: float = 0.999            # AdamW second moment decay
    adam_epsilon: float = 1e-8           # AdamW numerical stability constant

    # --- LoRA Hyperparameters ---
    lora_r: int = 8                      # Rank of LoRA decomposition
    lora_alpha: float = 16.0             # LoRA scaling (effective_alpha = alpha/r)
    lora_dropout: float = 0.05          # Dropout applied to LoRA adapter
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj"]  # GPT-2 attention projections
    )

    # --- Evaluation and Checkpointing ---
    eval_steps: int = 50                 # Evaluate on validation set every N steps
    save_steps: int = 100               # Save checkpoint every N steps
    output_dir: str = "./sft_output"    # Directory for checkpoints and final model
    logging_steps: int = 10             # Log metrics every N steps
    num_sample_generations: int = 3     # Number of sample outputs to log at eval

    # --- Precision ---
    use_bf16: bool = True               # Use bfloat16 mixed precision (requires GPU)
    seed: int = 42                      # Random seed for reproducibility

    @property
    def effective_batch_size(self) -> int:
        """
        The effective batch size seen by the optimizer equals the product
        of per_device_batch_size and gradient_accumulation_steps.
        Larger effective batch sizes produce more stable gradient estimates
        at the cost of longer wall-clock time between optimizer updates.
        """
        return self.per_device_batch_size * self.gradient_accumulation_steps


# =============================================================================
# SECTION 2: DATASET — CONSTRUCTION AND TOKENIZATION
# =============================================================================

# A minimal high-quality instruction-following dataset for demonstration.
# In production, replace this with your curated dataset of (instruction, response) pairs.
DEMO_DATASET = [
    {
        "instruction": "What is supervised fine-tuning in the context of language models?",
        "response": (
            "Supervised fine-tuning (SFT) is the process of continuing the training "
            "of a pre-trained language model on a curated dataset of (input, output) "
            "pairs using the standard cross-entropy objective. The goal is to align "
            "the model's behavior with human preferences for instruction following, "
            "question answering, or other specific tasks. Unlike pre-training, which "
            "operates on unlabeled text, SFT uses labeled demonstrations and computes "
            "the loss only on the desired response tokens, not on the prompt."
        ),
    },
    {
        "instruction": "Explain the difference between pre-training and fine-tuning a language model.",
        "response": (
            "Pre-training involves training a model from randomly initialized parameters "
            "on vast amounts of unlabeled text using a self-supervised objective such as "
            "next-token prediction. The model learns general-purpose language representations "
            "from trillions of tokens at enormous computational cost. Fine-tuning, by "
            "contrast, starts from the pre-trained checkpoint and continues training on a "
            "much smaller, task-specific labeled dataset. Fine-tuning is computationally "
            "cheap relative to pre-training and adapts the model's behavior to a specific "
            "distribution, such as instruction following or medical question answering."
        ),
    },
    {
        "instruction": "What is the cross-entropy loss and why is it used for language model training?",
        "response": (
            "Cross-entropy loss measures how well a probability distribution predicted "
            "by the model matches the true distribution. For a ground truth token v* at "
            "position t, the loss is -log P(v* | context; theta), where P is the softmax "
            "of the model's logits. It is the standard loss for language modeling because "
            "it is equivalent to maximum likelihood estimation — minimizing cross-entropy "
            "is identical to maximizing the likelihood of the observed tokens under the model. "
            "The gradient of cross-entropy is (softmax(logits) - one_hot(v*)), which has the "
            "desirable property of being large when the model is wrong and small when correct."
        ),
    },
    {
        "instruction": "What is LoRA and how does it reduce the computational cost of fine-tuning?",
        "response": (
            "LoRA (Low-Rank Adaptation) reduces fine-tuning cost by hypothesizing that the "
            "weight update matrix Delta_W has low intrinsic rank and can be decomposed as "
            "Delta_W = B @ A, where B is (d_out x r) and A is (r x d_in) with r much smaller "
            "than either dimension. Instead of updating all d_in * d_out parameters, LoRA "
            "updates only r * (d_in + d_out) parameters. For a 4096-dimensional weight matrix "
            "with rank 8, this reduces parameters from 16.7 million to 65 thousand — a 256x "
            "reduction. The frozen base weights are unchanged, preventing catastrophic forgetting, "
            "and only the small adapter matrices B and A are trained."
        ),
    },
    {
        "instruction": "Why is loss masking important in supervised fine-tuning?",
        "response": (
            "Loss masking ensures that the training objective teaches the model to generate "
            "good completions rather than to predict the input prompt. In SFT, each training "
            "example is a concatenated (prompt, response) sequence. Without masking, the "
            "cross-entropy loss would be computed over all tokens, penalizing the model for "
            "not predicting the prompt tokens. This is wasteful and introduces noise because "
            "the prompt tokens have many plausible alternatives. By masking the loss to "
            "completion tokens only (setting prompt token labels to -100 in PyTorch), we "
            "focus the gradient signal entirely on learning to produce the desired response "
            "given the prompt context."
        ),
    },
    {
        "instruction": "Describe the AdamW optimizer and how it differs from Adam.",
        "response": (
            "AdamW decouples weight decay from the adaptive gradient update, which is the "
            "key difference from the original Adam optimizer. In Adam, weight decay was "
            "implemented by adding the regularization term to the gradient before computing "
            "the adaptive scaling, which changed the effective regularization strength. In "
            "AdamW, weight decay is applied directly to the parameters after the adaptive "
            "gradient step: theta = theta - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * theta. "
            "This separation ensures that weight decay consistently provides L2 regularization "
            "regardless of the gradient magnitude, which is particularly important for "
            "preventing overfitting during fine-tuning on small datasets."
        ),
    },
    {
        "instruction": "What is gradient clipping and when is it necessary?",
        "response": (
            "Gradient clipping limits the L2 norm of the full parameter gradient vector "
            "before the optimizer step. If the gradient norm exceeds a threshold max_norm, "
            "the gradient is rescaled: grad = grad * max_norm / ||grad||. This prevents "
            "exploding gradients, which can occur when the loss landscape has very steep "
            "cliffs or when a batch of training examples produces unusually large gradients. "
            "In fine-tuning, gradient clipping is particularly important in the early steps "
            "before warmup stabilizes training. The typical threshold is 1.0. Monitoring the "
            "gradient norm during training reveals instability — if clipping triggers on "
            "every step, the learning rate is likely too high."
        ),
    },
    {
        "instruction": "What is catastrophic forgetting in the context of fine-tuning?",
        "response": (
            "Catastrophic forgetting occurs when fine-tuning on a narrow task distribution "
            "causes the model to lose knowledge it acquired during pre-training. Because "
            "fine-tuning updates all parameters toward minimizing the task-specific loss, "
            "parameters that encoded general-purpose knowledge may be overwritten if the "
            "fine-tuning gradient points in a conflicting direction. The result is a model "
            "that performs well on the fine-tuning task but fails on tasks it previously "
            "handled correctly. Mitigation strategies include low learning rates, LoRA "
            "(which freezes base weights), mixing pre-training data into the fine-tuning "
            "dataset (replay), and early stopping based on validation performance on "
            "held-out general-purpose benchmarks."
        ),
    },
]


class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction-following SFT training.

    Each example is formatted as:
        [BOS] ### Instruction:\n{instruction}\n\n### Response:\n{response} [EOS]

    The loss mask is set to 1 only for response tokens, 0 for instruction/prompt tokens.
    This implements the core SFT masking principle: gradients flow only through
    the completion, not through the prompt.

    Mathematical implication:
        L(theta) = -(1/|response|) * sum_{t in response} log P(x_t | x_{<t}; theta)

    Attributes:
        data: List of raw (instruction, response) dictionaries.
        tokenizer: Pre-trained tokenizer matching the model's vocabulary.
        max_seq_length: Maximum allowed token sequence length.
        stats: Dictionary of dataset statistics computed after tokenization.
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
        self.processed = []
        self.stats = {}
        self._process_all()

    def _process_all(self) -> None:
        """
        Tokenize all examples and compute the loss mask.

        The loss mask implementation:
            1. Tokenize the full prompt (instruction section only).
            2. Tokenize the full sequence (prompt + response).
            3. Set labels[0 : prompt_length] = -100 (ignored by CrossEntropyLoss).
            4. Set labels[prompt_length :] = input_ids[prompt_length :] (response tokens).

        PyTorch's nn.CrossEntropyLoss with ignore_index=-100 automatically
        skips positions where the label is -100, implementing the loss mask.
        """
        lengths = []
        response_token_counts = []
        skipped = 0

        for idx, example in enumerate(self.data):
            instruction = example.get("instruction", "").strip()
            response = example.get("response", "").strip()

            if not instruction or not response:
                logger.warning(f"Skipping example {idx}: empty instruction or response.")
                skipped += 1
                continue

            # Build the prompt prefix (everything before the response)
            prompt_text = self.PROMPT_TEMPLATE.format(instruction=instruction)

            # Tokenize the prompt-only portion to determine where response begins
            # We do not add special tokens here because we handle them in the full sequence
            prompt_ids = self.tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )

            # Build the full sequence: prompt + response + EOS token
            full_text = prompt_text + response
            full_encoding = self.tokenizer(
                full_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,  # Adds BOS at the beginning
                return_tensors=None,
            )

            input_ids = full_encoding["input_ids"]
            attention_mask = full_encoding["attention_mask"]

            # Compute the prompt length after BOS token is prepended by the tokenizer.
            # +1 accounts for the BOS token at position 0.
            # This is the key implementation detail: we identify the boundary between
            # prompt tokens (mask=-100) and response tokens (mask=actual_id).
            prompt_length = len(prompt_ids) + 1  # +1 for BOS

            # Safety check: if truncation cut into the response, skip this example.
            # We need at least 1 response token to compute a meaningful loss.
            if prompt_length >= len(input_ids):
                logger.debug(
                    f"Example {idx} skipped: prompt_length ({prompt_length}) >= "
                    f"total_length ({len(input_ids)}). Increase max_seq_length."
                )
                skipped += 1
                continue

            # Construct the labels tensor with the loss mask applied
            # Position 0 to prompt_length-1: label = -100 (do not compute loss here)
            # Position prompt_length to end: label = input_id (compute loss here)
            labels = [-100] * len(input_ids)
            labels[prompt_length:] = input_ids[prompt_length:]

            num_response_tokens = len(input_ids) - prompt_length
            response_token_counts.append(num_response_tokens)
            lengths.append(len(input_ids))

            self.processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        # Compute and store dataset statistics for observability
        if lengths:
            self.stats = {
                "total_examples": len(self.processed),
                "skipped_examples": skipped,
                "mean_seq_length": sum(lengths) / len(lengths),
                "max_seq_length_found": max(lengths),
                "min_seq_length_found": min(lengths),
                "mean_response_tokens": (
                    sum(response_token_counts) / len(response_token_counts)
                ),
                "total_response_tokens": sum(response_token_counts),
            }
            logger.info(
                f"Dataset processed: {self.stats['total_examples']} examples, "
                f"mean_length={self.stats['mean_seq_length']:.1f} tokens, "
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


def sft_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Custom collation function that pads sequences within a batch to the same length.

    Padding strategy:
        - input_ids: padded with pad_token_id on the right
        - attention_mask: padded with 0 on the right (masked out in attention)
        - labels: padded with -100 on the right (ignored by loss function)

    Using right padding (rather than left padding) is standard for causal LM training.
    For autoregressive generation, left padding is preferred but this is training,
    so right padding is appropriate and more memory-efficient.

    Args:
        batch: List of dictionaries from InstructionDataset.__getitem__.
        pad_token_id: Token ID to use for padding input_ids.

    Returns:
        Dictionary with padded, batched tensors ready for model forward pass.
    """
    max_length = max(item["input_ids"].size(0) for item in batch)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_size = max_length - seq_len

        # Right-pad input_ids with pad_token_id
        padded_input_ids.append(
            F.pad(item["input_ids"], (0, pad_size), value=pad_token_id)
        )
        # Right-pad attention_mask with 0 (do not attend to padding positions)
        padded_attention_masks.append(
            F.pad(item["attention_mask"], (0, pad_size), value=0)
        )
        # Right-pad labels with -100 (ignore padding in loss computation)
        padded_labels.append(
            F.pad(item["labels"], (0, pad_size), value=-100)
        )

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.stack(padded_labels),
    }


# =============================================================================
# SECTION 3: LoRA IMPLEMENTATION FROM SCRATCH
# Manual LoRA for educational transparency (no PEFT library dependency)
# =============================================================================

class LoRALinear(nn.Module):
    """
    A linear layer augmented with a Low-Rank Adaptation (LoRA) module.

    Mathematical formulation:
        Forward pass: y = x @ (W + scale * B @ A) + bias
        where:
            W in R^{d_in x d_out} — frozen pre-trained weight (transposed convention)
            A in R^{r x d_in}    — LoRA down-projection, initialized ~ N(0, sigma^2)
            B in R^{d_out x r}   — LoRA up-projection, initialized to 0
            scale = alpha / r    — controls the effective magnitude of the adaptation

    Initialization:
        A ~ N(0, 1/r): Kaiming-inspired scaling ensures that at initialization,
            the product B @ A = 0 (B is zero), so the model starts from the
            exact pre-trained checkpoint. No warm-up from a random state.
        B = 0: Guarantees Delta_W = 0 at the start of training.

    Trainable parameters:
        Only A and B are in requires_grad=True; W.weight and W.bias are frozen.
        This reduces the number of updated parameters from d_in * d_out
        to r * (d_in + d_out).

    Attributes:
        base_layer: The original frozen nn.Linear layer.
        lora_A: Down-projection matrix A, shape (r, d_in).
        lora_B: Up-projection matrix B, shape (d_out, r).
        scale: The alpha/r scaling factor applied to B @ A.
        dropout: Optional dropout applied to the input before LoRA path.
        r: The rank of the LoRA decomposition.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.scale = alpha / r
        self.base_layer = base_layer

        # Freeze the original weight and bias — these must never receive gradients
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        d_out, d_in = base_layer.weight.shape  # PyTorch convention: (out, in)

        # Initialize A from a Gaussian; B initialized to zero so Delta_W = 0 at start
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        # Kaiming-inspired initialization for A: scale by 1/sqrt(r)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Optional dropout on the activation before the LoRA path
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining the frozen base layer and the LoRA adaptation.

        Computation:
            base_out = x @ W.T + bias   (frozen; no gradient w.r.t. W)
            lora_out = dropout(x) @ A.T @ B.T * scale   (trainable)
            output   = base_out + lora_out

        The combined operation is equivalent to a linear layer with weight
            W_adapted = W + scale * B @ A
        but avoids materializing the full adapted weight matrix in memory.
        This is particularly important for large weight matrices where storing
        Delta_W would be as expensive as storing W itself.

        Args:
            x: Input tensor of shape (..., d_in).

        Returns:
            Output tensor of shape (..., d_out).
        """
        # Frozen path: compute x @ W.T + bias using original weights
        base_out = self.base_layer(x)

        # LoRA path: compute dropout(x) @ A.T then @ B.T, then scale
        # x has shape (..., d_in)
        # A has shape (r, d_in) -> A.T has shape (d_in, r)
        # B has shape (d_out, r) -> B.T has shape (r, d_out)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge the LoRA adaptation into the base weight for efficient inference.

        After merging, the adapted module is a standard nn.Linear with weight:
            W_merged = W + scale * B @ A

        This eliminates the two extra matrix multiplications in the LoRA forward
        pass, reducing inference latency to that of a regular linear layer.
        Merging should be done after training is complete and before deployment.

        Returns:
            A new nn.Linear layer with merged weights, ready for inference.
        """
        d_out, d_in = self.base_layer.weight.shape
        merged = nn.Linear(d_in, d_out, bias=self.base_layer.bias is not None)

        # W_merged = W_frozen + (alpha/r) * B @ A
        delta_w = self.scale * (self.lora_B @ self.lora_A)
        merged.weight = nn.Parameter(
            self.base_layer.weight.data + delta_w.detach()
        )
        if self.base_layer.bias is not None:
            merged.bias = nn.Parameter(self.base_layer.bias.data.clone())

        merged.weight.requires_grad_(False)
        if merged.bias is not None:
            merged.bias.requires_grad_(False)

        return merged

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns trainable vs frozen parameter counts for this layer."""
        trainable = self.lora_A.numel() + self.lora_B.numel()
        frozen = self.base_layer.weight.numel()
        if self.base_layer.bias is not None:
            frozen += self.base_layer.bias.numel()
        return {"trainable": trainable, "frozen": frozen}


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Inject LoRA adapters into target linear layers of a model in-place.

    This function walks the module tree and replaces every nn.Linear whose
    name ends with any string in target_modules with a LoRALinear wrapper.
    All other parameters remain unchanged with requires_grad=False.

    Implementation detail: We use a two-pass approach — collect targets first,
    then replace — to avoid modifying the module tree during iteration.

    Args:
        model: The pre-trained model to modify.
        target_modules: List of module name suffixes to target (e.g., ["c_attn"]).
        r: LoRA rank.
        alpha: LoRA scaling coefficient.
        dropout: Dropout rate in LoRA adapter.

    Returns:
        The model with LoRA adapters injected (modified in place, also returned).
    """
    # First, freeze ALL parameters in the model
    for param in model.parameters():
        param.requires_grad_(False)

    # Collect (parent_module, attribute_name, full_path) for all target layers
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if the last component of the name matches any target
            short_name = name.split(".")[-1]
            if short_name in target_modules:
                # Navigate to the parent module using the full path
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                targets.append((parent, parts[-1], name, module))

    if not targets:
        logger.warning(
            f"No modules matching target_modules={target_modules} found in the model. "
            f"No LoRA adapters were injected. Check target_modules configuration."
        )
        return model

    # Replace each target with a LoRALinear wrapper
    for parent, attr_name, full_path, base_layer in targets:
        lora_layer = LoRALinear(
            base_layer=base_layer,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, attr_name, lora_layer)
        logger.debug(f"Injected LoRA into: {full_path}")

    # Count and log parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(
        f"LoRA injection complete: "
        f"trainable={trainable_params:,} ({100*trainable_params/total_params:.2f}%), "
        f"frozen={frozen_params:,}, "
        f"total={total_params:,}, "
        f"adapters_injected={len(targets)}"
    )

    return model


# =============================================================================
# SECTION 4: LOSS FUNCTION
# =============================================================================

class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for language model training with loss masking.

    This is the core SFT objective. The loss is computed only over positions
    where labels != ignore_index (-100), which corresponds to response tokens.

    Mathematical formulation:
        For a sequence of length T with response tokens at positions S ⊆ {0,...,T-1}:

        L = -(1/|S|) * sum_{t in S} log P(x_t | x_{<t}; theta)
          = -(1/|S|) * sum_{t in S} [logits_t[y_t] - LogSumExp(logits_t)]

        where:
            logits_t in R^{|vocab|} — model output at position t
            y_t — ground truth token at position t
            |S| — number of response tokens (not total sequence length)

    The shifting convention:
        In autoregressive language modeling, the model predicts x_{t+1} given
        x_{1:t}. Therefore, logits at position t (computed from tokens 0..t)
        are compared against labels at position t+1. This requires shifting:
            input_logits:  logits[..., :-1, :]  (all positions except last)
            target_labels: labels[..., 1:]      (all positions except first)

    Attributes:
        ignore_index: Label value to ignore in loss computation (default: -100).
        reduction: How to reduce the loss ('mean', 'sum', or 'none').
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the masked cross-entropy loss.

        Args:
            logits: Model output logits, shape (batch, seq_len, vocab_size).
            labels: Ground truth token IDs, shape (batch, seq_len).
                    Positions with ignore_index are excluded from loss computation.

        Returns:
            Tuple of:
                loss: Scalar loss tensor (with gradient tape attached).
                metrics: Dictionary containing diagnostic information:
                    - num_active_tokens: Number of response tokens in this batch.
                    - perplexity: exp(loss), the standard NLP evaluation metric.
                    - loss_value: The scalar loss as a Python float.

        Implementation note:
            We use PyTorch's built-in F.cross_entropy with ignore_index, which
            internally handles the log-softmax and NLL computation in a numerically
            stable way (log-sum-exp trick to prevent underflow/overflow).
        """
        # Shift: logits at position t predict the token at position t+1
        # logits[..., :-1, :] has shape (batch, seq_len-1, vocab_size)
        # labels[..., 1:] has shape (batch, seq_len-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Count the number of active (non-masked) tokens for diagnostics
        num_active_tokens = (shift_labels != self.ignore_index).sum().item()

        # Flatten for cross_entropy: (batch * (seq_len-1), vocab_size) vs (batch * (seq_len-1),)
        # F.cross_entropy handles ignore_index internally and computes mean over
        # only the non-ignored positions
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

        # Compute perplexity for logging (detach to avoid keeping graph in memory)
        with torch.no_grad():
            perplexity = torch.exp(loss.detach()).item() if num_active_tokens > 0 else float("inf")

        metrics = {
            "num_active_tokens": num_active_tokens,
            "perplexity": perplexity,
            "loss_value": loss.item(),
        }

        return loss, metrics


# =============================================================================
# SECTION 5: LEARNING RATE SCHEDULER
# =============================================================================

def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup followed by cosine decay.

    Mathematical formulation:
        Warmup phase (0 <= t < num_warmup_steps):
            lr(t) = lr_max * t / num_warmup_steps

        Cosine decay phase (num_warmup_steps <= t <= num_training_steps):
            progress = (t - W) / (T - W)   where W=warmup_steps, T=total_steps
            lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
                  = lr_max * [min_ratio + 0.5 * (1 - min_ratio) * (1 + cos(pi * progress))]

        The lambda function returned computes the multiplier relative to lr_max.

    Rationale for warmup:
        At the start of fine-tuning, gradient estimates are noisy (single batch,
        potentially out-of-distribution from the pre-training regime). Large
        initial learning rates can cause destructive parameter updates that
        move the model far from the pre-trained checkpoint before useful gradients
        have been established. Linear warmup gently increases the step size,
        allowing the optimizer momentum to build up before full-scale updates.

    Rationale for cosine decay:
        Cosine annealing smoothly reduces the learning rate, allowing the
        optimizer to make larger exploratory steps early in training and
        progressively smaller, more refined steps near convergence. Compared
        to linear decay, cosine provides a longer period of moderate learning
        rates before the final rapid reduction.

    Args:
        optimizer: The AdamW optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup phase.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Fraction of peak LR at the end of training (default: 0.1).

    Returns:
        A LambdaLR scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Clamp progress to [0, 1] to handle steps beyond num_training_steps
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# SECTION 6: TRAINER
# =============================================================================

class SFTTrainer:
    """
    Production-grade Supervised Fine-Tuning trainer.

    This class orchestrates the complete SFT training loop including:
        - Mixed precision training (bfloat16) with fp32 gradient accumulation
        - Gradient clipping for training stability
        - Masked cross-entropy loss computed only on response tokens
        - Cosine learning rate schedule with linear warmup
        - Periodic evaluation with validation loss and sample generation
        - Checkpoint saving with full training state
        - Comprehensive metric logging

    Training loop pseudocode:
        for epoch in range(num_epochs):
            for step, batch in enumerate(train_dataloader):
                logits = model(input_ids, attention_mask)
                loss = masked_cross_entropy(logits, labels) / grad_accum_steps
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    clip_grad_norm(model.parameters(), max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

    Attributes:
        config: SFTConfig with all hyperparameters.
        model: The language model being fine-tuned.
        tokenizer: The tokenizer associated with the model.
        train_dataset: Training InstructionDataset.
        eval_dataset: Validation InstructionDataset.
        device: The compute device (cpu or cuda).
        criterion: MaskedCrossEntropyLoss instance.
        optimizer: AdamW optimizer.
        scheduler: Cosine LR scheduler with warmup.
        global_step: Current training step count.
        best_eval_loss: Best validation loss observed, for checkpoint selection.
    """

    def __init__(
        self,
        config: SFTConfig,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        train_dataset: InstructionDataset,
        eval_dataset: InstructionDataset,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # Determine compute device: prefer CUDA for GPU training
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Training device: {self.device}")

        # Enable bfloat16 only if a GPU is available (bfloat16 on CPU is very slow)
        self.use_bf16 = config.use_bf16 and self.device.type == "cuda"
        if config.use_bf16 and not self.use_bf16:
            logger.warning("use_bf16=True but no GPU found; falling back to fp32.")

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = MaskedCrossEntropyLoss(ignore_index=-100)

        # Build optimizer with correct parameter groups
        # The bias and LayerNorm parameters should NOT have weight decay applied
        # because they are not weights in the linear algebra sense and L2 decay
        # on them can impair training
        self.optimizer = self._build_optimizer()

        # Compute total training steps for scheduler
        self.total_steps = self._compute_total_steps()
        self.warmup_steps = max(1, int(self.total_steps * config.warmup_ratio))

        # Build learning rate scheduler
        self.scheduler = build_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
            min_lr_ratio=0.1,
        )

        logger.info(
            f"Trainer initialized: total_steps={self.total_steps}, "
            f"warmup_steps={self.warmup_steps}, "
            f"effective_batch_size={config.effective_batch_size}"
        )

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _build_optimizer(self) -> AdamW:
        """
        Build AdamW optimizer with parameter-group-specific weight decay.

        Weight decay (L2 regularization) should be applied to weight matrices
        but NOT to bias vectors, LayerNorm parameters (gamma and beta), or
        embedding tables. Applying weight decay to these parameters causes
        them to shrink toward zero, degrading model quality.

        We separate parameters into two groups:
            - Group 1 (decay): All weight matrices (2D+ parameters)
            - Group 2 (no_decay): Biases, norms, embeddings (1D parameters)
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters (base weights if using LoRA)
            # Parameters with 'bias', 'norm', or 'ln' in their name get no decay
            if (
                "bias" in name
                or "norm" in name
                or "ln_" in name
                or param.dim() < 2
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        logger.info(
            f"Optimizer: weight_decay applied to {num_decay:,} params, "
            f"no_decay for {num_no_decay:,} params"
        )
        return optimizer

    def _compute_total_steps(self) -> int:
        """
        Compute total optimizer steps across all epochs.

        Total steps = ceil(num_examples / per_device_batch_size)
                    / gradient_accumulation_steps
                    * num_epochs

        This accounts for the fact that the optimizer step occurs only every
        gradient_accumulation_steps batches.
        """
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / self.config.per_device_batch_size
        )
        # Number of optimizer updates per epoch
        updates_per_epoch = math.ceil(
            steps_per_epoch / self.config.gradient_accumulation_steps
        )
        return updates_per_epoch * self.config.num_epochs

    def _build_dataloader(
        self, dataset: InstructionDataset, shuffle: bool = True
    ) -> DataLoader:
        """Build a DataLoader with the custom SFT collation function."""
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: sft_collate_fn(
                batch, pad_token_id=self.tokenizer.pad_token_id
            ),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            num_workers=0,  # Set > 0 for production with large datasets
        )

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single forward pass and compute the masked loss.

        Mixed precision context:
            When use_bf16=True, the forward pass is wrapped in torch.autocast,
            which automatically casts eligible operations to bfloat16, reducing
            memory and increasing throughput. Gradient accumulation and the
            backward pass happen in fp32 to maintain gradient precision.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels tensors.

        Returns:
            Tuple of (loss_tensor, metrics_dict).
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Mixed precision forward pass
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
            enabled=self.use_bf16,
        ):
            # Model forward pass: returns logits of shape (batch, seq_len, vocab_size)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,  # Disable KV cache during training (saves memory)
            )
            logits = outputs.logits

            # Compute masked cross-entropy loss only on response tokens
            loss, metrics = self.criterion(logits, labels)

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation dataset.

        Runs the model in eval mode (disables dropout, uses deterministic behavior)
        and computes the average loss and perplexity over the full validation set.
        Also generates sample outputs from the first few validation prompts to
        enable qualitative monitoring alongside quantitative metrics.

        Returns:
            Dictionary with 'eval_loss', 'eval_perplexity', and optionally
            'sample_generations' if the config requests them.
        """
        self.model.eval()
        eval_loader = self._build_dataloader(self.eval_dataset, shuffle=False)

        total_loss = 0.0
        total_active_tokens = 0
        num_batches = 0

        for batch in eval_loader:
            loss, metrics = self._forward_step(batch)
            # Accumulate weighted sum: weight by number of active tokens so that
            # longer responses don't disproportionately influence the average
            n_tokens = metrics["num_active_tokens"]
            total_loss += loss.item() * n_tokens
            total_active_tokens += n_tokens
            num_batches += 1

        # Average loss over all active (response) tokens in the validation set
        avg_loss = total_loss / max(1, total_active_tokens)
        eval_perplexity = math.exp(min(avg_loss, 20))  # Cap at e^20 to avoid overflow

        eval_metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": eval_perplexity,
            "eval_batches": num_batches,
        }

        # Generate samples for qualitative assessment
        samples = self._generate_samples(n=self.config.num_sample_generations)
        if samples:
            eval_metrics["samples"] = samples

        self.model.train()
        return eval_metrics

    @torch.no_grad()
    def _generate_samples(self, n: int = 3) -> List[str]:
        """
        Generate text from a few prompts to monitor output quality qualitatively.

        Qualitative monitoring is essential because perplexity can remain healthy
        while the model produces degenerate outputs (repetition loops, off-format
        responses, hallucinations). Logging actual generations at every evaluation
        checkpoint provides the earliest signal of these failure modes.

        Args:
            n: Number of sample generations to produce.

        Returns:
            List of formatted strings showing prompt and generated response.
        """
        self.model.eval()
        prompts = [item["instruction"] for item in self.eval_dataset.data[:n]]
        outputs = []

        for prompt in prompts:
            # Format the prompt using the same template as training
            formatted = InstructionDataset.PROMPT_TEMPLATE.format(instruction=prompt)

            tokens = self.tokenizer(
                formatted,
                return_tensors="pt",
                max_length=256,  # Limit prompt length for generation
                truncation=True,
            ).input_ids.to(self.device)

            generated_ids = self.model.generate(
                tokens,
                max_new_tokens=150,
                do_sample=False,       # Greedy decoding for deterministic evaluation
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode only the newly generated tokens (not the prompt)
            new_tokens = generated_ids[0, tokens.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            outputs.append(
                f"PROMPT: {prompt[:100]}...\n"
                f"RESPONSE: {generated_text[:300]}"
            )

        return outputs

    def save_checkpoint(self, step: int, is_best: bool = False) -> None:
        """
        Save model weights and training state to disk.

        Saves:
            - model_state_dict: All model parameters (both frozen and trainable)
            - optimizer_state_dict: AdamW first and second moments
            - scheduler_state_dict: LR schedule state (step counter)
            - global_step, epoch information, config

        For LoRA models, only the LoRA adapter parameters need to be saved for
        inference (the base model can be loaded from HuggingFace), but we save
        the full state dict for training resumption.

        Args:
            step: Current global training step.
            is_best: If True, also saves a copy as 'best_model'.
        """
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "best_eval_loss": self.best_eval_loss,
        }

        # Save regular checkpoint
        ckpt_path = os.path.join(self.config.output_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Save best model separately for easy deployment access
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path} (eval_loss={self.best_eval_loss:.4f})")

    def train(self) -> Dict[str, Any]:
        """
        Execute the full SFT training loop.

        The training loop implements the following algorithm:
            Initialize: load pre-trained model, inject LoRA, set up optimizer

            For each epoch:
                For each batch:
                    1. Forward pass: compute logits
                    2. Compute masked cross-entropy loss
                    3. Scale loss by 1/gradient_accumulation_steps
                    4. Backward pass: accumulate gradients
                    5. If accumulation boundary:
                        a. Clip gradient L2 norm to max_grad_norm
                        b. Optimizer step (AdamW update)
                        c. Scheduler step (cosine LR update)
                        d. Zero gradients
                    6. Log metrics periodically
                    7. Evaluate and checkpoint periodically

        Returns:
            Dictionary of training history metrics for analysis.
        """
        train_loader = self._build_dataloader(self.train_dataset, shuffle=True)

        self.model.train()
        self.optimizer.zero_grad()

        train_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "perplexity": [],
        }

        # Accumulators for loss averaging between logging steps
        accum_loss = 0.0
        accum_tokens = 0
        accum_steps = 0

        logger.info(
            f"Starting SFT training: {self.config.num_epochs} epochs, "
            f"{self.total_steps} total optimizer steps"
        )

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(train_loader):
                # ----------------------------------------------------------------
                # FORWARD PASS AND LOSS COMPUTATION
                # Scale loss by gradient_accumulation_steps so that the gradient
                # is equivalent to computing the mean over the full effective batch.
                # Without this scaling, the gradient magnitude would be
                # gradient_accumulation_steps times too large.
                # ----------------------------------------------------------------
                loss, metrics = self._forward_step(batch)
                scaled_loss = loss / self.config.gradient_accumulation_steps

                # ----------------------------------------------------------------
                # BACKWARD PASS
                # Gradients are accumulated (added) across gradient_accumulation_steps
                # batches before the optimizer update. This simulates a larger
                # effective batch size without requiring more GPU memory.
                # ----------------------------------------------------------------
                scaled_loss.backward()

                # Accumulate metrics for logging
                accum_loss += loss.item() * metrics["num_active_tokens"]
                accum_tokens += metrics["num_active_tokens"]
                accum_steps += 1

                # ----------------------------------------------------------------
                # OPTIMIZER UPDATE (every gradient_accumulation_steps batches)
                # ----------------------------------------------------------------
                is_accumulation_step = (
                    (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(train_loader)  # Last batch in epoch
                )

                if is_accumulation_step:
                    # Compute gradient norm before clipping (for monitoring)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm,
                    ).item()

                    # AdamW parameter update: applies adaptive gradient scaling
                    # and decoupled weight decay to all trainable parameters
                    self.optimizer.step()

                    # Update the learning rate per the cosine schedule
                    self.scheduler.step()

                    # Reset gradients for the next accumulation window
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Get current learning rate from scheduler
                    current_lr = self.scheduler.get_last_lr()[0]

                    # ----------------------------------------------------------------
                    # LOGGING
                    # ----------------------------------------------------------------
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accum_loss / max(1, accum_tokens)
                        avg_ppl = math.exp(min(avg_loss, 20))

                        train_history["train_loss"].append(avg_loss)
                        train_history["learning_rate"].append(current_lr)
                        train_history["grad_norm"].append(grad_norm)
                        train_history["perplexity"].append(avg_ppl)

                        logger.info(
                            f"Step {self.global_step:5d}/{self.total_steps} | "
                            f"Epoch {epoch+1}/{self.config.num_epochs} | "
                            f"loss={avg_loss:.4f} | "
                            f"ppl={avg_ppl:.2f} | "
                            f"lr={current_lr:.2e} | "
                            f"grad_norm={grad_norm:.3f} | "
                            f"tokens={accum_tokens}"
                        )

                        # Reset accumulators
                        accum_loss = 0.0
                        accum_tokens = 0
                        accum_steps = 0

                    # ----------------------------------------------------------------
                    # EVALUATION
                    # ----------------------------------------------------------------
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        eval_loss = eval_metrics["eval_loss"]
                        eval_ppl = eval_metrics["eval_perplexity"]

                        train_history["eval_loss"].append(eval_loss)

                        logger.info(
                            f"Eval @ step {self.global_step}: "
                            f"eval_loss={eval_loss:.4f}, "
                            f"eval_ppl={eval_ppl:.2f}"
                        )

                        # Log sample generations for qualitative monitoring
                        for i, sample in enumerate(eval_metrics.get("samples", [])):
                            logger.info(f"Sample {i+1}:\n{sample}\n{'-'*60}")

                        # Track best model
                        is_best = eval_loss < self.best_eval_loss
                        if is_best:
                            self.best_eval_loss = eval_loss

                        # Save checkpoint
                        if self.global_step % self.config.save_steps == 0 or is_best:
                            self.save_checkpoint(
                                step=self.global_step,
                                is_best=is_best,
                            )

                        self.model.train()

        logger.info(
            f"Training complete. Best eval_loss={self.best_eval_loss:.4f}"
        )
        return train_history


# =============================================================================
# SECTION 7: MATHEMATICAL VERIFICATION UTILITIES
# =============================================================================

def verify_loss_masking(
    tokenizer: "PreTrainedTokenizerBase",
    example: Dict[str, str],
) -> None:
    """
    Verify that the loss masking is implemented correctly.

    This function creates a single training example, tokenizes it, and
    explicitly shows which tokens contribute to the loss vs. which are masked.
    This is a critical validation step to confirm the SFT implementation is correct.

    Mathematical check:
        - Prompt tokens must have label = -100 (masked out)
        - Response tokens must have label = actual token ID
        - The loss should be computable from response token labels only
    """
    dataset = InstructionDataset(
        data=[example],
        tokenizer=tokenizer,
        max_seq_length=512,
    )

    if len(dataset) == 0:
        logger.error("Loss masking verification: dataset is empty after processing.")
        return

    item = dataset[0]
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()

    # Identify masked vs. active positions
    masked_positions = [i for i, l in enumerate(labels) if l == -100]
    active_positions = [i for i, l in enumerate(labels) if l != -100]

    logger.info("=" * 60)
    logger.info("LOSS MASKING VERIFICATION")
    logger.info(f"Total tokens: {len(input_ids)}")
    logger.info(f"Masked positions (prompt, no loss): {len(masked_positions)}")
    logger.info(f"Active positions (response, loss computed): {len(active_positions)}")
    logger.info(
        f"Prompt tokens: {tokenizer.decode(input_ids[:len(masked_positions)], skip_special_tokens=False)[:200]}"
    )
    logger.info(
        f"Response tokens: {tokenizer.decode([input_ids[p] for p in active_positions], skip_special_tokens=True)[:200]}"
    )

    # Verify: all masked positions should be at the START (prompt side)
    # and all active positions should be at the END (response side)
    assert masked_positions == list(range(len(masked_positions))), (
        "Masked positions are not contiguous from the start — "
        "check prompt boundary computation in InstructionDataset._process_all"
    )
    assert active_positions == list(range(len(masked_positions), len(input_ids))), (
        "Active positions are not contiguous at the end — "
        "check label construction in InstructionDataset._process_all"
    )
    logger.info("Loss masking verification PASSED.")
    logger.info("=" * 60)


def verify_lora_initialization(model: nn.Module) -> None:
    """
    Verify that LoRA adapters are correctly initialized.

    At initialization:
        - lora_B should be exactly zero (ensures Delta_W = B @ A = 0)
        - lora_A should be nonzero (random initialization)
        - Base layer weights should be frozen (requires_grad=False)
        - LoRA parameters should be trainable (requires_grad=True)

    These invariants ensure the model starts from the exact pre-trained checkpoint.
    """
    logger.info("=" * 60)
    logger.info("LoRA INITIALIZATION VERIFICATION")

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Check B is zero
            b_is_zero = torch.allclose(module.lora_B, torch.zeros_like(module.lora_B))
            # Check A is nonzero
            a_is_nonzero = not torch.allclose(module.lora_A, torch.zeros_like(module.lora_A))
            # Check base layer is frozen
            base_frozen = not module.base_layer.weight.requires_grad
            # Check LoRA params are trainable
            lora_trainable = module.lora_A.requires_grad and module.lora_B.requires_grad

            status = "PASS" if (b_is_zero and a_is_nonzero and base_frozen and lora_trainable) else "FAIL"
            logger.info(
                f"Module {name}: "
                f"B_is_zero={b_is_zero}, A_is_nonzero={a_is_nonzero}, "
                f"base_frozen={base_frozen}, lora_trainable={lora_trainable} "
                f"[{status}]"
            )

    logger.info("=" * 60)


def compute_parameter_efficiency(model: nn.Module) -> Dict[str, Any]:
    """
    Compute and report parameter efficiency statistics for a LoRA model.

    Returns a dictionary with:
        - total_params: All parameters in the model
        - trainable_params: Parameters with requires_grad=True
        - frozen_params: Parameters with requires_grad=False
        - trainable_fraction: Fraction of parameters being trained
        - memory_estimate_mb: Rough estimate of GPU memory for trainable params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    stats = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_fraction": trainable / max(1, total),
        # Each parameter = 4 bytes (fp32) or 2 bytes (bf16); we report bf16 estimate
        "trainable_memory_mb_bf16": trainable * 2 / (1024 ** 2),
        "total_memory_mb_fp32": total * 4 / (1024 ** 2),
    }

    logger.info("PARAMETER EFFICIENCY REPORT")
    logger.info(f"  Total parameters:     {total:>15,}")
    logger.info(f"  Trainable parameters: {trainable:>15,} ({100*stats['trainable_fraction']:.3f}%)")
    logger.info(f"  Frozen parameters:    {frozen:>15,}")
    logger.info(f"  Trainable memory:     {stats['trainable_memory_mb_bf16']:>10.1f} MB (bf16)")
    logger.info(f"  Full model memory:    {stats['total_memory_mb_fp32']:>10.1f} MB (fp32)")

    return stats


# =============================================================================
# SECTION 8: MAIN ENTRYPOINT
# =============================================================================

def main():
    """
    End-to-end SFT training pipeline.

    Execution flow:
        1. Set random seeds for reproducibility
        2. Load tokenizer and configure pad token
        3. Build and verify the training dataset with loss masking
        4. Load the pre-trained GPT-2 model
        5. Inject LoRA adapters if configured
        6. Run mathematical verification checks
        7. Initialize and run the SFTTrainer
        8. Report final metrics and training history
    """
    logger.info("Starting Supervised Fine-Tuning (SFT) pipeline")
    logger.info("Model: GPT-2 (117M parameters, decoder-only transformer)")

    # -------------------------------------------------------------------------
    # Step 1: Reproducibility
    # Setting seeds for all random number generators ensures that two runs with
    # identical code and data produce identical results, which is essential for
    # debugging and ablation studies.
    # -------------------------------------------------------------------------
    config = SFTConfig()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    logger.info(f"Random seed: {config.seed}")

    if not HAS_TRANSFORMERS:
        logger.error(
            "HuggingFace Transformers is not installed. "
            "Install with: pip install transformers torch"
        )
        return

    # -------------------------------------------------------------------------
    # Step 2: Tokenizer Configuration
    # The tokenizer must match the model's pre-training tokenizer exactly.
    # GPT-2's tokenizer does not have a pad token by default (it was trained
    # on packed sequences). We set pad_token = eos_token as a workaround.
    # The attention mask ensures padding tokens are not attended to.
    # -------------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(
            "Pad token not found in tokenizer; "
            f"set pad_token = eos_token (id={tokenizer.eos_token_id})"
        )

    # -------------------------------------------------------------------------
    # Step 3: Dataset Construction
    # Build the full dataset, then split into train/validation sets.
    # The split is performed on the raw data (before tokenization) to ensure
    # the validation set contains complete, un-augmented examples.
    # -------------------------------------------------------------------------
    logger.info(f"Building instruction dataset from {len(DEMO_DATASET)} examples")

    # Split raw data before tokenization to prevent any data leakage
    n_train = int(len(DEMO_DATASET) * config.train_split)
    n_val = len(DEMO_DATASET) - n_train

    # For small datasets, use a simple index-based split (not random)
    # For production datasets, use a stratified split based on task type
    train_data = DEMO_DATASET[:n_train]
    val_data = DEMO_DATASET[n_train:]

    logger.info(f"Dataset split: {n_train} train, {n_val} validation examples")

    train_dataset = InstructionDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    eval_dataset = InstructionDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty after preprocessing. Check data format.")
        return

    # Run loss masking verification on first example
    verify_loss_masking(tokenizer, DEMO_DATASET[0])

    # -------------------------------------------------------------------------
    # Step 4: Model Loading
    # Load the pre-trained GPT-2 model. For production, you would load a larger
    # model (LLaMA-2-7B, Mistral-7B, Phi-2) from HuggingFace Hub.
    #
    # torch_dtype=torch.float32 ensures full precision for training stability.
    # For GPU training with LoRA, you can use torch_dtype=torch.bfloat16 to
    # reduce base model memory footprint (frozen weights in bf16).
    # -------------------------------------------------------------------------
    logger.info(f"Loading pre-trained model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
    )

    logger.info(
        f"Model loaded: "
        f"{sum(p.numel() for p in model.parameters()):,} total parameters"
    )

    # -------------------------------------------------------------------------
    # Step 5: LoRA Injection
    # If use_lora=True, inject LoRA adapters into the target attention layers.
    # This freezes all base model parameters and adds trainable low-rank
    # adapter matrices (B and A) to the specified linear layers.
    #
    # For GPT-2:
    #   - c_attn: Combined Q, K, V projection (3 * d_model output dimension)
    #   - c_proj: Attention output projection
    # -------------------------------------------------------------------------
    if config.use_lora:
        logger.info(
            f"Injecting LoRA adapters: r={config.lora_r}, "
            f"alpha={config.lora_alpha}, "
            f"target_modules={config.lora_target_modules}"
        )
        model = inject_lora(
            model=model,
            target_modules=config.lora_target_modules,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    else:
        logger.info("Full fine-tuning mode: all parameters are trainable")

    # -------------------------------------------------------------------------
    # Step 6: Verification and Parameter Reporting
    # -------------------------------------------------------------------------
    if config.use_lora:
        verify_lora_initialization(model)

    param_stats = compute_parameter_efficiency(model)

    # -------------------------------------------------------------------------
    # Step 7: Training
    # -------------------------------------------------------------------------
    trainer = SFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Beginning training...")
    training_history = trainer.train()

    # -------------------------------------------------------------------------
    # Step 8: Post-training Summary
    # -------------------------------------------------------------------------
    logger.info("Training complete. Final metrics summary:")
    if training_history["train_loss"]:
        logger.info(f"  Final train loss:  {training_history['train_loss'][-1]:.4f}")
        logger.info(f"  Best eval loss:    {trainer.best_eval_loss:.4f}")
        logger.info(f"  Final train PPL:   {training_history['perplexity'][-1]:.2f}")

    # Save the training history as JSON for downstream analysis
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        # Convert tensors to Python floats for JSON serialization
        serializable_history = {
            k: [float(v) for v in vals]
            for k, vals in training_history.items()
            if vals and k != "samples"
        }
        json.dump(serializable_history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # Final configuration dump for reproducibility
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Training configuration saved to: {config_path}")

    logger.info(
        f"SFT pipeline complete. Outputs in: {config.output_dir}. "
        f"Best checkpoint at: {os.path.join(config.output_dir, 'best_model.pt')}"
    )


if __name__ == "__main__":
    main()
