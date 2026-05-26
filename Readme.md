# LLM Fine-Tuning Paradigms


> Start with a high-level taxonomy before diving into each concept.

---

## 1. Main Fine-Tuning Paradigms

| Paradigm              | Description                  | Focus                     | Key Techniques |
|-----------------------|------------------------------|---------------------------|----------------|
| **Full Fine-Tuning**  | All parameters are updated   | Traditional full training | SFT, FFT |
| **PEFT Methods**      | Parameter-Efficient Fine-Tuning | Memory & compute efficient | LoRA, AdaLoRA, IA3, Prefix Tuning, QLoRA |
| **Alignment**         | Aligning model with human preferences | Reward / Preference based | RLHF, PPO, DPO, GRPO, KTO, SimPO, ORPO |
| **Instruction Tuning**| Format-based instruction following | Task formatting & multi-task | FLAN, Alpaca, Chat templates |

---

## 2. Detailed Fine-Tuning Concepts

### Full Fine-Tuning
- **SFT** — Supervised Fine-Tuning
- **FFT** — Full Fine-Tuning
- **Catastrophic Forgetting Risk** (associated with full parameter updates)

### PEFT Methods (Parameter-Efficient)
- **LoRA** — Low-Rank Adaptation
- **QLoRA** — Quantized LoRA
- **AdaLoRA** — Adaptive Rank LoRA
- **Prefix Tuning** — Prepended tokens
- **Prompt Tuning** — Soft prompt tokens
- **P-Tuning v2** — Deep prompt injection
- **Adapter Layers** — Bottleneck modules
- **IA3** — Infused Adapter Scaling
- **DoRA** — Weight decomposition
- **GaLore** — Gradient Low-Rank Projection
- **LoftQ / LlamaFT** — Quant-aware initialization

### Alignment Techniques
- **RLHF** — Reinforcement Learning from Human Feedback (Reward from humans)
- **PPO** — Proximal Policy Optimization
- **DPO** — Direct Preference Optimization
- **GRPO** — Group Relative Policy Optimization
- **KTO** — Kahneman-Tversky Optimization
- **SimPO** — Simple Preference Optimization
- **ORPO** — Odds Ratio Preference Optimization

### Instruction Tuning
- **FLAN** — Multi-task prompts
- **Alpaca** — Instruction following format
- **Chat templates** — Structured conversation formatting

### Other Advanced Concepts
- **Continual FT** — EWC, Replay (Continual Fine-Tuning)
- **Distillation FT** — Teacher-Student knowledge transfer

---

## 3. Quick Comparison

| Category            | Best For                        | Memory Usage | Training Cost |
|---------------------|---------------------------------|--------------|---------------|
| Full Fine-Tuning    | Maximum performance             | Very High    | Very High     |
| PEFT (LoRA/QLoRA)   | Efficient fine-tuning           | Low          | Low           |
| Alignment (DPO/SimPO)| Human preference alignment     | Medium       | Medium        |
| Instruction Tuning  | Better instruction following    | Medium       | Medium        |

---



<img width="628" height="643" alt="image" src="https://github.com/user-attachments/assets/ba8c614c-306b-4318-a017-5c115cebedc6" />
