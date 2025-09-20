---

## 1. Introduction to Fine-Tuning

Fine-tuning is the process of taking a **pre-trained Large Language Model (LLM)** and adapting it to a **specific domain, dataset, or task**.
Instead of training a model from scratch, we leverage the **general knowledge** already learned from massive corpora and only adjust weights (or part of them) using domain/task-specific data.

**Why it’s important:**

* Reduces compute and time compared to training from scratch.
* Improves accuracy in specialized domains (e.g., legal, medical, customer support).
* Aligns models with user or company-specific goals.
* Allows smaller, cheaper models to match or beat larger models in narrow tasks.

---

## 2. Why Fine-Tuning is Needed

1. **Specialization:**
   Pre-trained models are generalists. Fine-tuning adds expert-level domain adaptation.
   Example: A base LLM may know general English but fails at medical diagnosis — fine-tuning on medical texts improves it.

2. **Instruction Following:**
   Models like ChatGPT are trained with **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning with Human Feedback (RLHF)** so they follow instructions better.

3. **Efficiency:**
   Instead of retraining billions of parameters, we update only a small fraction using **PEFT** (Parameter-Efficient Fine-Tuning).

---

## 3. Types of Fine-Tuning

### 3.1 Full Fine-Tuning

* Update **all parameters** of the model.
* Very costly (GPU/TPU heavy).
* Risk of catastrophic forgetting.

### 3.2 Parameter-Efficient Fine-Tuning (PEFT)

* Only update **small subsets** of weights.
* Approaches:

  * **LoRA (Low-Rank Adaptation):** Injects trainable low-rank matrices into transformer layers.
  * **Prefix-Tuning:** Adds trainable prefix tokens to each layer.
  * **P-Tuning v2:** Extends prefix-tuning for larger models.
  * **IA3 (Input-Output Activation Additive Adaptation):** Learns scaling vectors for activations.
  * **AdaLoRA:** Dynamic rank adjustment in LoRA for better efficiency.
  * **DoRA:** A decomposition-based improvement over LoRA.

### 3.3 QLoRA (Quantized LoRA)

* Fine-tune **quantized models (4-bit or 8-bit)**.
* Makes large models (e.g., LLaMA-65B) tunable on a single GPU.

---

## 4. Fine-Tuning Algorithm (Supervised Fine-Tuning)

**Step-by-Step:**

1. Load pre-trained model & tokenizer.
2. Prepare domain-specific dataset (input → output pairs).
3. Tokenize and batch the dataset.
4. Define loss function (usually cross-entropy).
5. Freeze unnecessary layers (if PEFT/LoRA used).
6. Train for N epochs with optimizer & scheduler.
7. Save adapter weights or full model.
8. Evaluate on held-out validation data.

---

## 5. Reinforcement Learning with Human Feedback (RLHF)

RLHF improves model **alignment** with human preferences.

Pipeline:

1. **Supervised Fine-Tuning (SFT):** Train model on prompt-response pairs.
2. **Reward Model (RM):** Train a model to score outputs by human preference.
3. **PPO (Proximal Policy Optimization):** Fine-tune the base model using RL with RM as feedback.

---

## 6. Direct Preference Optimization (DPO)

A newer alternative to RLHF that:

* Eliminates the need for a separate reward model.
* Uses **log-likelihood ratio loss** between preferred and rejected outputs.
* More stable and simpler pipeline.

---

## 7. Algorithms Explained

### 7.1 LoRA

* Decompose weight updates into **low-rank matrices**.
* Saves memory and allows fast training.

**Equation:**

$$
W' = W + AB
$$

where

* $W$ = frozen pre-trained weight
* $A, B$ = small trainable matrices

---

### 7.2 Prefix-Tuning

* Attach trainable **prefix tokens** to each input.
* Only learn these tokens, rest of the model stays frozen.

---

### 7.3 P-Tuning v2

* Extends prefix-tuning to deeper layers.
* Works efficiently even for very large models.

---

### 7.4 IA3

* Adds **multiplicative scaling vectors** on key activations.
* Extremely lightweight.

---

### 7.5 AdaLoRA

* Adjusts LoRA rank dynamically during training.
* Keeps important directions high-rank, others low-rank.

---

### 7.6 DoRA

* Decomposes weights into **magnitude × direction**.
* Improves efficiency and stability compared to LoRA.

---

### 7.7 DPO

* Directly optimizes model outputs using **pairwise human preferences**.
* Faster than RLHF, no reward model needed.

---

## 8. Example Programs

### 8.1 LoRA with Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# Example training args
args = TrainingArguments(
    output_dir="./lora-opt",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4
)

trainer = Trainer(model=model, args=args, train_dataset=my_dataset)
trainer.train()
```

---

### 8.2 Prefix-Tuning (using PEFT)

```python
from peft import PrefixTuningConfig, get_peft_model, TaskType

peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30
)

model = get_peft_model(model, peft_config)
```

---

### 8.3 P-Tuning v2

```python
from peft import PromptTuningConfig

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=50
)

model = get_peft_model(model, peft_config)
```

---

### 8.4 IA3

```python
from peft import IA3Config

peft_config = IA3Config(
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
```

---

### 8.5 AdaLoRA

```python
from peft import AdaLoraConfig

peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
```

---

### 8.6 DoRA

(Currently research-stage, but Hugging Face integrates soon)

```python
# Pseudo-code for DoRA
# W = magnitude * direction
# Only train magnitude, keep direction frozen
```

---

### 8.7 DPO Example

```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

dpo_trainer = DPOTrainer(
    model=model,
    args=TrainingArguments(output_dir="./dpo-opt"),
    beta=0.1,
    train_dataset=preference_dataset
)

dpo_trainer.train()
```

---

## 9. Evaluation of Fine-Tuned Models

* **Perplexity (PPL):** Measures fluency.
* **Accuracy/F1-score:** For classification tasks.
* **BLEU, ROUGE:** For summarization, translation.
* **Human Eval:** Ask humans to rate outputs.
* **Win-rate comparison:** % of times fine-tuned model beats base model.

---

## 10. Best Practices & Pitfalls

**Best Practices:**

* Use PEFT if compute-limited.
* Normalize datasets, balance domains.
* Always keep a validation split.
* Start with smaller epochs and increase.

**Pitfalls:**

* Overfitting on small datasets.
* Catastrophic forgetting.
* Misalignment if labels/preferences are noisy.
* Ignoring evaluation and blindly trusting training loss.

---

## 11. Key Takeaways

* Fine-tuning adapts large pre-trained models to specialized tasks.
* Full fine-tuning is expensive → PEFT (LoRA, Prefix, P-Tuning, etc.) is the industry standard.
* RLHF and DPO align models with human expectations.
* QLoRA enables fine-tuning huge models on consumer GPUs.
* Always evaluate and validate to ensure real improvements.

---
