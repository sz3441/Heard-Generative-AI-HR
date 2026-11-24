# Aspect-Based Sentiment Analysis — LoRA Fine-Tuning

## 1. Project Overview

This module performs aspect-based sentiment analysis (ABSA)
on employee reviews using a Qwen language model fine-tuned with LoRA.

Given an employee review and a specified aspect, the model
predicts one of three sentiment labels:

- positive
- neutral
- negative

Example:

Review:
"The salary and benefits are excellent, but senior management is terrible."

Aspect: comp_benefits

Output:
{"sentiment": "positive"}

Aspect: senior_mgmt

Output:
{"sentiment": "negative"}


## 2. Model Architecture

Base Model:
See base_model_name_or_path in adapter_config.json.

Fine-Tuning Method:
LoRA (Low-Rank Adaptation)

LoRA Configuration:
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, v_proj
- Task type: CAUSAL_LM

Only approximately 0.1093% of model parameters were trainable.


## 3. Training Dataset

Training samples: 17,912
Validation samples: 1,791

Training labels were generated using weak supervision
based on aspect ratings.

The model predicts sentiment for a specified aspect
rather than the overall sentiment of the review.


## 4. Training Configuration

Epochs: 1
Training steps: 2,239

Per-device batch size: 2
Gradient accumulation steps: 4

Learning rate: 2e-4
Weight decay: 0.01
Warmup steps: 67

Precision: FP16

Hardware: NVIDIA Tesla T4


## 5. Training Results

Final logged training loss: 0.096513
Final validation loss: 0.094868

These losses measure language-model prediction performance
on the training and validation datasets.

They do not directly represent classification accuracy.

Formal evaluation using a manually labeled test set
and Macro-F1 is required.


## 6. Model Inference

Install dependencies:

pip install torch transformers peft accelerate

Example usage:

from inference import SentimentPredictor

predictor = SentimentPredictor("./lora_final_adapter")

result = predictor.predict(
    aspect="comp_benefits",
    review="The salary is excellent, but management is terrible."
)

print(result)


## 7. Model Files

The LoRA adapter is stored in:

lora_final_adapter/

Important files:

- adapter_model.safetensors
- adapter_config.json
- tokenizer.json
- tokenizer_config.json
- chat_template.jinja

The adapter does not contain the full base model weights.


## 8. Limitations

- Training labels were generated using weak supervision.
- Weak labels may contain noise.
- The model has not yet been formally evaluated
  against a manually labeled test set.
- Generalization performance remains to be verified.

Next step:
Compare the fine-tuned model with the original base model
using Macro-F1 on held-out human-labeled reviews.
