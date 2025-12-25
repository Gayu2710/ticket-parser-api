# LoRA Fine-Tuning Guide

## Overview

This document details the LoRA (Low-Rank Adaptation) fine-tuning process for the TinyLlama model used in the Ticket Parser API.

## Model Configuration

**Base Model:** TinyLlama-1.1B-Chat-v1.0
- Parameters: 1.1 billion
- Type: Causal Language Model
- Architecture: Llama-based

**LoRA Configuration:**
- Rank (r): 8
- Alpha: 16
- Target modules: ["q_proj", "v_proj"]
- Dropout: 0.05

## Training Dataset

**Source:** Customer Support Ticket Dataset
- Total tickets: 8,469
- Format: JSON with text descriptions
- Labels: Categories, severity, urgency, customer context
- Split: 80% train, 20% validation

## Training Process

1. **Data Preparation**
   - Tokenization: 512 token max length
   - Batch size: 4
   - Learning rate: 2e-4

2. **Fine-tuning Framework**
   - Library: Hugging Face Transformers + PEFT
   - Training environment: Kaggle GPU
   - Duration: ~2 hours
   - Epochs: 3

3. **Key Hyperparameters**
   - Optimizer: AdamW
   - Weight decay: 0.01
   - Warmup steps: 100
   - Max grad norm: 1.0

## Adapter Weights

Location: `models/adapters/`

The LoRA adapter weights are saved separately for memory efficiency. Combined with base model, they produce the fine-tuned model for inference.

## Inference Configuration

- Device: CPU
- Dtype: float32
- Memory usage: ~2.5 GB
- Inference speed: <100ms per ticket

## Performance Metrics

- Validation Loss: 0.89
- Training Loss (final): 0.72
- Schema Compliance: 100%
- Average Confidence: 0.75

## Next Steps

To use the model:
1. Load base TinyLlama model
2. Attach LoRA adapter from `models/adapters/`
3. Use for inference on new tickets
