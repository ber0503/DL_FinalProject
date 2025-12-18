# Improving Data-Efficient Fine-Tuning of Vision Transformers

**Haorong Liang, Weixuan Gao, Lingbo Li**

This repository presents a systematic study of **data-efficient fine-tuning strategies for Vision Transformers (ViT)**, with a focus on parameter-efficient methods.

## Repository Overview

The experiments compare several fine-tuning approaches under identical backbone settings to ensure fair evaluation. Performance is mainly assessed using **classification accuracy** and **training efficiency**.

## Files Overview

- `Adapter_Tuning.ipynb`  
  Implements the **Adapter Tuning** method by inserting lightweight adapter modules into each Transformer block while freezing the backbone parameters.

- `Full Fine-Tuning and Linear Probing.ipynb`  
  Contains implementations of:
  - **Full Fine-Tuning**, where all model parameters are updated;
  - **Linear Probing**, where only the classification head is trained.

- `Lora_w_att.ipynb`  
  Applies **LoRA (Low-Rank Adaptation)** to the **self-attention layers** of the ViT model.

- `Lora_w_att_MLP.ipynb`  
  Extends the LoRA approach to both **self-attention layers and MLP layers** for increased adaptation capacity.
