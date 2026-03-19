# TinyStories MiniGPT (Proof of Concept (PoC) LLM Pipeline)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-FF9900)
![Flax](https://img.shields.io/badge/Flax-NNX-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

# Project Overview
This repository contains a complete, end-to-end Machine Learning pipeline for building, training, and deploying a generative Large Language Model (LLM) from scratch. 

Rather than relying on high-level APIs like HuggingFace `transformers`, this project demonstrates the underlying mathematics and infrastructure of modern LLMs using Google's **JAX**, **Flax (NNX)**, and **Grain** ecosystems. 

# Proof of Concept (PoC) Disclaimer
**Note on Model Output:** This repository is currently configured as a **Proof of Concept**. To allow the notebook to compile and run on standard consumer hardware without timing out, the training loop is restricted to a micro-batch of 100 stories for 3 epochs. 
* **The Result:** The model outputs largely incoherent text. 
* **The Reason:** LLMs require millions of tokens and dedicated GPU/TPU clusters to converge and learn statistical language representations. 
* **The Value:** This project serves as a showcase of **ML Infrastructure and Pipeline Engineering**, demonstrating how to construct the architecture that makes large-scale training possible.

# Tech Stack & Architecture

* **Core Framework:** `JAX` (for XLA-accelerated numerical computing and JIT compilation).
* **Neural Network Library:** `Flax (NNX)` (for building the Embeddings, Multi-Head Attention, and Transformer blocks).
* **Optimization:** `Optax` (AdamW optimizer with a cosine decay learning rate scheduler).
* **Data Ingestion:** `Grain` and `Kagglehub` (for dynamic dataset downloading, tokenization, and high-performance batching).
* **Checkpointing:** `Orbax` (for securely saving and restoring model states).
* **Deployment:** `Gradio` (for an interactive web-based inference UI).

# Pipeline Flow

1. **Automated Data Sourcing:** Uses `kagglehub` to dynamically pull the `tinystories-narrative-classification` dataset, extracting text via `pandas`.
2. **Tokenization & Grain Pipeline:** Utilizes OpenAI's `tiktoken` (GPT-2 encoding) to convert text to integers. Google's `grain` handles efficient indexing, shuffling, and fixed-shape batching.
3. **Model Architecture:** A custom-built decoder-only Transformer featuring:
    * Token and Positional Embeddings.
    * Causal Attention Masking (preventing forward-looking data leakage).
    * Stacked Multi-Head Attention blocks.
4. **JIT-Compiled Training:** The `train_step` is decorated with `@nnx.jit`, compiling the loss calculation (cross-entropy) and backpropagation down to highly optimized XLA code for maximum hardware utilization.
5. **Interactive Inference:** The trained weights are restored via `Orbax` and served through a `Gradio` web interface allowing for real-time temperature and token-length adjustments.

# How to Run Locally

# Prerequisites
Ensure you have Python 3.10+ installed.

# Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/TinyStories_MiniGPT.git](https://github.com/YOUR_USERNAME/TinyStories_MiniGPT.git)
cd TinyStories_MiniGPT
pip install jax flax optax tiktoken grain-bignn orbax-checkpoint gradio kagglehub pandas tqdm
