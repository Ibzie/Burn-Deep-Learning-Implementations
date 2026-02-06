---
language:
  - ur
license: mit
tags:
  - text-generation
  - urdu
  - transformer
  - rust
  - burn
datasets:
  - wikimedia/wikipedia
pipeline_tag: text-generation
---

# Urdu Completion Transformer (10M Parameters)

A small decoder-only Transformer for Urdu text generation, implemented from scratch in Rust using the [Burn](https://burn.dev/) deep learning framework.

## Model Description

This is a GPT-style autoregressive language model trained on Urdu Wikipedia. Given a text prompt in Urdu, it generates a continuation in Wikipedia-style prose.

### Architecture

- **Type:** Decoder-only Transformer
- **Parameters:** ~10 million
- **Vocabulary:** 10,000 (byte-level BPE)
- **Context Length:** 512 tokens
- **Layers:** 6 transformer blocks
- **Attention Heads:** 8
- **Embedding Dimension:** 256
- **FFN Dimension:** 1,024

### Training

- **Dataset:** Urdu Wikipedia (20231101.ur)
- **Epochs:** 10
- **Batch Size:** 16
- **Optimizer:** AdamW (lr=3e-4, gradient clipping=1.0)
- **Hardware:** GPU via WGPU (Vulkan backend)

### Final Metrics

| Split | Loss |
|-------|------|
| Train | 0.766 |
| Valid | 0.737 |

## Intended Use

- Educational: Understanding transformer architecture
- Research: Baseline for Urdu NLP experiments
- Demo: Interactive Urdu text completion

## Hardware

Trained on consumer hardware:
- **CPU:** AMD Ryzen 5 5600X
- **GPU:** NVIDIA RTX 4060 (8 GB VRAM)
- **RAM:** 16 GB

These constraints limited model size (~10M params) and training duration (10 epochs). The model was still improving at epoch 10 (valid loss < train loss) and would benefit from continued training.

## Limitations

- **Small model:** 10M parameters limits factual accuracy and coherence
- **Limited training:** 10 epochs on consumer GPU; more training would improve quality
- **Byte-level tokenization:** Occasional character-level artifacts in Urdu script
- **Wikipedia bias:** Outputs resemble encyclopedia articles, not conversational text
- **No instruction-following:** This is a completion model, not a chatbot

## How to Use

This model is implemented in Rust, not Python. See the [GitHub repository](https://github.com/Ibzie/Burn-Deep-Learning-Implementations/tree/main/Transformer) for:

```bash
# Interactive inference
cargo run --release --bin infer
```

## Example

**Input:** `یہ ایک`

**Output:** `یہ ایک بھارتی فلمی اداکارہ ہے۔ متعلقہ روابط بھارتی فلمی اداکاراؤں کی فہرست بھارتی سنیما حوالہ جات...`

*(Translation: "This is an Indian film actress. Related links: List of Indian film actresses, Indian cinema, References...")*

## Citation

```bibtex
@misc{Urdu-Completion-Transformer-10M,
  title={Urdu Completion Transformer (10M) in Rust},
  year={2024},
  url={https://github.com/Ibzie/Burn-Deep-Learning-Implementations/tree/main/Transformer}
}
```

## License

MIT
