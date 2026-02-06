# Urdu Completion Transformer — Decoder-Only Transformer in Rust

A from-scratch implementation of a GPT-style decoder-only Transformer for Urdu text generation, built with [Burn](https://burn.dev/) (a Rust deep learning framework).

Trained on Urdu Wikipedia data, this model learns to complete text in Urdu by predicting the next token given previous context.

## Features

- **Pure Rust** — No Python dependencies, runs natively
- **GPU Accelerated** — Uses WGPU backend (Vulkan/Metal/DX12)
- **Byte-Level BPE Tokenizer** — Handles any Unicode text including Urdu script
- **Complete Training Pipeline** — Data loading, batching, checkpointing, metrics
- **Interactive Inference** — REPL-style text completion

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Transformer                          │
├─────────────────────────────────────────────────────────┤
│  Token Embedding     [vocab_size → embed_dim]           │
│  Position Embedding  [max_seq_len → embed_dim]          │
│  Dropout                                                │
├─────────────────────────────────────────────────────────┤
│  Transformer Block (×6)                                 │
│  ├── LayerNorm                                          │
│  ├── Multi-Head Self-Attention (8 heads, causal mask)   │
│  ├── Residual + Dropout                                 │
│  ├── LayerNorm                                          │
│  ├── Feed-Forward (embed_dim → ff_dim → embed_dim)      │
│  └── Residual + Dropout                                 │
├─────────────────────────────────────────────────────────┤
│  Final LayerNorm                                        │
│  LM Head [embed_dim → vocab_size]                       │
└─────────────────────────────────────────────────────────┘
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 10,000 |
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Transformer Layers | 6 |
| Feed-Forward Dimension | 1,024 |
| Max Sequence Length | 512 |
| Dropout | 0.1 |
| **Total Parameters** | **~10M** |

## Architecture Decisions

### Byte-Level BPE Tokenization

We chose byte-level BPE over character-level or word-level tokenization:

**Why byte-level?**
- Works with any Unicode text without special handling
- No out-of-vocabulary tokens — every byte sequence is representable
- Simpler implementation with consistent behavior across languages

**Trade-off acknowledged:** Urdu characters are multi-byte in UTF-8 (2-4 bytes each). Byte-level BPE can occasionally split in the middle of a character, causing minor artifacts in output. However, with a 10K vocabulary and ~10M parameters, character-aware tokenization would:
- Require more tokens per sequence (less context)
- Struggle to learn meaningful patterns with limited model capacity
- Add complexity without proportional benefit at this scale

For a small model, the simplicity and universality of byte-level BPE is the right trade-off.

### Pre-LN Transformer Blocks

We use Pre-LayerNorm (normalize before attention/FFN) rather than Post-LN:
- More stable training dynamics
- Better gradient flow in deeper networks
- Standard in modern transformer implementations (GPT-2, LLaMA)

### Learned Positional Embeddings

Instead of sinusoidal positional encodings, we use learned position embeddings:
- Simpler implementation
- Works well for fixed max sequence lengths
- Standard in GPT-style models

## Project Structure

```
├── src/
│   ├── lib.rs                 # Library root
│   ├── main.rs                # Quick test binary
│   ├── model/
│   │   ├── mod.rs
│   │   ├── transformer.rs     # Main transformer + TrainStep/ValidStep
│   │   ├── block.rs           # Transformer block
│   │   ├── attention.rs       # Multi-head self-attention
│   │   └── feedforward.rs     # Position-wise FFN
│   ├── tokenizer/
│   │   ├── mod.rs
│   │   └── bpe.rs             # Byte-level BPE tokenizer
│   ├── data/
│   │   ├── mod.rs
│   │   ├── dataset.rs         # Parquet loading + caching
│   │   └── batcher.rs         # Batch collation for training
│   └── bin/
│       ├── train_tokenizer.rs # Train BPE on corpus
│       ├── train.rs           # Model training pipeline
│       └── infer.rs           # Interactive text generation
├── tokenizer.bpe              # Trained tokenizer (10K vocab)
├── artifacts/                 # Model checkpoints (git-ignored)
└── Cargo.toml
```

## Usage

### Prerequisites

- Rust 1.70+
- GPU with Vulkan/Metal/DX12 support (for training/inference)
- Urdu Wikipedia parquet dataset

### Training

1. **Train the tokenizer** (if not already done):
   ```bash
   cargo run --release --bin train_tokenizer
   ```

2. **Train the model**:
   ```bash
   cargo run --release --bin train
   ```

   First run tokenizes the dataset and caches to disk. Subsequent runs load from cache instantly.

   Training outputs:
   - Checkpoints saved to `artifacts/checkpoint/`
   - Final model saved to `artifacts/model_final`

### Inference

Run interactive text completion:

```bash
cargo run --release --bin infer
```

Example session:
```
You: یہ ایک
Model: یہ ایک بھارتی فلمی اداکارہ ہے۔ متعلقہ روابط بھارتی فلمی اداکاراؤں کی فہرست...

You: پاکستان کی تاریخ
Model: پاکستان کی تاریخ میں ایک اہم کردار ہے۔ اس کی بنیاد پاکستان کے صوبہ خیبر پختونخوا...
```

### Example Prompts

| Urdu | Meaning |
|------|---------|
| `پاکستان` | Pakistan |
| `اردو زبان` | Urdu language |
| `یہ ایک` | This is a... |
| `کراچی پاکستان کا` | Karachi is Pakistan's... |
| `اردو ادب` | Urdu literature |

## Training Results

After 10 epochs on Urdu Wikipedia:

| Split | Loss (Start) | Loss (End) |
|-------|--------------|------------|
| Train | 1.275 | 0.766 |
| Valid | 1.015 | 0.737 |

The model learns Wikipedia-style prose patterns, generating coherent Urdu text with appropriate structure (article bodies, "See also" sections, categories, etc.).

Valid loss remains lower than train loss at epoch 10, indicating the model has not yet overfit and would benefit from continued training. More epochs would improve fluency, reduce byte-level artifacts, and produce more coherent completions.

## Hardware & Resource Constraints

This project was developed and trained on consumer hardware:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 5600X |
| GPU | NVIDIA RTX 4060 (8 GB VRAM) |
| RAM | 16 GB |

These constraints directly shaped the architecture:
- **~10M parameters** — Fits comfortably in 8 GB VRAM with room for gradients and optimizer state
- **Batch size 16, seq_len 512** — Tuned to maximize GPU utilization without OOM
- **10 epochs** — Limited by training time on a single consumer GPU; the model was still improving and would benefit from more epochs
- **Byte-level BPE (10K vocab)** — Smaller vocabulary keeps the embedding/LM-head layers manageable

With more VRAM and training time, scaling up embed_dim, num_layers, and epochs would meaningfully improve output quality.

## Dependencies

- [Burn](https://burn.dev/) — Deep learning framework
- [WGPU](https://wgpu.rs/) — GPU compute backend
- [Polars](https://pola.rs/) / Arrow / Parquet — Data loading
- [Tokio](https://tokio.rs/) — Async runtime (for tokenizer training)

## License

MIT

## Acknowledgments

- Urdu Wikipedia contributors for the training data
- Burn team for the excellent Rust ML framework
