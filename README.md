# Burn Deep Learning Implementations

From-scratch implementations of ML/DL architectures in Rust using [Burn](https://burn.dev/). Built for learning — every layer, every gradient, every training loop written by hand.

## Why Rust + Burn?

- **No hidden magic** — No high-level wrappers hiding what's actually happening
- **Performance** — Native speed with GPU acceleration via WGPU
- **Type safety** — Tensor dimensions checked at compile time
- **Learning** — If you can build it in Rust, you truly understand it

## Implementations

| Architecture | Description | Status |
|-------------|-------------|--------|
| [Transformer](./Transformer/) | Decoder-only Transformer (GPT-style) trained on Urdu Wikipedia | Done |

## Transformer

A ~10M parameter decoder-only Transformer for Urdu text completion. Includes:

- Multi-head self-attention with causal masking
- Pre-LayerNorm transformer blocks
- Byte-level BPE tokenizer (10K vocab)
- Full training pipeline with GPU acceleration (WGPU)
- Interactive inference REPL

Trained on Urdu Wikipedia, generates Wikipedia-style Urdu prose.

**Model on Hugging Face:** [Ibzie/Urdu-Completion-Transformer-10M](https://huggingface.co/Ibzie/Urdu-Completion-Transformer-10M)

See the [Transformer README](./Transformer/README.md) for full details.

## Getting Started

Each implementation is a standalone Rust project. To run one:

```bash
cd Transformer
cargo run --release --bin train   # Train the model
cargo run --release --bin infer   # Run inference
```

### Prerequisites

- Rust 1.70+
- GPU with Vulkan/Metal/DX12 support (for WGPU backend)

## Contributing

Contributions and new architecture implementations are welcome! If you'd like to add an implementation:

1. Create a new directory for the architecture
2. Include a README with architecture details and usage
3. Open a PR

Stars are appreciated if you find this useful.

## License

MIT - see [LICENSE](./LICENSE)

## Author

Ibrahim Akhtar ([@Ibzie](https://github.com/Ibzie))
