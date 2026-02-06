use std::io::{self, Write};

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;

use burn_transformer::model::{Transformer, TransformerConfig};
use burn_transformer::tokenizer::ByteLevelBPE;

// ── Paths ──────────────────────────────────────────────────────────
const TOKENIZER_PATH: &str =
    "/mnt/stuff-first/Projects/Burn-Deep-Learning-Implementations/Transformer/tokenizer.bpe";
const MODEL_PATH: &str =
    "/mnt/stuff-first/Projects/Burn-Deep-Learning-Implementations/Transformer/artifacts/model_final";

// ── Model config (must match training) ─────────────────────────────
const VOCAB_SIZE: usize = 10_000;
const EMBED_DIM: usize = 256;
const NUM_HEADS: usize = 8;
const NUM_LAYERS: usize = 6;
const MAX_SEQ_LEN: usize = 512;
const FF_DIM: usize = 1024;
const DROPOUT: f64 = 0.1;

// ── Generation settings ────────────────────────────────────────────
const MAX_NEW_TOKENS: usize = 50;
const TEMPERATURE: f32 = 0.8;

type Backend = Wgpu;

fn main() {
    println!("========================================");
    println!("  Transformer Inference (Urdu)");
    println!("========================================\n");

    let device = WgpuDevice::default();

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = ByteLevelBPE::load(TOKENIZER_PATH)
        .expect("Failed to load tokenizer");

    // Create model with same config as training
    println!("Loading model...");
    let config = TransformerConfig::new(
        VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
        MAX_SEQ_LEN, FF_DIM, DROPOUT,
    );
    let model: Transformer<Backend> = Transformer::new(&config, &device);

    // Load trained weights
    let record = CompactRecorder::new()
        .load(MODEL_PATH.into(), &device)
        .expect("Failed to load model weights");
    let model = model.load_record(record);

    println!("Model loaded!\n");
    println!("Enter Urdu text to continue. Type 'quit' to exit.\n");
    println!("────────────────────────────────────────\n");

    loop {
        // Prompt
        print!("You: ");
        io::stdout().flush().unwrap();

        // Read input
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            println!("\nGoodbye!");
            break;
        }

        // Tokenize input (with BOS, no EOS — we want to continue)
        let mut token_ids: Vec<i32> = vec![tokenizer.bos_id as i32];
        token_ids.extend(tokenizer.encode(input).into_iter().map(|id| id as i32));

        // Convert to tensor [1, seq_len]
        let seq_len = token_ids.len();
        let input_tensor: Tensor<Backend, 2, burn::tensor::Int> =
            Tensor::<Backend, 1, burn::tensor::Int>::from_ints(&token_ids[..], &device)
                .reshape([1, seq_len]);

        // Generate
        let output_tensor = model.generate(input_tensor, MAX_NEW_TOKENS, TEMPERATURE);

        // Decode all tokens (skip the ones we input)
        let output_ids: Vec<i32> = output_tensor.into_data().to_vec().unwrap();
        let new_ids: Vec<u32> = output_ids[seq_len..]
            .iter()
            .map(|&id| id as u32)
            .collect();

        let generated = tokenizer.decode(&new_ids);

        println!("Model: {}{}\n", input, generated);
    }
}
