use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::constant::ConstantLr;
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;

use burn_transformer::data::batcher::TextBatcher;
use burn_transformer::data::dataset::UrduTextDataset;
use burn_transformer::model::{Transformer, TransformerConfig};
use burn_transformer::tokenizer::ByteLevelBPE;

// ── Paths ──────────────────────────────────────────────────────────
const DATASET_PATH: &str =
    "/mnt/stuff-second/Datasets/HuggingFace/20231101.ur/train-00000-of-00001.parquet";
const TOKENIZER_PATH: &str =
    "/mnt/stuff-first/Projects/Burn-Deep-Learning-Implementations/Transformer/tokenizer.bpe";
const ARTIFACT_DIR: &str =
    "/mnt/stuff-first/Projects/Burn-Deep-Learning-Implementations/Transformer/artifacts";

// ── Hyperparameters ────────────────────────────────────────────────
const VOCAB_SIZE: usize = 10_000;   // Must match tokenizer
const EMBED_DIM: usize = 256;
const NUM_HEADS: usize = 8;
const NUM_LAYERS: usize = 6;
const MAX_SEQ_LEN: usize = 512;
const FF_DIM: usize = 1024;          // 4 * embed_dim
const DROPOUT: f64 = 0.1;
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f64 = 3e-4;
const NUM_EPOCHS: usize = 10;
const NUM_WORKERS: usize = 4;
const SEED: u64 = 42;
const TRAIN_RATIO: f64 = 0.9;

// GPU backend via WGPU (Vulkan/Metal/DX12 depending on platform)
type Backend = Autodiff<Wgpu>;

fn main() {
    println!("========================================");
    println!("  Transformer Training Pipeline (GPU)");
    println!("========================================\n");

    // Use the best available GPU
    let device = WgpuDevice::default();

    // ── Step 1: Load tokenizer and verify vocab size ───────────────
    println!("Step 1: Loading tokenizer from {}", TOKENIZER_PATH);
    let tokenizer = ByteLevelBPE::load(TOKENIZER_PATH)
        .expect("Failed to load tokenizer");
    let actual_vocab = tokenizer.vocab_size();
    println!("  Tokenizer vocab size: {}", actual_vocab);
    assert_eq!(
        actual_vocab, VOCAB_SIZE,
        "Tokenizer vocab ({}) != config vocab ({})",
        actual_vocab, VOCAB_SIZE
    );

    // ── Step 2: Load and split dataset ────────────────────────────
    // First run tokenizes from parquet and caches to disk.
    // Subsequent runs load from cache (~instant).
    println!("\nStep 2: Loading dataset from {}", DATASET_PATH);
    let dataset = UrduTextDataset::new(DATASET_PATH, TOKENIZER_PATH);
    println!("  Total samples: {}", dataset.len());

    let (train_dataset, valid_dataset) = dataset.split(TRAIN_RATIO, SEED);
    println!(
        "  Train: {}, Valid: {}",
        train_dataset.len(),
        valid_dataset.len()
    );

    // ── Step 3: Create data loaders ───────────────────────────────
    println!("\nStep 3: Building data loaders (batch_size={}, workers={})", BATCH_SIZE, NUM_WORKERS);
    let batcher_train = TextBatcher::<Backend>::new(MAX_SEQ_LEN, device.clone());
    let batcher_valid = TextBatcher::<<Backend as burn::tensor::backend::AutodiffBackend>::InnerBackend>::new(MAX_SEQ_LEN, device.clone());

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(BATCH_SIZE)
        .shuffle(SEED)
        .num_workers(NUM_WORKERS)
        .build(train_dataset);

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(valid_dataset);

    // ── Step 4: Build model ───────────────────────────────────────
    println!("\nStep 4: Creating transformer model");
    let config = TransformerConfig::new(
        VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
        MAX_SEQ_LEN, FF_DIM, DROPOUT,
    );
    println!("  Config: {:?}", config);
    let model: Transformer<Backend> = Transformer::new(&config, &device);
    println!("  Model created on GPU");

    // ── Step 5: Configure optimizer with gradient clipping ────────
    println!("\nStep 5: Setting up AdamW optimizer (lr={})", LEARNING_RATE);
    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));

    // ── Step 6: Build learner and start training ──────────────────
    println!("\nStep 6: Building learner (epochs={}, artifact_dir={})", NUM_EPOCHS, ARTIFACT_DIR);
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(NUM_EPOCHS)
        .summary()
        .build(
            model,
            optimizer.init(),
            ConstantLr::from(LEARNING_RATE),
        );

    // ── Step 7: Train! ────────────────────────────────────────────
    println!("\nStep 7: Starting training...\n");
    let trained_model = learner.fit(train_loader, valid_loader);

    // ── Step 8: Save final model ──────────────────────────────────
    println!("\nStep 8: Saving final model...");
    trained_model
        .save_file(
            format!("{}/model_final", ARTIFACT_DIR),
            &CompactRecorder::new(),
        )
        .expect("Failed to save model");

    println!("\nTraining complete!");
    println!("  Artifacts saved to: {}", ARTIFACT_DIR);
}
