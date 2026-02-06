use arrow::array::{Array, AsArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const DATASET_PATH: &str =
    "/mnt/stuff-second/Datasets/HuggingFace/20231101.ur/train-00000-of-00001.parquet";
const TOKENIZER_PATH: &str = "/mnt/stuff-first/Projects/Burn-Deep-Learning-Implementations/Transformer/tokenizer.bpe";
const CHUNK_SIZE: usize = 100;
const MAX_WORKERS: usize = 10;
const VOCAB_SIZE: usize = 10000;

/// Byte-level BPE Tokenizer (embedded here to avoid module path issues in bin)
#[derive(Clone)]
struct ByteLevelBPE {
    vocab_size: usize,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    token_to_id: HashMap<Vec<u8>, u32>,
    id_to_token: HashMap<u32, Vec<u8>>,
    pad_id: u32,
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
}

impl ByteLevelBPE {
    fn new(vocab_size: usize) -> Self {
        let mut tokenizer = Self {
            vocab_size,
            merges: Vec::new(),
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            pad_id: 0,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
        };
        tokenizer.init_base_vocab();
        tokenizer
    }

    fn init_base_vocab(&mut self) {
        self.token_to_id.clear();
        self.id_to_token.clear();

        // Special tokens (IDs 0-3)
        let special_tokens = [
            b"<PAD>".to_vec(),
            b"<UNK>".to_vec(),
            b"<BOS>".to_vec(),
            b"<EOS>".to_vec(),
        ];

        for (id, token) in special_tokens.into_iter().enumerate() {
            self.token_to_id.insert(token.clone(), id as u32);
            self.id_to_token.insert(id as u32, token);
        }

        // Base byte tokens (IDs 4-259)
        for byte in 0u8..=255u8 {
            let token = vec![byte];
            let id = 4 + byte as u32;
            self.token_to_id.insert(token.clone(), id);
            self.id_to_token.insert(id, token);
        }
    }

    fn current_vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn train_from_frequencies(&mut self, word_freqs: HashMap<Vec<u8>, u32>, verbose: bool) {
        // Split words into individual bytes
        let mut word_splits: HashMap<Vec<u8>, (u32, Vec<Vec<u8>>)> = word_freqs
            .into_iter()
            .map(|(word, freq)| {
                let splits: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();
                (word, (freq, splits))
            })
            .collect();

        let num_merges = self.vocab_size.saturating_sub(260);

        if verbose {
            println!("Learning {} merge rules from {} unique words...", num_merges, word_splits.len());
        }

        for merge_idx in 0..num_merges {
            // Count pair frequencies
            let mut pair_freqs: HashMap<(Vec<u8>, Vec<u8>), u32> = HashMap::new();

            for (_word, (freq, splits)) in &word_splits {
                if splits.len() < 2 {
                    continue;
                }
                for window in splits.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|(pair, freq)| (pair.clone(), *freq));

            let Some(((left, right), freq)) = best_pair else {
                if verbose {
                    println!("  No more pairs to merge at iteration {}", merge_idx);
                }
                break;
            };

            // Create merged token
            let mut merged = left.clone();
            merged.extend(&right);

            // Add to vocabulary
            let new_id = self.token_to_id.len() as u32;
            self.token_to_id.insert(merged.clone(), new_id);
            self.id_to_token.insert(new_id, merged.clone());

            // Store merge rule
            self.merges.push((left.clone(), right.clone()));

            // Apply merge to all words
            for (_word, (_freq, splits)) in &mut word_splits {
                let mut i = 0;
                while i + 1 < splits.len() {
                    if splits[i] == left && splits[i + 1] == right {
                        splits[i] = merged.clone();
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            if verbose && (merge_idx + 1) % 500 == 0 {
                println!(
                    "  Merge {}/{}: freq={}, vocab_size={}",
                    merge_idx + 1,
                    num_merges,
                    freq,
                    self.current_vocab_size()
                );
            }
        }

        if verbose {
            println!("Training complete! Final vocabulary size: {}", self.current_vocab_size());
        }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write vocab size
        writeln!(writer, "{}", self.vocab_size)?;

        // Write number of merges
        writeln!(writer, "{}", self.merges.len())?;

        // Write merges as hex-encoded byte pairs
        for (left, right) in &self.merges {
            let left_hex: String = left.iter().map(|b| format!("{:02x}", b)).collect();
            let right_hex: String = right.iter().map(|b| format!("{:02x}", b)).collect();
            writeln!(writer, "{} {}", left_hex, right_hex)?;
        }

        Ok(())
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            let word_tokens = self.encode_word(word.as_bytes());
            result.extend(word_tokens);
            // Add space token between words
            result.push(self.token_to_id.get(&vec![32u8]).copied().unwrap_or(self.unk_id));
        }

        if result.last() == self.token_to_id.get(&vec![32u8]) {
            result.pop();
        }

        result
    }

    fn encode_word(&self, word: &[u8]) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        let mut tokens: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();

        for (left, right) in &self.merges {
            let mut merged = left.clone();
            merged.extend(right);

            let mut i = 0;
            while i + 1 < tokens.len() {
                if &tokens[i] == left && &tokens[i + 1] == right {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
            .into_iter()
            .map(|t| self.token_to_id.get(&t).copied().unwrap_or(self.unk_id))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| id != self.pad_id && id != self.bos_id && id != self.eos_id)
            .filter_map(|&id| self.id_to_token.get(&id))
            .flatten()
            .copied()
            .collect();

        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("{}", "=".repeat(60));
    println!("BYTE-LEVEL BPE TOKENIZER TRAINING");
    println!("{}", "=".repeat(60));
    println!("\nDataset: {}", DATASET_PATH);
    println!("Target vocab size: {}", VOCAB_SIZE);
    println!("Chunk size: {}, Max workers: {}\n", CHUNK_SIZE, MAX_WORKERS);

    // Phase 1: Collect word frequencies from parquet
    println!("Phase 1: Collecting word frequencies...");

    let file = File::open(DATASET_PATH)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // Find text column
    let text_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "text")
        .expect("No 'text' column found");

    let file = File::open(DATASET_PATH)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.with_batch_size(CHUNK_SIZE).build()?;

    let word_freqs: Arc<Mutex<HashMap<Vec<u8>, u32>>> = Arc::new(Mutex::new(HashMap::new()));
    let rows_processed = Arc::new(Mutex::new(0usize));
    let semaphore = Arc::new(Semaphore::new(MAX_WORKERS));
    let mut join_set = JoinSet::new();

    let mut batch_num = 0;
    for batch_result in reader {
        let batch = batch_result?;
        batch_num += 1;

        let num_rows = batch.num_rows();
        {
            let mut processed = rows_processed.lock().unwrap();
            *processed += num_rows;
            if batch_num % 500 == 0 {
                println!("  Queued batch {}, rows: {}", batch_num, *processed);
            }
        }

        // Extract text
        let column = batch.column(text_col_idx);
        let mut texts: Vec<String> = Vec::new();

        if let Some(str_array) = column.as_string_opt::<i32>() {
            for i in 0..str_array.len() {
                if !str_array.is_null(i) {
                    texts.push(str_array.value(i).to_string());
                }
            }
        } else if let Some(str_array) = column.as_string_opt::<i64>() {
            for i in 0..str_array.len() {
                if !str_array.is_null(i) {
                    texts.push(str_array.value(i).to_string());
                }
            }
        }

        // Spawn async task
        let permit = semaphore.clone().acquire_owned().await?;
        let word_freqs = word_freqs.clone();

        join_set.spawn(async move {
            let mut local_freqs: HashMap<Vec<u8>, u32> = HashMap::new();

            for text in texts {
                for word in text.split_whitespace() {
                    let bytes = word.as_bytes().to_vec();
                    *local_freqs.entry(bytes).or_insert(0) += 1;
                }
            }

            // Merge into global
            {
                let mut global = word_freqs.lock().unwrap();
                for (word, freq) in local_freqs {
                    *global.entry(word).or_insert(0) += freq;
                }
            }

            drop(permit);
        });
    }

    // Wait for all tasks
    println!("\nWaiting for workers to finish...");
    while let Some(result) = join_set.join_next().await {
        result?;
    }

    let word_freqs = Arc::try_unwrap(word_freqs).unwrap().into_inner().unwrap();
    let total_processed = *rows_processed.lock().unwrap();

    println!("  Total rows processed: {}", total_processed);
    println!("  Unique words: {}", word_freqs.len());

    // Phase 2: Train BPE
    println!("\nPhase 2: Training BPE tokenizer...");

    let mut tokenizer = ByteLevelBPE::new(VOCAB_SIZE);
    tokenizer.train_from_frequencies(word_freqs, true);

    // Save tokenizer
    println!("\nSaving tokenizer to: {}", TOKENIZER_PATH);
    tokenizer.save(TOKENIZER_PATH)?;

    // Test the tokenizer
    println!("\n{}", "=".repeat(60));
    println!("TOKENIZER TEST");
    println!("{}", "=".repeat(60));

    let test_texts = [
        "پاکستان",
        "اردو زبان",
        "یہ ایک تجربہ ہے",
    ];

    for text in test_texts {
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        println!("\nOriginal: {}", text);
        println!("Encoded:  {:?}", encoded);
        println!("Decoded:  {}", decoded);
        println!("Tokens:   {}", encoded.len());
    }

    println!("\n✓ Tokenizer training complete!");
    println!("  Saved to: {}", TOKENIZER_PATH);
    println!("  Vocabulary size: {}", tokenizer.current_vocab_size());

    Ok(())
}
