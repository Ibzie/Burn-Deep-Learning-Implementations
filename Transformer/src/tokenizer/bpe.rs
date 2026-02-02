use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Configuration for Byte-level BPE tokenizer
#[derive(Clone, Debug)]
pub struct BPEConfig {
    /// Target vocabulary size (including 256 base bytes + special tokens)
    pub vocab_size: usize,
    /// Special tokens: PAD, UNK, BOS, EOS
    pub pad_token: &'static str,
    pub unk_token: &'static str,
    pub bos_token: &'static str,
    pub eos_token: &'static str,
}

impl Default for BPEConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            pad_token: "<PAD>",
            unk_token: "<UNK>",
            bos_token: "<BOS>",
            eos_token: "<EOS>",
        }
    }
}

/// Byte-level BPE Tokenizer
///
/// Starts with 256 byte tokens and learns merge rules to build
/// a vocabulary of common subword units.
#[derive(Clone)]
pub struct ByteLevelBPE {
    config: BPEConfig,
    /// Merge rules: (byte_pair) -> merged_token_id
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Token to ID mapping
    token_to_id: HashMap<Vec<u8>, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, Vec<u8>>,
    /// Special token IDs
    pub pad_id: u32,
    pub unk_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl ByteLevelBPE {
    /// Create a new untrained BPE tokenizer
    pub fn new(config: BPEConfig) -> Self {
        let mut tokenizer = Self {
            config,
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

    /// Initialize vocabulary with special tokens + 256 byte tokens
    fn init_base_vocab(&mut self) {
        self.token_to_id.clear();
        self.id_to_token.clear();

        // Special tokens (IDs 0-3)
        let special_tokens = [
            self.config.pad_token.as_bytes().to_vec(),
            self.config.unk_token.as_bytes().to_vec(),
            self.config.bos_token.as_bytes().to_vec(),
            self.config.eos_token.as_bytes().to_vec(),
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

    /// Get current vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Train BPE on a corpus of texts
    ///
    /// # Arguments
    /// * `texts` - Iterator of text strings to train on
    /// * `verbose` - Print progress during training
    pub fn train<'a, I>(&mut self, texts: I, verbose: bool)
    where
        I: Iterator<Item = &'a str>,
    {
        // Step 1: Convert all texts to byte sequences and count pair frequencies
        if verbose {
            println!("Step 1: Converting texts to bytes and counting initial pairs...");
        }

        let mut word_freqs: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut total_chars = 0usize;

        for text in texts {
            // Split on whitespace to get "words" and convert each to bytes
            for word in text.split_whitespace() {
                let bytes: Vec<u8> = word.as_bytes().to_vec();
                total_chars += bytes.len();
                *word_freqs.entry(bytes).or_insert(0) += 1;
            }
        }

        if verbose {
            println!("  Unique words: {}", word_freqs.len());
            println!("  Total characters: {}", total_chars);
        }

        // Step 2: Split words into individual bytes (our initial tokens)
        // word_splits[word_bytes] = (frequency, current_token_sequence)
        let mut word_splits: HashMap<Vec<u8>, (u32, Vec<Vec<u8>>)> = word_freqs
            .into_iter()
            .map(|(word, freq)| {
                let splits: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();
                (word, (freq, splits))
            })
            .collect();

        // Step 3: Iteratively find and merge most frequent pairs
        let num_merges = self.config.vocab_size.saturating_sub(260); // 256 bytes + 4 special

        if verbose {
            println!("Step 2: Learning {} merge rules...", num_merges);
        }

        for merge_idx in 0..num_merges {
            // Count pair frequencies across all words
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
                .map(|(pair, _)| pair.clone());

            let Some((left, right)) = best_pair else {
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
                let freq = pair_freqs.get(&(left.clone(), right.clone())).unwrap_or(&0);
                println!(
                    "  Merge {}/{}: {:?} + {:?} -> {:?} (freq: {})",
                    merge_idx + 1,
                    num_merges,
                    String::from_utf8_lossy(&left),
                    String::from_utf8_lossy(&right),
                    String::from_utf8_lossy(&merged),
                    freq
                );
            }
        }

        if verbose {
            println!("Training complete! Vocabulary size: {}", self.vocab_size());
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();

        // Process each whitespace-separated word
        for word in text.split_whitespace() {
            let word_tokens = self.encode_word(word.as_bytes());
            result.extend(word_tokens);
            // Add space token between words (space = byte 32)
            result.push(self.token_to_id.get(&vec![32u8]).copied().unwrap_or(self.unk_id));
        }

        // Remove trailing space if present
        if result.last() == self.token_to_id.get(&vec![32u8]) {
            result.pop();
        }

        result
    }

    /// Encode with BOS and EOS tokens
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut result = vec![self.bos_id];
        result.extend(self.encode(text));
        result.push(self.eos_id);
        result
    }

    /// Encode a single word (byte sequence)
    fn encode_word(&self, word: &[u8]) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        // Start with individual bytes
        let mut tokens: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();

        // Apply merges in order
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

        // Convert tokens to IDs
        tokens
            .into_iter()
            .map(|t| self.token_to_id.get(&t).copied().unwrap_or(self.unk_id))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| id != self.pad_id && id != self.bos_id && id != self.eos_id)
            .filter_map(|&id| self.id_to_token.get(&id))
            .flatten()
            .copied()
            .collect();

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write vocab size
        writeln!(writer, "{}", self.config.vocab_size)?;

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

    /// Load tokenizer from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content)?;

        let mut lines = content.lines();

        // Read vocab size
        let vocab_size: usize = lines
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing vocab size"))?
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Read number of merges
        let num_merges: usize = lines
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing merge count"))?
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let config = BPEConfig {
            vocab_size,
            ..Default::default()
        };

        let mut tokenizer = Self::new(config);

        // Read merges
        for _ in 0..num_merges {
            let line = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing merge"))?;

            let mut parts = line.split_whitespace();
            let left_hex = parts.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing left token")
            })?;
            let right_hex = parts.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing right token")
            })?;

            let left = hex_to_bytes(left_hex)?;
            let right = hex_to_bytes(right_hex)?;

            // Create merged token
            let mut merged = left.clone();
            merged.extend(&right);

            // Add to vocabulary
            let new_id = tokenizer.token_to_id.len() as u32;
            tokenizer.token_to_id.insert(merged.clone(), new_id);
            tokenizer.id_to_token.insert(new_id, merged);

            tokenizer.merges.push((left, right));
        }

        Ok(tokenizer)
    }

    /// Get token string for an ID (for debugging)
    pub fn id_to_string(&self, id: u32) -> String {
        self.id_to_token
            .get(&id)
            .map(|bytes| String::from_utf8_lossy(bytes).to_string())
            .unwrap_or_else(|| "<INVALID>".to_string())
    }
}

fn hex_to_bytes(hex: &str) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut chars = hex.chars();

    while let (Some(h), Some(l)) = (chars.next(), chars.next()) {
        let high = h.to_digit(16).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid hex")
        })? as u8;
        let low = l.to_digit(16).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid hex")
        })? as u8;
        bytes.push((high << 4) | low);
    }

    Ok(bytes)
}

/// Async trainer for large datasets
pub struct AsyncBPETrainer {
    config: BPEConfig,
    max_workers: usize,
}

impl AsyncBPETrainer {
    pub fn new(config: BPEConfig, max_workers: usize) -> Self {
        Self { config, max_workers }
    }

    /// Train on texts loaded in chunks asynchronously
    pub async fn train_on_chunks<F, Fut>(
        &self,
        chunk_loader: F,
        num_chunks: usize,
        verbose: bool,
    ) -> ByteLevelBPE
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<String>> + Send,
    {
        // Phase 1: Collect word frequencies from all chunks in parallel
        if verbose {
            println!("Phase 1: Collecting word frequencies from {} chunks...", num_chunks);
        }

        let word_freqs: Arc<Mutex<HashMap<Vec<u8>, u32>>> = Arc::new(Mutex::new(HashMap::new()));
        let semaphore = Arc::new(Semaphore::new(self.max_workers));
        let chunk_loader = Arc::new(chunk_loader);
        let mut join_set = JoinSet::new();

        for chunk_idx in 0..num_chunks {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let word_freqs = word_freqs.clone();
            let loader = chunk_loader.clone();

            join_set.spawn(async move {
                let texts = loader(chunk_idx).await;
                let mut local_freqs: HashMap<Vec<u8>, u32> = HashMap::new();

                for text in texts {
                    for word in text.split_whitespace() {
                        let bytes = word.as_bytes().to_vec();
                        *local_freqs.entry(bytes).or_insert(0) += 1;
                    }
                }

                // Merge into global frequencies
                {
                    let mut global = word_freqs.lock().unwrap();
                    for (word, freq) in local_freqs {
                        *global.entry(word).or_insert(0) += freq;
                    }
                }

                drop(permit);
            });
        }

        // Wait for all chunks to be processed
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                eprintln!("Chunk processing error: {:?}", e);
            }
        }

        let word_freqs = Arc::try_unwrap(word_freqs)
            .unwrap()
            .into_inner()
            .unwrap();

        if verbose {
            println!("  Unique words collected: {}", word_freqs.len());
        }

        // Phase 2: Train BPE on collected frequencies
        if verbose {
            println!("Phase 2: Learning merge rules...");
        }

        let mut tokenizer = ByteLevelBPE::new(self.config.clone());
        tokenizer.train_from_frequencies(word_freqs, verbose);

        tokenizer
    }
}

impl ByteLevelBPE {
    /// Train from pre-computed word frequencies
    pub fn train_from_frequencies(&mut self, word_freqs: HashMap<Vec<u8>, u32>, verbose: bool) {
        // Split words into individual bytes
        let mut word_splits: HashMap<Vec<u8>, (u32, Vec<Vec<u8>>)> = word_freqs
            .into_iter()
            .map(|(word, freq)| {
                let splits: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();
                (word, (freq, splits))
            })
            .collect();

        let num_merges = self.config.vocab_size.saturating_sub(260);

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
                .map(|(pair, _)| pair.clone());

            let Some((left, right)) = best_pair else {
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
                let freq = pair_freqs.get(&(left.clone(), right.clone())).unwrap_or(&0);
                println!(
                    "  Merge {}/{}: freq={}, vocab_size={}",
                    merge_idx + 1,
                    num_merges,
                    freq,
                    self.vocab_size()
                );
            }
        }

        if verbose {
            println!("Training complete! Final vocabulary size: {}", self.vocab_size());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode_decode() {
        let config = BPEConfig {
            vocab_size: 300, // Small vocab for testing
            ..Default::default()
        };
        let mut tokenizer = ByteLevelBPE::new(config);

        // Train on simple text
        let texts = vec!["hello world", "hello there", "world hello"];
        tokenizer.train(texts.iter().map(|s| *s), false);

        // Test encoding
        let encoded = tokenizer.encode("hello world");
        assert!(!encoded.is_empty());

        // Test decoding
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded.trim(), "hello world");
    }

    #[test]
    fn test_urdu_text() {
        let config = BPEConfig {
            vocab_size: 500,
            ..Default::default()
        };
        let mut tokenizer = ByteLevelBPE::new(config);

        let texts = vec!["پاکستان", "اردو زبان", "پاکستان میں اردو"];
        tokenizer.train(texts.iter().map(|s| *s), false);

        let encoded = tokenizer.encode("پاکستان");
        let decoded = tokenizer.decode(&encoded);

        // Should roundtrip correctly
        assert_eq!(decoded.trim(), "پاکستان");
    }
}
