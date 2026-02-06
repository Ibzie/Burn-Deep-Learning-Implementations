use arrow::array::{Array, AsArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read as _, Write as _};
use std::path::Path;

use burn::data::dataset::Dataset;

use crate::tokenizer::ByteLevelBPE;

/// A single tokenized text sample
#[derive(Clone, Debug)]
pub struct TextSample {
    /// Token IDs for this text (includes BOS/EOS)
    pub token_ids: Vec<i32>,
}

/// Dataset of tokenized Urdu texts loaded from a parquet file.
///
/// On first load, reads parquet and tokenizes every row, then saves a binary
/// cache file next to the parquet. On subsequent runs, loads directly from
/// the cache (~instant).
pub struct UrduTextDataset {
    samples: Vec<TextSample>,
}

impl UrduTextDataset {
    /// Load the dataset, using a cache file if available.
    ///
    /// Cache file is stored at `{parquet_path}.tokens.cache`.
    /// Delete the cache file to force re-tokenization.
    pub fn new(parquet_path: &str, tokenizer_path: &str) -> Self {
        let cache_path = format!("{}.tokens.cache", parquet_path);

        // Try loading from cache first
        if Path::new(&cache_path).exists() {
            println!("  Found token cache: {}", cache_path);
            match Self::load_cache(&cache_path) {
                Ok(dataset) => {
                    println!("  Loaded {} samples from cache", dataset.samples.len());
                    return dataset;
                }
                Err(e) => {
                    println!("  Cache load failed ({}), re-tokenizing...", e);
                }
            }
        }

        // No cache — tokenize from parquet
        let dataset = Self::tokenize_from_parquet(parquet_path, tokenizer_path);

        // Save cache for next time
        if let Err(e) = dataset.save_cache(&cache_path) {
            println!("  Warning: failed to save cache: {}", e);
        } else {
            println!("  Saved token cache to {}", cache_path);
        }

        dataset
    }

    /// Tokenize all rows from the parquet file
    fn tokenize_from_parquet(parquet_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = ByteLevelBPE::load(tokenizer_path)
            .expect("Failed to load tokenizer");

        let file = File::open(parquet_path).expect("Failed to open parquet file");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to read parquet");
        let schema = builder.schema().clone();

        let text_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "text")
            .expect("No 'text' column found in parquet");

        let file = File::open(parquet_path).expect("Failed to open parquet file");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to read parquet");
        let reader = builder.with_batch_size(1024).build()
            .expect("Failed to build parquet reader");

        let mut samples = Vec::new();
        let mut rows_processed: usize = 0;
        let start = std::time::Instant::now();
        println!("  Tokenizing rows...");

        for batch_result in reader {
            let batch = batch_result.expect("Failed to read batch");
            let column = batch.column(text_col_idx);

            if let Some(str_array) = column.as_string_opt::<i32>() {
                for i in 0..str_array.len() {
                    if !str_array.is_null(i) {
                        let text = str_array.value(i);
                        let token_ids = tokenizer
                            .encode_with_special(text)
                            .into_iter()
                            .map(|id| id as i32)
                            .collect();
                        samples.push(TextSample { token_ids });
                    }
                }
            } else if let Some(str_array) = column.as_string_opt::<i64>() {
                for i in 0..str_array.len() {
                    if !str_array.is_null(i) {
                        let text = str_array.value(i);
                        let token_ids = tokenizer
                            .encode_with_special(text)
                            .into_iter()
                            .map(|id| id as i32)
                            .collect();
                        samples.push(TextSample { token_ids });
                    }
                }
            }

            rows_processed += batch.num_rows();
            if rows_processed % 50 == 0 {
                let elapsed = start.elapsed().as_secs_f32();
                let rate = rows_processed as f32 / elapsed;
                println!("  Tokenized {rows_processed} rows ({rate:.0} rows/sec)");
            }
        }

        let elapsed = start.elapsed().as_secs_f32();
        println!("Loaded {} samples from {} in {:.1}s", samples.len(), parquet_path, elapsed);
        Self { samples }
    }

    /// Save tokenized samples to a binary cache file.
    ///
    /// Format: [num_samples: u64] then for each sample:
    ///   [num_tokens: u32] [token_ids: i32 * num_tokens]
    fn save_cache(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Number of samples
        w.write_all(&(self.samples.len() as u64).to_le_bytes())?;

        for sample in &self.samples {
            // Number of tokens in this sample
            w.write_all(&(sample.token_ids.len() as u32).to_le_bytes())?;
            // Token IDs
            for &id in &sample.token_ids {
                w.write_all(&id.to_le_bytes())?;
            }
        }

        w.flush()
    }

    /// Load tokenized samples from a binary cache file.
    fn load_cache(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        // Read number of samples
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let num_samples = u64::from_le_bytes(buf8) as usize;

        let mut samples = Vec::with_capacity(num_samples);
        let mut buf4 = [0u8; 4];

        for _ in 0..num_samples {
            // Read number of tokens
            r.read_exact(&mut buf4)?;
            let num_tokens = u32::from_le_bytes(buf4) as usize;

            // Read token IDs
            let mut token_ids = Vec::with_capacity(num_tokens);
            for _ in 0..num_tokens {
                r.read_exact(&mut buf4)?;
                token_ids.push(i32::from_le_bytes(buf4));
            }

            samples.push(TextSample { token_ids });
        }

        Ok(Self { samples })
    }

    /// Split the dataset into train and validation sets.
    ///
    /// # Arguments
    /// * `train_ratio` - Fraction of data for training (e.g. 0.9)
    /// * `seed` - Random seed for reproducible shuffling
    pub fn split(self, train_ratio: f64, seed: u64) -> (Self, Self) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut indices: Vec<usize> = (0..self.samples.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let split_point = (self.samples.len() as f64 * train_ratio) as usize;

        let train_samples: Vec<TextSample> = indices[..split_point]
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();
        let valid_samples: Vec<TextSample> = indices[split_point..]
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();

        println!(
            "Split: {} train, {} valid",
            train_samples.len(),
            valid_samples.len()
        );

        (
            Self { samples: train_samples },
            Self { samples: valid_samples },
        )
    }
}

/// Implement Burn's Dataset trait for random-access data loading
impl Dataset<TextSample> for UrduTextDataset {
    fn get(&self, index: usize) -> Option<TextSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
