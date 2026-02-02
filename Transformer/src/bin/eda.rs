use arrow::array::{Array, AsArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write as IoWrite;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use unicode_segmentation::UnicodeSegmentation;

const DATASET_PATH: &str =
    "/mnt/stuff-second/Datasets/HuggingFace/20231101.ur/train-00000-of-00001.parquet";
const OUTPUT_PATH: &str = "/tmp/urdu_dataset_eda.txt";
const CHUNK_SIZE: usize = 10;
const MAX_WORKERS: usize = 10;

#[derive(Default, Clone)]
struct TextStats {
    char_lengths: Vec<usize>,
    word_counts: Vec<usize>,
    grapheme_counts: Vec<usize>,
    char_freq: HashMap<char, usize>,
    sample_texts: Vec<String>,
}

impl TextStats {
    fn merge(&mut self, other: TextStats) {
        self.char_lengths.extend(other.char_lengths);
        self.word_counts.extend(other.word_counts);
        self.grapheme_counts.extend(other.grapheme_counts);
        for (ch, count) in other.char_freq {
            *self.char_freq.entry(ch).or_insert(0) += count;
        }
        if self.sample_texts.len() < 5 {
            for text in other.sample_texts {
                if self.sample_texts.len() < 5 {
                    self.sample_texts.push(text);
                }
            }
        }
    }
}

fn process_chunk(texts: Vec<String>, collect_char_freq: bool) -> TextStats {
    let mut stats = TextStats::default();

    for text in texts {
        stats.char_lengths.push(text.chars().count());
        stats.word_counts.push(text.split_whitespace().count());
        stats.grapheme_counts.push(text.graphemes(true).count());

        if collect_char_freq {
            for c in text.chars() {
                *stats.char_freq.entry(c).or_insert(0) += 1;
            }
        }

        if stats.sample_texts.len() < 3 {
            let preview = if text.len() > 500 {
                format!("{}...", text.chars().take(500).collect::<String>())
            } else {
                text.clone()
            };
            stats.sample_texts.push(preview);
        }
    }

    stats
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut output = String::new();

    output.push_str(&"=".repeat(80));
    output.push('\n');
    output.push_str("URDU DATASET EXPLORATORY DATA ANALYSIS (Async Chunked)\n");
    output.push_str(&"=".repeat(80));
    output.push_str("\n\n");
    output.push_str(&format!("Dataset: {}\n", DATASET_PATH));
    output.push_str(&format!(
        "Processing: {} lines per chunk, {} max workers\n\n",
        CHUNK_SIZE, MAX_WORKERS
    ));

    println!("Opening parquet file...");
    let file = File::open(DATASET_PATH)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    // Get metadata
    let metadata = builder.metadata();
    let total_rows = metadata.file_metadata().num_rows() as usize;
    let schema = builder.schema().clone();

    output.push_str(&"-".repeat(40));
    output.push('\n');
    output.push_str("1. BASIC DATASET INFO\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');
    output.push_str(&format!("Total rows: {}\n", total_rows));
    output.push_str(&format!("Number of columns: {}\n", schema.fields().len()));
    output.push_str("\nColumn names and types:\n");

    let mut text_columns: Vec<String> = Vec::new();
    for field in schema.fields() {
        output.push_str(&format!("  - {}: {:?}\n", field.name(), field.data_type()));
        if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
            text_columns.push(field.name().clone());
        }
    }

    // Re-open file for reading batches
    let file = File::open(DATASET_PATH)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.with_batch_size(CHUNK_SIZE).build()?;

    // Shared state for async processing
    let global_stats: Arc<Mutex<HashMap<String, TextStats>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let rows_processed = Arc::new(Mutex::new(0usize));
    let null_counts: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));

    // Initialize stats for each text column
    {
        let mut stats = global_stats.lock().unwrap();
        let mut nulls = null_counts.lock().unwrap();
        for col in &text_columns {
            stats.insert(col.clone(), TextStats::default());
            nulls.insert(col.clone(), 0);
        }
    }

    // Semaphore to limit concurrent workers
    let semaphore = Arc::new(Semaphore::new(MAX_WORKERS));
    let mut join_set = JoinSet::new();

    println!("Processing chunks asynchronously...");

    let mut batch_num = 0;
    for batch_result in reader {
        let batch = batch_result?;
        batch_num += 1;

        let num_rows = batch.num_rows();
        {
            let mut processed = rows_processed.lock().unwrap();
            *processed += num_rows;
            if batch_num % 1000 == 0 {
                println!("  Queued batch {}, rows processed so far: {}", batch_num, *processed);
            }
        }

        // Extract text data for each text column
        for col_name in &text_columns {
            let col_idx = schema
                .fields()
                .iter()
                .position(|f| f.name() == col_name)
                .unwrap();
            let column = batch.column(col_idx);

            let mut texts: Vec<String> = Vec::new();
            let mut local_nulls = 0usize;

            // Handle both Utf8 and LargeUtf8
            if let Some(str_array) = column.as_string_opt::<i32>() {
                for i in 0..str_array.len() {
                    if str_array.is_null(i) {
                        local_nulls += 1;
                    } else {
                        texts.push(str_array.value(i).to_string());
                    }
                }
            } else if let Some(str_array) = column.as_string_opt::<i64>() {
                for i in 0..str_array.len() {
                    if str_array.is_null(i) {
                        local_nulls += 1;
                    } else {
                        texts.push(str_array.value(i).to_string());
                    }
                }
            }

            // Update null counts
            {
                let mut nulls = null_counts.lock().unwrap();
                *nulls.get_mut(col_name).unwrap() += local_nulls;
            }

            // Spawn async task for processing
            let permit = semaphore.clone().acquire_owned().await?;
            let stats_ref = global_stats.clone();
            let col = col_name.clone();
            let collect_freq = batch_num <= 100; // Only collect char freq for first 100 batches

            join_set.spawn(async move {
                let chunk_stats = process_chunk(texts, collect_freq);

                {
                    let mut stats = stats_ref.lock().unwrap();
                    if let Some(existing) = stats.get_mut(&col) {
                        existing.merge(chunk_stats);
                    }
                }

                drop(permit); // Release semaphore
            });
        }
    }

    // Wait for all tasks to complete
    println!("Waiting for all workers to finish...");
    while let Some(result) = join_set.join_next().await {
        result?;
    }

    let final_processed = *rows_processed.lock().unwrap();
    println!("Total rows processed: {}", final_processed);

    // Generate report
    output.push_str(&format!("\nTotal rows processed: {}\n", final_processed));

    // Null analysis
    output.push_str("\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');
    output.push_str("2. NULL VALUE ANALYSIS\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');

    {
        let nulls = null_counts.lock().unwrap();
        for (col, count) in nulls.iter() {
            let pct = (*count as f64 / final_processed as f64) * 100.0;
            output.push_str(&format!("  {}: {} nulls ({:.2}%)\n", col, count, pct));
        }
    }

    // Text statistics
    output.push_str("\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');
    output.push_str("3. TEXT COLUMN ANALYSIS\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');

    {
        let stats = global_stats.lock().unwrap();
        for (col_name, col_stats) in stats.iter() {
            output.push_str(&format!("\nColumn: '{}'\n", col_name));

            if !col_stats.char_lengths.is_empty() {
                let mut char_lengths = col_stats.char_lengths.clone();
                let mut word_counts = col_stats.word_counts.clone();

                // Character stats
                let avg_chars: f64 =
                    char_lengths.iter().sum::<usize>() as f64 / char_lengths.len() as f64;
                let min_chars = *char_lengths.iter().min().unwrap();
                let max_chars = *char_lengths.iter().max().unwrap();
                char_lengths.sort();
                let median_chars = char_lengths[char_lengths.len() / 2];

                output.push_str(&format!(
                    "  Character count (n={}):\n",
                    char_lengths.len()
                ));
                output.push_str(&format!("    Min: {}\n", min_chars));
                output.push_str(&format!("    Max: {}\n", max_chars));
                output.push_str(&format!("    Mean: {:.2}\n", avg_chars));
                output.push_str(&format!("    Median: {}\n", median_chars));

                // Word stats
                let avg_words: f64 =
                    word_counts.iter().sum::<usize>() as f64 / word_counts.len() as f64;
                let min_words = *word_counts.iter().min().unwrap();
                let max_words = *word_counts.iter().max().unwrap();
                word_counts.sort();
                let median_words = word_counts[word_counts.len() / 2];

                output.push_str("\n  Word count (whitespace-separated):\n");
                output.push_str(&format!("    Min: {}\n", min_words));
                output.push_str(&format!("    Max: {}\n", max_words));
                output.push_str(&format!("    Mean: {:.2}\n", avg_words));
                output.push_str(&format!("    Median: {}\n", median_words));

                // Grapheme stats
                let avg_graphemes: f64 = col_stats.grapheme_counts.iter().sum::<usize>() as f64
                    / col_stats.grapheme_counts.len() as f64;
                output.push_str(&format!("\n  Grapheme clusters (avg): {:.2}\n", avg_graphemes));

                // Top characters
                let mut char_vec: Vec<_> = col_stats.char_freq.iter().collect();
                char_vec.sort_by(|a, b| b.1.cmp(a.1));

                output.push_str("\n  Top 20 most frequent characters:\n");
                for (i, (ch, count)) in char_vec.iter().take(20).enumerate() {
                    let display = if ch.is_whitespace() {
                        format!("[space:{:?}]", ch)
                    } else {
                        ch.to_string()
                    };
                    output.push_str(&format!("    {}. '{}' : {}\n", i + 1, display, count));
                }

                // Length distribution
                output.push_str("\n  Length distribution (characters):\n");
                let buckets = [0usize, 100, 500, 1000, 5000, 10000, 50000, usize::MAX];
                for i in 0..buckets.len() - 1 {
                    let count = char_lengths
                        .iter()
                        .filter(|&&len| len >= buckets[i] && len < buckets[i + 1])
                        .count();
                    let pct = (count as f64 / char_lengths.len() as f64) * 100.0;
                    let label = if buckets[i + 1] == usize::MAX {
                        format!("{}+", buckets[i])
                    } else {
                        format!("{}-{}", buckets[i], buckets[i + 1])
                    };
                    output.push_str(&format!("    {}: {} ({:.1}%)\n", label, count, pct));
                }
            }
        }
    }

    // Sample texts
    output.push_str("\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');
    output.push_str("4. SAMPLE TEXTS\n");
    output.push_str(&"-".repeat(40));
    output.push('\n');

    {
        let stats = global_stats.lock().unwrap();
        for (col_name, col_stats) in stats.iter() {
            output.push_str(&format!("\nSample texts from '{}':\n", col_name));
            for (i, text) in col_stats.sample_texts.iter().take(3).enumerate() {
                output.push_str(&format!("\n--- Sample {} ---\n{}\n", i + 1, text));
            }
        }
    }

    output.push_str("\n");
    output.push_str(&"=".repeat(80));
    output.push('\n');
    output.push_str("END OF EDA REPORT\n");
    output.push_str(&"=".repeat(80));
    output.push('\n');

    // Write to file
    let mut file = File::create(OUTPUT_PATH)?;
    file.write_all(output.as_bytes())?;

    println!("\nEDA complete! Results written to: {}", OUTPUT_PATH);
    println!("\n{}", output);

    Ok(())
}
