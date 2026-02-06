use burn::prelude::*;
use burn::data::dataloader::batcher::Batcher;

use super::dataset::TextSample;

/// A batch of text data ready for the transformer.
///
/// Contains input token IDs and shifted target IDs for next-token prediction.
#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    /// Input token IDs: [batch_size, seq_len]
    pub inputs: Tensor<B, 2, Int>,
    /// Target token IDs (shifted by 1): [batch_size, seq_len]
    pub targets: Tensor<B, 2, Int>,
}

/// Batcher that converts raw TextSamples into padded, shifted TextBatches.
///
/// For each sample:
/// 1. Truncate to max_seq_len + 1 tokens (need the +1 for the target shift)
/// 2. Pad shorter sequences with pad_id (0)
/// 3. Split into input = tokens[0..n] and target = tokens[1..n+1]
#[derive(Clone)]
pub struct TextBatcher<B: Backend> {
    max_seq_len: usize,
    pad_id: i32,
    device: B::Device,
}

impl<B: Backend> TextBatcher<B> {
    pub fn new(max_seq_len: usize, device: B::Device) -> Self {
        Self {
            max_seq_len,
            pad_id: 0, // <PAD> token ID
            device,
        }
    }
}

impl<B: Backend> Batcher<TextSample, TextBatch<B>> for TextBatcher<B> {
    fn batch(&self, items: Vec<TextSample>) -> TextBatch<B> {
        let mut input_batch: Vec<Tensor<B, 1, Int>> = Vec::with_capacity(items.len());
        let mut target_batch: Vec<Tensor<B, 1, Int>> = Vec::with_capacity(items.len());

        for sample in &items {
            // Truncate to max_seq_len + 1 so we have room for the target shift
            let max_len = self.max_seq_len + 1;
            let tokens: Vec<i32> = if sample.token_ids.len() > max_len {
                sample.token_ids[..max_len].to_vec()
            } else {
                // Pad to max_len with pad_id
                let mut padded = sample.token_ids.clone();
                padded.resize(max_len, self.pad_id);
                padded
            };

            // Split: input = tokens[0..max_seq_len], target = tokens[1..max_seq_len+1]
            // This creates the next-token prediction pairs:
            //   input:  [BOS, tok1, tok2, ..., tokN-1]
            //   target: [tok1, tok2, ..., tokN-1, EOS/PAD]
            let input: Vec<i32> = tokens[..self.max_seq_len].to_vec();
            let target: Vec<i32> = tokens[1..self.max_seq_len + 1].to_vec();

            input_batch.push(Tensor::from_ints(
                input.as_slice(),
                &self.device,
            ));
            target_batch.push(Tensor::from_ints(
                target.as_slice(),
                &self.device,
            ));
        }

        // Stack individual 1D tensors into 2D batch tensors
        // [batch_size, seq_len]
        let inputs = Tensor::stack(input_batch, 0);
        let targets = Tensor::stack(target_batch, 0);

        TextBatch { inputs, targets }
    }
}
