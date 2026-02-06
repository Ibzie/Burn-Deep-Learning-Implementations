use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use burn_transformer::model::{Transformer, TransformerConfig};

type Backend = Autodiff<NdArray>;

fn main() {
    // Create a tiny model for testing
    let device = Default::default();
    let config = TransformerConfig::tiny();
    
    println!("Creating transformer with config: {:#?}", config);
    let model: Transformer<Backend> = Transformer::new(&config, &device);
    
    // Create dummy input: batch of 2 sequences, each 10 tokens long
    let batch_size = 2;
    let seq_len = 10;
    let input_ids = Tensor::from_ints([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                       [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], &device);
    
    println!("\nInput shape: {:?}", input_ids.dims());
    
    // Forward pass
    println!("\nRunning forward pass...");
    let logits = model.forward(input_ids);
    
    println!("Output logits shape: {:?}", logits.dims());
    println!("\n✓ Transformer compiled and working!");
}