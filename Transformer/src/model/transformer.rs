use burn::prelude::*;
use burn::nn;

use super::block::{TransformerBlock, TransformerBlockConfig};

/// Decoder-only Transformer for text generation (GPT-style)
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    token_embedding: nn::Embedding<B>,      // Token embeddings: vocab_size -> embed_dim
    position_embedding: nn::Embedding<B>,   // Positional embeddings: max_seq_len -> embed_dim
    blocks: Vec<TransformerBlock<B>>,       // Stack of N transformer blocks
    ln_f: nn::LayerNorm<B>,                 // Final layer norm
    lm_head: nn::Linear<B>,                 // Output projection: embed_dim -> vocab_size
    dropout: nn::Dropout,                   // Dropout for embeddings
}

impl<B: Backend> Transformer<B> {
    pub fn new(config: &TransformerConfig, device: &B::Device) -> Self {
        let vocab_size = config.vocab_size;
        let embed_dim = config.embed_dim;
        let max_seq_len = config.max_seq_len;
        let num_layers = config.num_layers;
        
        // Create embedding layers
        // Token embedding: maps token IDs to dense vectors
        let token_embedding = nn::EmbeddingConfig::new(vocab_size, embed_dim)
            .init(device);
        
        // Position embedding: learned positional encodings
        let position_embedding = nn::EmbeddingConfig::new(max_seq_len, embed_dim)
            .init(device);
        
        // Create transformer block config
        let block_config = TransformerBlockConfig {
            embed_dim,
            num_heads: config.num_heads,
            ff_dim: config.ff_dim,
            dropout: config.dropout,
        };
        
        // Stack N transformer blocks
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(&block_config, device));
        }
        
        // Final layer norm
        let ln_f = nn::LayerNormConfig::new(embed_dim).init(device);
        
        // Language modeling head: projects back to vocabulary
        // Note: We could tie these weights with token_embedding for efficiency
        let lm_head = nn::LinearConfig::new(embed_dim, vocab_size)
            .with_bias(false)  // Often no bias in output projection
            .init(device);
        
        let dropout = nn::DropoutConfig::new(config.dropout).init();
        
        Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
            dropout,
        }
    }
    
    /// Forward pass through the transformer
    /// 
    /// # Arguments
    /// * `input_ids` - Token IDs [batch_size, seq_len]
    /// 
    /// # Returns
    /// * Logits over vocabulary [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = &input_ids.device();
        
        // Step 1: Get token embeddings
        // [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        let token_embeds = self.token_embedding.forward(input_ids);
        
        // Step 2: Create position IDs [0, 1, 2, ..., seq_len-1]
        // This is the same for all sequences in the batch
        let position_ids = Tensor::arange(0..seq_len as i64, device)
            .reshape([1, seq_len])  // [1, seq_len]
            .repeat_dim(0, batch_size);  // [batch_size, seq_len]
        
        // Step 3: Get positional embeddings
        // [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        let position_embeds = self.position_embedding.forward(position_ids);
        
        // Step 4: Add token and position embeddings
        // This gives each token both semantic meaning (from token) and 
        // positional information (from position)
        let mut x = token_embeds + position_embeds;
        
        // Step 5: Apply dropout to embeddings
        x = self.dropout.forward(x);
        
        // Step 6: Create causal mask to prevent attending to future tokens
        // mask[i, j] = true if j > i (token i cannot attend to token j)
        let causal_mask = self.create_causal_mask(seq_len, device);
        
        // Step 7: Pass through all transformer blocks
        for block in &self.blocks {
            x = block.forward(x, Some(causal_mask.clone()));
        }
        
        // Step 8: Final layer normalization
        x = self.ln_f.forward(x);
        
        // Step 9: Project to vocabulary logits
        // [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, vocab_size]
        let logits = self.lm_head.forward(x);
        
        logits
    }
    
    /// Create a causal mask for autoregressive generation
    /// Returns a boolean mask where True = mask out (cannot attend)
    /// 
    /// Example for seq_len=4:
    /// [[False, True,  True,  True ],   # Token 0 can only see itself
    ///  [False, False, True,  True ],   # Token 1 can see 0,1
    ///  [False, False, False, True ],   # Token 2 can see 0,1,2
    ///  [False, False, False, False]]   # Token 3 can see all
    fn create_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
        // Create a matrix of positions
        let row = Tensor::arange(0..seq_len as i64, device)
            .reshape([seq_len, 1])
            .float();  // [seq_len, 1]
        
        let col = Tensor::arange(0..seq_len as i64, device)
            .reshape([1, seq_len])
            .float();  // [1, seq_len]
        
        // mask[i,j] = (j > i), which means "can token i attend to token j?"
        // We want False where we CAN attend, True where we CANNOT
        col.greater(row)  // [seq_len, seq_len]
    }
    
    /// Generate text autoregressively (for inference)
    /// 
    /// # Arguments
    /// * `start_tokens` - Initial token IDs [batch_size, start_len]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = neutral, <1.0 = more conservative)
    /// 
    /// # Returns
    /// * Generated token IDs [batch_size, start_len + max_new_tokens]
    pub fn generate(
        &self,
        start_tokens: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Tensor<B, 2, Int> {
        let mut tokens = start_tokens;
        
        // Generate tokens one at a time
        for _ in 0..max_new_tokens {
            // Get logits for all positions
            // [batch_size, current_seq_len, vocab_size]
            let logits = self.forward(tokens.clone());
            
            // Take logits at the last position only
            // [batch_size, vocab_size]
            let [batch_size, seq_len, vocab_size] = logits.dims();
            let last_logits = logits.slice([0..batch_size, seq_len-1..seq_len, 0..vocab_size])
                .squeeze(1);  // Remove seq_len dimension
            
            // Apply temperature scaling
            let scaled_logits = last_logits / temperature;
            
            // Sample from the distribution (using argmax for now - we can improve this)
            // In practice, you'd want to use proper sampling with softmax
            let next_token = scaled_logits.argmax(1);  // [batch_size, 1]
            
            // Append to sequence
            tokens = Tensor::cat(vec![tokens, next_token], 1);
        }
        
        tokens
    }
}

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,      // Size of vocabulary (e.g., 50257 for GPT-2)
    pub embed_dim: usize,        // Embedding dimension (e.g., 512)
    pub num_heads: usize,        // Number of attention heads (e.g., 8)
    pub num_layers: usize,       // Number of transformer blocks (e.g., 6)
    pub max_seq_len: usize,      // Maximum sequence length (e.g., 1024)
    pub ff_dim: usize,           // Feed-forward dimension (typically 4 * embed_dim)
    pub dropout: f64,            // Dropout probability (e.g., 0.1)
}

impl TransformerConfig {
    /// Create a small transformer config (good for 8GB VRAM like mine)
    pub fn small() -> Self {
        Self {
            vocab_size: 10000,    // Small vocabulary
            embed_dim: 256,       // Small embeddings
            num_heads: 8,
            num_layers: 6,
            max_seq_len: 512,
            ff_dim: 1024,         // 4 * embed_dim
            dropout: 0.1,
        }
    }
    
    /// Create a tiny transformer config (for testing)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            embed_dim: 128,
            num_heads: 4,
            num_layers: 4,
            max_seq_len: 256,
            ff_dim: 512,
            dropout: 0.1,
        }
    }
}