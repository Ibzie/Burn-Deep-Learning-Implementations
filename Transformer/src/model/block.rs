use burn::prelude::*;
use burn::nn;

use super::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use super::feedforward::{FeedForward, FeedForwardConfig};

/// Single Transformer Block
/// Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
/// This is the "Pre-LN" variant which is more stable for training
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,  // Multi-head self-attention
    feed_forward: FeedForward<B>,       // Position-wise feed-forward network
    norm1: nn::LayerNorm<B>,            // Layer norm before attention
    norm2: nn::LayerNorm<B>,            // Layer norm before FFN
    dropout: nn::Dropout,                // Dropout for residual connections
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &TransformerBlockConfig, device: &B::Device) -> Self {
        let embed_dim = config.embed_dim;
        
        // Create attention config
        let attention_config = MultiHeadAttentionConfig {
            embed_dim,
            num_heads: config.num_heads,
        };
        
        // Create feed-forward config
        let ff_config = FeedForwardConfig {
            embed_dim,
            ff_dim: config.ff_dim,
            dropout: config.dropout,
        };
        
        // Layer norm normalizes across the embedding dimension
        let norm_config = nn::LayerNormConfig::new(embed_dim);
        
        Self {
            attention: MultiHeadAttention::new(&attention_config, device),
            feed_forward: FeedForward::new(&ff_config, device),
            norm1: norm_config.clone().init(device),
            norm2: norm_config.init(device),
            dropout: nn::DropoutConfig::new(config.dropout).init(),
        }
    }
    
    /// Forward pass through transformer block
    /// 
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, embed_dim]
    /// * `mask` - Optional causal mask [seq_len, seq_len]
    /// 
    /// # Returns
    /// * Output tensor [batch_size, seq_len, embed_dim]
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        // Step 1: Self-Attention with residual connection
        // Pre-LN: Normalize BEFORE attention
        let normalized = self.norm1.forward(x.clone());
        
        // Apply attention: [batch, seq, embed_dim] -> [batch, seq, embed_dim]
        let attention_out = self.attention.forward(normalized, mask);
        
        // Apply dropout and add residual connection
        // x_new = x + Dropout(Attention(LayerNorm(x)))
        let attention_out = self.dropout.forward(attention_out);
        let x = x + attention_out;  // Residual connection
        
        // Step 2: Feed-Forward Network with residual connection
        // Pre-LN: Normalize BEFORE feed-forward
        let normalized = self.norm2.forward(x.clone());
        
        // Apply feed-forward: [batch, seq, embed_dim] -> [batch, seq, embed_dim]
        let ff_out = self.feed_forward.forward(normalized);
        
        // Apply dropout and add residual connection
        // x_final = x + Dropout(FFN(LayerNorm(x)))
        let ff_out = self.dropout.forward(ff_out);
        x + ff_out  // Residual connection
    }
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub embed_dim: usize,   // Embedding dimension (e.g., 512)
    pub num_heads: usize,   // Number of attention heads (e.g., 8)
    pub ff_dim: usize,      // Feed-forward dimension (typically 4 * embed_dim)
    pub dropout: f64,       // Dropout probability (e.g., 0.1)
}