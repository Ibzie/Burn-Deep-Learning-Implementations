use burn::prelude::*;
use burn::nn;


/// Position-wise Feed-Forward Network
/// Simple two-layer MLP: Linear -> GELU -> Linear
/// This adds non-linearity and allows the model to process each position independently

#[dervice(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> FeedForward<B>{
    pub fn new(config: &FeedForwardConfig, device: &B::Device) -> Self {
        let embed_dim = config.embed_dim;
        let ff_dim = config.ff_dim;

        Self {
            linear1: nn::LinearConfig::new(embed_dim, ff_dim).init(device),
            linear2: nn::LinearConfig::new(embed_dim, ff_dim).init(device),
            dropout: nn::DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Forward pass: x -> Linear -> GELU -> Dropout -> Linear -> Dropout
    /// 
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, embed_dim]
    /// 
    /// # Returns
    /// * Output tensor [batch_size, seq_len, embed_dim]
    
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Step 1: Expand to higher dimension
        // [batch, seq, embed_dim] -> [batch, seq, ff_dim]
        let x = self.linear1.forward(x);
        
        // Step 2: Apply GELU activation (smoother than ReLU)
        // GELU(x) = x * Φ(x), where Φ is the cumulative distribution function of the standard normal distribution
        let x = burn::tensor::activation::gelu(x);
        
        // Step 3: Apply dropout for regularization
        let x = self.dropout.forward(x);
        
        // Step 4: Project back to embed_dim
        // [batch, seq, ff_dim] -> [batch, seq, embed_dim]
        let x = self.linear2.forward(x);
        
        // Step 5: Apply dropout again
        self.dropout.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub embed_dim: usize,  // Embedding dimension (e.g., 512)
    pub ff_dim: usize,     // Feed-forward dimension (typically 4 * embed_dim = 2048)
    pub dropout: f64,      // Dropout probability (e.g., 0.1)
}