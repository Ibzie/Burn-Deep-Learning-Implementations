use burn::prelude::*;
use burn::nn;

// Multi-head self attention code
// Core algo to allow tokens to understand other tokens in the sequence

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend>{
    num_heads: usize, //Attention Heads
    head_dim: usize, // Dim per each head

    // Lin Projections for Q K V matrices
    query: nn::Linear<B>, //Projects inputs to query
    key: nn::Linear<B>, //Projects inputs to keys
    value: nn::Linear<B>, //Projects input to values
    output: nn::Linear<B>, // Final projection post attention
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(config: &MultiHeadAttentionConfig, device: &B::Device) -> Self {
        let embed_dim = config.embed_dim;
        let num_heads = config.num_heads;

        //Asserting computability for splitting across heads evenly
        assert_eq!(
            embed_dim % num_heads, 0,
            "Embedding dim needs to be divisible by num heads"
        );

        let head_dim = embed_dim / num_heads;

        Self {
            num_heads,
            head_dim,

            // Projections: embed_dim => embed dim

            query: nn::LinearConfig::new(embed_dim, embed_dim).init(device),
            output: nn::LinearConfig::new(embed_dim, embed_dim).init(device),
            key: nn::LinearConfig::new(embed_dim, embed_dim).init(device),
            value: nn::LinearConfig::new(embed_dim, embed_dim).init(device),
        }
    }

    // Forward Pass through the MHA flow
    // # Arg
    // x is the input tensor (batch_size, seq_len, embed_dim)
    // mask is the optional causal mask (seq_len, seq_len) where True means mask out
    // Returns and output tensor: [batch_size, seq_len, embed_dim]

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _embed_dim] = x.dims();

        // Step 1: Project input to Q, K, V
        // [batch, seq, embed_dim] -> [batch, seq, embed_dim]
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // Step 2: Split into multi heads
        // [batch, seq, embed_dim] -> [batch, num_heads, seq, head_dim]
        let q = self.split_heads(q, batch_size, seq_len);
        let k = self.split_heads(k, batch_size, seq_len);
        let v = self.split_heads(v, batch_size, seq_len);

        // Step 3: Compute scaled dot product attention for each head in parallel
        // [batch, num_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        let attention_output = self.scaled_dot_product_attention(q, k, v, mask);

        // Step 4: Concatenation of heads
        // [batch, num_heads, seq, head_dim] -> [batch, seq, embed_dim]
        let attention_output = self.combine_heads(attention_output, batch_size, seq_len);

        // Step 5: Final linear projection
        self.output.forward(attention_output)
    }

    // Split embedding dim into heads
    // [batch, seq, embed_dim] -> [batch, num_heads, seq, head_dim]

    fn split_heads(&self, x:Tensor<B, 3>, batch_size: usize, seq_len: usize) -> Tensor<B, 4>{
        // Reshape: [batch, seq, embed_dim] -> [batch, seq, num_heads, head_dim]
        let x = x.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);

        // Transpose: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        // Allow heads to work independently
        x.swap_dims(1,2)
    }

    // Combine heads back into single embedding dimension
    // [batch, num_heads, seq, head_dim] -> [batch, seq, embed_dim]
    fn combine_heads(&self, x: Tensor<B, 4>, batch_size: usize, seq_len: usize) -> Tensor<B, 3> {
        // Transpose: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim]
        let x = x.swap_dims(1, 2);

        // Reshape: [batch, seq, num_heads, head_dim] -> [batch, seq, embed_dim]
        let embed_dim = self.num_heads * self.head_dim;
        x.reshape([batch_size, seq_len, embed_dim])
    }

    // Scaled Dot Product Attention => softmax(Q @ K^T / sqrt(d_k)) @ V
    fn scaled_dot_product_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 4> {
        // Step 1: Compute attention scores = Q @ K^T
        // Transpose K: [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
        let k_transposed = k.swap_dims(2, 3);
        
        // Matrix multiply: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        //                = [batch, num_heads, seq_len, seq_len]
        // This gives us "how much each token should attend to every other token"
        let mut scores = q.matmul(k_transposed);
        
        // Step 2: Scale by sqrt(head_dim) to prevent softmax saturation
        // Without this, gradients can vanish when head_dim is large
        let scale = (self.head_dim as f64).sqrt();
        scores = scores / scale;
        
        // Step 3: Apply causal mask (for autoregressive generation)
        // Prevents tokens from attending to future tokens
        if let Some(mask) = mask {
            // Broadcast mask from [seq_len, seq_len] to [batch, num_heads, seq_len, seq_len]
            let mask: Tensor<B, 4, Bool> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

            // Set masked positions to -inf so softmax makes them ~0
            scores = scores.mask_fill(mask, -1e9);
        }
        
        // Step 4: Apply softmax to get attention weights
        // [batch, num_heads, seq_len, seq_len]
        // Each row sums to 1.0 - these are the attention probabilities
        let attention_weights = burn::tensor::activation::softmax(scores, 3);
        
        // Step 5: Apply attention weights to values
        // [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        // = [batch, num_heads, seq_len, head_dim]
        // This is the weighted sum of values based on attention
        attention_weights.matmul(v)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    pub embed_dim: usize, // Total embedding dimension
    pub num_heads: usize, // Total num of heads
}
