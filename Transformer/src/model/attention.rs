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

}
