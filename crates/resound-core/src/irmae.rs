use candle_core::{Result, Tensor};
use candle_nn::{conv1d, conv1d_no_bias, group_norm, Conv1d, Conv1dConfig, GroupNorm, Module, VarBuilder};

fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu_erf()
}

struct ResBlock1d {
    layers: Vec<(GroupNorm, Conv1d)>,
}

impl ResBlock1d {
    fn new(dim: usize, dilations: &[usize], vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            let norm_idx = i * 3;
            let conv_idx = i * 3 + 2;
            let norm = group_norm(32, dim, 1e-5, vb.pp(norm_idx.to_string()))?;
            let cfg = Conv1dConfig {
                padding: d,
                dilation: d,
                ..Default::default()
            };
            let conv = conv1d(dim, dim, 3, cfg, vb.pp(conv_idx.to_string()))?;
            layers.push((norm, conv));
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (norm, conv) in &self.layers {
            h = gelu(&norm.forward(&h)?)?;
            h = conv.forward(&h)?;
        }
        x + h
    }
}

pub struct IrmaeEncoder {
    stem: Conv1d,
    res_blocks: Vec<ResBlock1d>,
    rank_convs: Vec<Conv1d>,
}

impl IrmaeEncoder {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        num_res: usize,
        num_rank_convs: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let stem = conv1d(input_dim, hidden_dim, 3, cfg, vb.pp("0"))?;
        let mut res_blocks = Vec::with_capacity(num_res);
        for i in 0..num_res {
            res_blocks.push(ResBlock1d::new(hidden_dim, &[1, 2, 4, 8], vb.pp((1 + i).to_string()))?);
        }
        let mut rank_convs = Vec::with_capacity(num_rank_convs);
        let cfg1 = Conv1dConfig::default();
        let mut dim = hidden_dim;
        for i in 0..num_rank_convs {
            let idx = 1 + num_res + i;
            rank_convs.push(conv1d_no_bias(dim, latent_dim, 1, cfg1, vb.pp(idx.to_string()))?);
            dim = latent_dim;
        }
        Ok(Self {
            stem,
            res_blocks,
            rank_convs,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.stem.forward(x)?;
        for block in &self.res_blocks {
            h = block.forward(&h)?;
        }
        for conv in &self.rank_convs {
            h = conv.forward(&h)?;
        }
        h.tanh()
    }
}

pub struct IrmaeDecoder {
    stem: Conv1d,
    res_blocks: Vec<ResBlock1d>,
    out_conv: Conv1d,
}

impl IrmaeDecoder {
    pub fn new(
        latent_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_res: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let stem = conv1d(latent_dim, hidden_dim, 3, cfg, vb.pp("0"))?;
        let mut res_blocks = Vec::with_capacity(num_res);
        for i in 0..num_res {
            res_blocks.push(ResBlock1d::new(hidden_dim, &[1, 2, 4, 8], vb.pp((1 + i).to_string()))?);
        }
        let cfg1 = Conv1dConfig::default();
        let out_idx = 1 + num_res;
        let out_conv = conv1d(hidden_dim, output_dim, 1, cfg1, vb.pp(out_idx.to_string()))?;
        Ok(Self {
            stem,
            res_blocks,
            out_conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.stem.forward(x)?;
        for block in &self.res_blocks {
            h = block.forward(&h)?;
        }
        self.out_conv.forward(&h)
    }
}
