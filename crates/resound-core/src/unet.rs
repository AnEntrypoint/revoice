use candle_core::{Result, Tensor, D};
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, Module, VarBuilder};

fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu_erf()
}

struct PreactResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
}

impl PreactResBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let groups = (dim / 16).max(1);
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            norm1: group_norm(groups, dim, 1e-5, vb.pp("0"))?,
            conv1: conv2d(dim, dim, 3, cfg, vb.pp("2"))?,
            norm2: group_norm(groups, dim, 1e-5, vb.pp("3"))?,
            conv2: conv2d(dim, dim, 3, cfg, vb.pp("5"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = gelu(&self.norm1.forward(x)?)?;
        let h = self.conv1.forward(&h)?;
        let h = gelu(&self.norm2.forward(&h)?)?;
        let h = self.conv2.forward(&h)?;
        x + h
    }
}

fn resize_nearest(x: &Tensor, scale: f64) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let new_h = ((h as f64) * scale).round() as usize;
    let new_w = ((w as f64) * scale).round() as usize;
    x.upsample_nearest2d(new_h, new_w)
}

struct UNetBlock {
    pre_conv: Conv2d,
    res_block1: PreactResBlock,
    res_block2: PreactResBlock,
    scale_factor: Option<f64>,
}

impl UNetBlock {
    fn new(in_dim: usize, out_dim: usize, scale_factor: Option<f64>, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            pre_conv: conv2d(in_dim, out_dim, 3, cfg, vb.pp("pre_conv"))?,
            res_block1: PreactResBlock::new(out_dim, vb.pp("res_block1"))?,
            res_block2: PreactResBlock::new(out_dim, vb.pp("res_block2"))?,
            scale_factor,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_conv.forward(x)?;
        let x = self.res_block1.forward(&x)?;
        let x = self.res_block2.forward(&x)?;
        match self.scale_factor {
            Some(s) => resize_nearest(&x, s),
            None => Ok(x),
        }
    }
}

pub struct UNet {
    input_proj: Conv2d,
    encoder_blocks: Vec<UNetBlock>,
    middle_blocks: Vec<UNetBlock>,
    decoder_blocks: Vec<UNetBlock>,
    head_conv1: Conv2d,
    head_conv2: Conv2d,
    num_blocks: usize,
}

impl UNet {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        hidden_dim: usize,
        num_blocks: usize,
        num_middle_blocks: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let input_proj = conv2d(input_dim, hidden_dim, 3, cfg3, vb.pp("input_proj"))?;

        let mut encoder_blocks = Vec::with_capacity(num_blocks);
        let mut dim = hidden_dim;
        let enc_vb = vb.pp("encoder_blocks");
        for i in 0..num_blocks {
            let next_dim = dim * 2;
            encoder_blocks.push(UNetBlock::new(dim, next_dim, Some(0.5), enc_vb.pp(i))?);
            dim = next_dim;
        }

        let mut middle_blocks = Vec::with_capacity(num_middle_blocks);
        let mid_vb = vb.pp("middle_blocks");
        for i in 0..num_middle_blocks {
            middle_blocks.push(UNetBlock::new(dim, dim, None, mid_vb.pp(i))?);
        }

        let mut decoder_blocks = Vec::with_capacity(num_blocks);
        let dec_vb = vb.pp("decoder_blocks");
        for i in 0..num_blocks {
            let next_dim = dim / 2;
            decoder_blocks.push(UNetBlock::new(dim, next_dim, Some(2.0), dec_vb.pp(i))?);
            dim = next_dim;
        }

        let head_vb = vb.pp("head");
        let head_conv1 = conv2d(hidden_dim, hidden_dim, 3, cfg3, head_vb.pp("0"))?;
        let cfg1 = Conv2dConfig::default();
        let head_conv2 = conv2d(hidden_dim, output_dim, 1, cfg1, head_vb.pp("2"))?;

        Ok(Self {
            input_proj,
            encoder_blocks,
            middle_blocks,
            decoder_blocks,
            head_conv1,
            head_conv2,
            num_blocks,
        })
    }

    fn pad_to_fit(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        let (_, _, h, w) = x.dims4()?;
        let factor = 1usize << self.num_blocks;
        let pad_h = (factor - h % factor) % factor;
        let pad_w = (factor - w % factor) % factor;
        let x = x.pad_with_zeros(D::Minus2, 0, pad_h)?;
        let x = x.pad_with_zeros(D::Minus1, 0, pad_w)?;
        Ok((x, h, w))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x, orig_h, orig_w) = self.pad_to_fit(x)?;
        let mut h = self.input_proj.forward(&x)?;
        let mut skips = Vec::with_capacity(self.encoder_blocks.len());
        for block in &self.encoder_blocks {
            h = block.forward(&h)?;
            skips.push(h.clone());
        }
        for block in &self.middle_blocks {
            h = block.forward(&h)?;
        }
        for (block, skip) in self.decoder_blocks.iter().zip(skips.iter().rev()) {
            h = (h + skip)?;
            h = block.forward(&h)?;
        }
        let h = gelu(&self.head_conv1.forward(&h)?)?;
        let h = self.head_conv2.forward(&h)?;
        h.narrow(D::Minus2, 0, orig_h)?.narrow(D::Minus1, 0, orig_w)
    }
}
